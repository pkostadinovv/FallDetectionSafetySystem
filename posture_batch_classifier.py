#!/usr/bin/env python3
"""
Posture Batch Classifier: Standing vs Lying
-------------------------------------------

Scans a folder of images, runs MediaPipe Pose on each image, and classifies
whether the main detected person is likely LYING or STANDING.

Heuristic (configurable):
- Compute the torso vector from hip-midpoint to shoulder-midpoint.
- Measure its angle relative to the horizontal axis.
  * Small angle (close to horizontal)  -> LYING
  * Large angle (close to vertical)    -> STANDING
- Back up with the bounding box aspect ratio from keypoints.

Outputs:
- Prints per-image results to the console.
- Writes a CSV with columns: filename, label, confidence, angle_deg, aspect_ratio, note

Dependencies:
    pip install mediapipe opencv-python numpy pandas
"""

import argparse
import os
import sys
import math
import glob
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as e:
    print("Error: mediapipe is required. Install with: pip install mediapipe")
    raise

@dataclass
class PostureResult:
    filename: str
    label: str                     # 'lying', 'standing', or 'no_person_detected'
    confidence: float              # 0..1 (heuristic)
    angle_deg: Optional[float]     # torso angle vs horizontal (degrees)
    aspect_ratio: Optional[float]  # bbox width / height
    note: str = ""


def angle_degrees_from_horizontal(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Returns the absolute angle (0..90] between the vector p1->p2 and the horizontal axis, in degrees.
    0  -> perfectly horizontal
    90 -> perfectly vertical
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    ang = abs(math.degrees(math.atan2(dy, dx)))
    if ang > 90:
        ang = 180 - ang
    return ang


def compute_bbox_from_landmarks(landmarks_xy: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute (min_x, min_y, max_x, max_y) from visible landmark (x, y) pairs in pixels.
    """
    xs = landmarks_xy[:, 0]
    ys = landmarks_xy[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def extract_keypoints(image_bgr, pose) -> Optional[dict]:
    """
    Runs MediaPipe Pose on a single BGR image and returns a dict with pixel-space keypoints
    and their visibility if a pose is detected; otherwise None.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if not results.pose_landmarks:
        return None

    h, w = image_bgr.shape[:2]
    kp = {}
    for idx, lm in enumerate(results.pose_landmarks.landmark):
        kp[idx] = {
            "x": lm.x * w,
            "y": lm.y * h,
            "z": lm.z,
            "vis": lm.visibility
        }
    return kp


def midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def landmarks_to_numpy(kp: dict, min_vis: float = 0.3) -> np.ndarray:
    """
    Convert visible landmarks to Nx2 numpy array (pixels). Filters by visibility.
    """
    pts = []
    for v in kp.values():
        if v["vis"] >= min_vis:
            pts.append((v["x"], v["y"]))
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    return np.array(pts, dtype=np.float32)


def classify_posture(kp: dict,
                     angle_threshold_deg: float = 30.0,
                     aspect_ratio_threshold: float = 1.20) -> Tuple[str, float, float, float, str]:
    """
    Returns:
        label ('lying'|'standing'),
        confidence (0..1),
        angle_deg (torso vs horizontal),
        aspect_ratio (bbox width/height),
        note (str)
    """
    # MediaPipe indices
    # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # Common ones:
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    # Check required keypoints visibility
    req = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    try:
        pts = {i: (kp[i]["x"], kp[i]["y"]) for i in req if kp[i]["vis"] >= 0.3}
    except KeyError:
        return "standing", 0.0, float("nan"), float("nan"), "missing keypoints"

    if len(pts) < 4:
        # Fallback: not enough for torso; try bbox-only heuristic
        visible = landmarks_to_numpy(kp, min_vis=0.4)
        if visible.shape[0] < 6:
            return "standing", 0.0, float("nan"), float("nan"), "insufficient landmarks"
        minx, miny, maxx, maxy = compute_bbox_from_landmarks(visible)
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        ar = w / h
        label = "lying" if ar > aspect_ratio_threshold else "standing"
        conf = min(1.0, abs(ar - 1.0))  # rough
        return label, conf, float("nan"), ar, "bbox-only heuristic"

    # Torso angle
    sh_mid = midpoint(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
    hip_mid = midpoint(pts[LEFT_HIP], pts[RIGHT_HIP])
    angle_deg = angle_degrees_from_horizontal(hip_mid, sh_mid)

    # Bounding box of visible landmarks (for backup/consistency)
    visible = landmarks_to_numpy(kp, min_vis=0.4)
    if visible.shape[0] >= 4:
        minx, miny, maxx, maxy = compute_bbox_from_landmarks(visible)
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        aspect_ratio = w / h
    else:
        aspect_ratio = float("nan")

    # Decision logic:
    lying_by_angle = angle_deg < angle_threshold_deg
    lying_by_ar = (not math.isnan(aspect_ratio)) and (aspect_ratio > aspect_ratio_threshold)

    if lying_by_angle and lying_by_ar:
        label = "lying"
        # Confidence grows as angle -> 0 and AR grows beyond threshold
        conf = min(1.0, (angle_threshold_deg - angle_deg) / angle_threshold_deg * 0.6
                   + min(1.0, (aspect_ratio - aspect_ratio_threshold) / aspect_ratio_threshold) * 0.6)
        note = "angle+aspect agree"
    elif lying_by_angle:
        label = "lying"
        conf = max(0.5, (angle_threshold_deg - angle_deg) / angle_threshold_deg)
        note = "angle suggests lying"
    elif lying_by_ar:
        label = "lying"
        conf = max(0.5, min(1.0, (aspect_ratio - aspect_ratio_threshold) / aspect_ratio_threshold))
        note = "aspect suggests lying"
    else:
        label = "standing"
        # Confidence grows as angle -> 90 and AR -> 1 or smaller
        ang_conf = min(1.0, (angle_deg / 90.0))
        ar_conf = 1.0 if math.isnan(aspect_ratio) else min(1.0, 1.0 / max(1.0, aspect_ratio))
        conf = max(0.4, 0.5 * ang_conf + 0.5 * ar_conf)
        note = "angle+aspect suggest standing"

    return label, float(conf), float(angle_deg), float(aspect_ratio), note


def process_image(path: str,
                  pose,
                  angle_threshold_deg: float,
                  aspect_ratio_threshold: float) -> PostureResult:
    img = cv2.imread(path)
    if img is None:
        return PostureResult(os.path.basename(path), "no_person_detected", 0.0, None, None, "failed to read image")

    kp = extract_keypoints(img, pose)
    if kp is None:
        return PostureResult(os.path.basename(path), "no_person_detected", 0.0, None, None, "no pose found")

    label, conf, ang, ar, note = classify_posture(
        kp,
        angle_threshold_deg=angle_threshold_deg,
        aspect_ratio_threshold=aspect_ratio_threshold
    )
    return PostureResult(os.path.basename(path), label, conf, ang, ar, note)


def find_images(folder: str) -> List[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
        files.extend(glob.glob(os.path.join(folder, "**", e), recursive=True))
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in files:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Detect LYING vs STANDING in images within a folder.")
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--csv", default="posture_results.csv", help="CSV output filename (default: posture_results.csv)")
    parser.add_argument("--angle-threshold", type=float, default=30.0,
                        help="Torso angle (deg) below which counts as lying (default: 30)")
    parser.add_argument("--aspect-threshold", type=float, default=1.20,
                        help="BBox width/height above which supports lying (default: 1.20)")
    parser.add_argument("--min-det-confidence", type=float, default=0.5,
                        help="MediaPipe detection confidence (default: 0.5)")
    parser.add_argument("--min-track-confidence", type=float, default=0.5,
                        help="MediaPipe tracking confidence (default: 0.5)")
    parser.add_argument("--max-images", type=int, default=0,
                        help="Limit number of images processed (0 = no limit)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: not a folder: {args.folder}")
        sys.exit(1)

    image_paths = find_images(args.folder)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print("No images found.")
        sys.exit(0)

    mp_pose = mp.solutions.pose

    # static_image_mode=True is ideal for single images
    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=args.min_det_confidence,
                      min_tracking_confidence=args.min_track_confidence) as pose:

        results: List[PostureResult] = []
        for i, path in enumerate(image_paths, 1):
            res = process_image(
                path,
                pose,
                angle_threshold_deg=args.angle_threshold,
                aspect_ratio_threshold=args.aspect_threshold
            )
            results.append(res)
            print(f"[{i:04d}/{len(image_paths)}] {res.filename:40s} -> {res.label:9s} "
                  f"(conf={res.confidence:.2f}, angle={res.angle_deg if res.angle_deg is not None else 'NA'}, "
                  f"AR={res.aspect_ratio if res.aspect_ratio is not None else 'NA'}) - {res.note}")

    # Write CSV
    out = args.csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "confidence", "angle_deg", "aspect_ratio", "note"])
        for r in results:
            writer.writerow([
                r.filename,
                r.label,
                f"{r.confidence:.4f}",
                "" if r.angle_deg is None or math.isnan(r.angle_deg) else f"{r.angle_deg:.2f}",
                "" if r.aspect_ratio is None or math.isnan(r.aspect_ratio) else f"{r.aspect_ratio:.3f}",
                r.note
            ])
    print(f"\nSaved results to: {out}")
    print("Done.")


if __name__ == "__main__":
    main()
