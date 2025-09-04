#!/usr/bin/env python3
"""
Posture Batch Classifier: Standing vs Lying (with annotated outputs)
-------------------------------------------------------------------

Scans a folder of images, runs MediaPipe Pose on each image, classifies
whether the main person is LYING or STANDING, and writes:

1) A CSV report.
2) An annotated copy of each image with:
   - Bounding box around visible pose landmarks
   - (Optional) Pose landmarks + torso line
   - Label and confidence text

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
except Exception:
    print("Error: mediapipe is required. Install with: pip install mediapipe")
    raise

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


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


def compute_bbox_from_landmarks(landmarks_xy: np.ndarray, pad: float = 0.02, img_w: int = None, img_h: int = None) -> Tuple[int, int, int, int]:
    """
    Compute integer (x1, y1, x2, y2) from visible landmark (x, y) pairs in pixels.
    Adds a small padding (fraction of image dimension) if img_w/h provided.
    """
    xs = landmarks_xy[:, 0]
    ys = landmarks_xy[:, 1]
    minx, miny, maxx, maxy = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    if img_w is not None and img_h is not None:
        pad_x = pad * img_w
        pad_y = pad * img_h
        minx -= pad_x; miny -= pad_y; maxx += pad_x; maxy += pad_y
        minx = max(0, minx); miny = max(0, miny)
        maxx = min(img_w - 1, maxx); maxy = min(img_h - 1, maxy)

    return int(round(minx)), int(round(miny)), int(round(maxx)), int(round(maxy))


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
    return kp, results.pose_landmarks


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
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_HIP, RIGHT_HIP = 23, 24

    # Check required keypoints visibility
    req = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]
    try:
        pts = {i: (kp[i]["x"], kp[i]["y"]) for i in req if kp[i]["vis"] >= 0.3}
    except KeyError:
        return "standing", 0.0, float("nan"), float("nan"), "missing keypoints"

    if len(pts) < 4:
        visible = landmarks_to_numpy(kp, min_vis=0.4)
        if visible.shape[0] < 6:
            return "no_person_detected", 0.0, float("nan"), float("nan"), "insufficient landmarks"
        minx, miny, maxx, maxy = visible[:,0].min(), visible[:,1].min(), visible[:,0].max(), visible[:,1].max()
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        ar = w / h
        label = "lying" if ar > aspect_ratio_threshold else "standing"
        conf = min(1.0, abs(ar - 1.0))
        return label, conf, float("nan"), ar, "bbox-only heuristic"

    sh_mid = midpoint(pts[LEFT_SHOULDER], pts[RIGHT_SHOULDER])
    hip_mid = midpoint(pts[LEFT_HIP], pts[RIGHT_HIP])
    angle_deg = angle_degrees_from_horizontal(hip_mid, sh_mid)

    visible = landmarks_to_numpy(kp, min_vis=0.4)
    if visible.shape[0] >= 4:
        minx, miny, maxx, maxy = visible[:,0].min(), visible[:,1].min(), visible[:,0].max(), visible[:,1].max()
        w = max(1.0, maxx - minx)
        h = max(1.0, maxy - miny)
        aspect_ratio = w / h
    else:
        aspect_ratio = float("nan")

    lying_by_angle = angle_deg < angle_threshold_deg
    lying_by_ar = (not math.isnan(aspect_ratio)) and (aspect_ratio > aspect_ratio_threshold)

    if lying_by_angle and lying_by_ar:
        label = "lying"
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
        ang_conf = min(1.0, (angle_deg / 90.0))
        ar_conf = 1.0 if math.isnan(aspect_ratio) else min(1.0, 1.0 / max(1.0, aspect_ratio))
        conf = max(0.4, 0.5 * ang_conf + 0.5 * ar_conf)
        note = "angle+aspect suggest standing"

    return label, float(conf), float(angle_deg), float(aspect_ratio), note


def draw_annotations(
    img: np.ndarray,
    kp: Optional[dict],
    pose_landmarks,
    label: str,
    confidence: float,
    angle_deg: Optional[float],
    bbox_xyxy: Optional[Tuple[int, int, int, int]],
    draw_landmarks: bool = True,
    draw_torso: bool = True
) -> np.ndarray:
    """Draw bbox, label, and optional pose landmarks/torso line on a copy of the image."""
    out = img.copy()

    # Bounding box
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

    # Landmarks
    if draw_landmarks and pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            out,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )

    # Torso line
    if draw_torso and kp is not None:
        try:
            L_SH, R_SH, L_HIP, R_HIP = 11, 12, 23, 24
            if min(kp[L_SH]["vis"], kp[R_SH]["vis"], kp[L_HIP]["vis"], kp[R_HIP]["vis"]) >= 0.3:
                sh_mid = (int(round((kp[L_SH]["x"] + kp[R_SH]["x"]) / 2)),
                          int(round((kp[L_SH]["y"] + kp[R_SH]["y"]) / 2)))
                hip_mid = (int(round((kp[L_HIP]["x"] + kp[R_HIP]["x"]) / 2)),
                           int(round((kp[L_HIP]["y"] + kp[R_HIP]["y"]) / 2)))
                cv2.line(out, hip_mid, sh_mid, (255, 0, 0), 2)
                cv2.circle(out, hip_mid, 4, (255, 0, 0), -1)
                cv2.circle(out, sh_mid, 4, (255, 0, 0), -1)
        except Exception:
            pass

    # Label text
    text = f"{label.upper()}  (conf {confidence:.2f}"
    if angle_deg is not None and not math.isnan(angle_deg):
        text += f", angle {angle_deg:.1f}°"
    text += ")"

    # Background box for readability
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    pad = 6
    cv2.rectangle(out, (10, 10), (10 + tw + 2*pad, 10 + th + baseline + 2*pad), (0, 0, 0), -1)
    cv2.putText(out, text, (10 + pad, 10 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return out


def process_image(path: str,
                  pose,
                  angle_threshold_deg: float,
                  aspect_ratio_threshold: float,
                  draw_landmarks: bool) -> Tuple[PostureResult, Optional[np.ndarray]]:
    img = cv2.imread(path)
    if img is None:
        res = PostureResult(os.path.basename(path), "no_person_detected", 0.0, None, None, "failed to read image")
        return res, None

    extracted = extract_keypoints(img, pose)
    if extracted is None:
        res = PostureResult(os.path.basename(path), "no_person_detected", 0.0, None, None, "no pose found")
        # Draw a simple banner indicating no person
        annotated = img.copy()
        msg = "NO PERSON DETECTED"
        (tw, th), base = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.rectangle(annotated, (10, 10), (10 + tw + 20, 10 + th + base + 20), (0, 0, 255), -1)
        cv2.putText(annotated, msg, (20, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        return res, annotated

    kp, pose_landmarks = extracted

    label, conf, ang, ar, note = classify_posture(
        kp,
        angle_threshold_deg=angle_threshold_deg,
        aspect_ratio_threshold=aspect_ratio_threshold
    )
    res = PostureResult(os.path.basename(path), label, conf, ang, ar, note)

    # Bounding box from visible keypoints
    visible = landmarks_to_numpy(kp, min_vis=0.4)
    bbox_xyxy = None
    if visible.shape[0] >= 4:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = compute_bbox_from_landmarks(visible, pad=0.02, img_w=w, img_h=h)
        bbox_xyxy = (x1, y1, x2, y2)

    annotated = draw_annotations(
        img=img,
        kp=kp,
        pose_landmarks=pose_landmarks,
        label=label,
        confidence=conf,
        angle_deg=ang,
        bbox_xyxy=bbox_xyxy,
        draw_landmarks=draw_landmarks,
        draw_torso=True
    )

    return res, annotated


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
    parser = argparse.ArgumentParser(description="Detect LYING vs STANDING in images within a folder and save annotated copies.")
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--csv", default="posture_results.csv", help="CSV output filename (default: posture_results.csv)")
    parser.add_argument("--out-dir", default="annotated", help="Directory to save annotated images (default: annotated/)")
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
    parser.add_argument("--no-landmarks", action="store_true",
                        help="Do not draw pose landmarks (still draws bbox + label)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: not a folder: {args.folder}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = find_images(args.folder)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        print("No images found.")
        sys.exit(0)

    with mp_pose.Pose(static_image_mode=True,
                      model_complexity=1,
                      enable_segmentation=False,
                      min_detection_confidence=args.min_det_confidence,
                      min_tracking_confidence=args.min_track_confidence) as pose:

        results: List[PostureResult] = []
        for i, path in enumerate(image_paths, 1):
            res, annotated = process_image(
                path,
                pose,
                angle_threshold_deg=args.angle_threshold,
                aspect_ratio_threshold=args.aspect_threshold,
                draw_landmarks=(not args.no_landmarks),
            )
            results.append(res)

            # Save annotated image next to out-dir with same filename
            out_path = os.path.join(args.out_dir, os.path.basename(path))
            if annotated is not None:
                cv2.imwrite(out_path, annotated)

            print(f"[{i:04d}/{len(image_paths)}] {res.filename:40s} -> {res.label:9s} "
                  f"(conf={res.confidence:.2f}, angle={res.angle_deg if res.angle_deg is not None else 'NA'}, "
                  f"AR={res.aspect_ratio if res.aspect_ratio is not None else 'NA'}) - saved: {out_path}")

    # Write CSV
    out_csv = args.csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
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
    print(f"\nSaved results to: {out_csv}")
    print("Annotated images saved to:", os.path.abspath(args.out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
