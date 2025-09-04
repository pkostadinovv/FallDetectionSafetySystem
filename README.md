# Fall Detection System/Posture Classifier: Standing vs Lying

This system scans images and detects key joints in the human body by utilizing [MediaPipe](https://developers.google.com/mediapipe) – an open-source, cross-platform framework by Google. Then it applies geometric heuristics (torso angle and bounding box aspect ratio) to classify whether a person is **lying down** or **standing**. The system outputs both a **CSV report** and **annotated images** with bounding boxes and labels.

---

## Features
- Detects human body keypoints from static images.
- Classifies **standing** vs **lying** posture.
- Generates annotated copies of images with bounding boxes and classification labels.
- Exports a CSV file with details (filename, label, confidence, angle, aspect ratio).
- Works on an entire folder of images at once.
- Easy to extend (e.g., add "sitting" or train a machine learning classifier).

---

## Example Output
Input image → Annotated output:

## Requirements:

Python 3.8+

MediaPipe

OpenCV

NumPy

Pandas

## Usage

Place your test images in a folder, e.g. ./images/.

Run the script:

python posture_batch_classifier.py ./images --out annotated --csv results.csv

## Options

--out : Folder to save annotated images (default: annotated/).

--csv : Filename for CSV results (default: posture_results.csv).

--angle-threshold : Angle (deg) below which torso counts as lying (default: 30).

--aspect-threshold : Bounding box width/height ratio above which suggests lying (default: 1.20).

--max-images : Limit number of images processed (default: 0 = no limit).

## Output

Annotated images → saved in the folder specified by --out.

CSV report → includes:

filename

posture label (standing / lying / no_person_detected)

confidence (0–1)

torso angle (degrees)

aspect ratio (width ÷ height)

notes (e.g., angle+aspect agree)

# How It Works

Pose Estimation

MediaPipe Pose detects 33 body landmarks from each image.

Feature Extraction

Computes torso angle (shoulder–hip line vs horizontal).

Computes bounding box aspect ratio from visible landmarks.

## Classification

If torso is nearly horizontal OR bbox is wide → label = lying.

Otherwise → label = standing.

Confidence is derived from how strongly the measurements match each posture.

# Future Work

Add sitting posture classification.

Support multi-person detection.

Replace heuristic rules with a trained machine learning classifier (SVM, neural net).

Real-time webcam/video processing mode.
