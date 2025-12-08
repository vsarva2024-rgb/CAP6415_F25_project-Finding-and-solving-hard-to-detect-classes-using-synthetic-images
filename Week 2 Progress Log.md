

# Week 2 Progress Log  
**Project:** Improving Rare-Class Object Detection Using Synthetic Data

---

## 1. Objectives
- Build the Unity JSON → YOLO TXT conversion system.
- Prepare a small COCO subset for baseline training.
- Validate the pipeline using placeholder synthetic data.

---

## 2. Work Completed
- Implemented conversion logic for Unity Perception JSON files.
- Built `download_coco_subset.py` to extract a consistent set of COCO classes.
- Generated ~250 synthetic and ~250 placeholder real images for testing.
- Added basic YOLOv8 `train.py` and `evaluate.py` scripts.
- Organized datasets under `datasets/real_subset/` and `datasets/synthetic_test/`.

---

## 3. Issues / Blockers
- COCO must be obtained manually (Kaggle).
- Placeholder synthetic data lacks diversity but works for pipeline verification.

---

## 4. Next Steps
- Download full COCO subset and replace placeholders.
- Begin generating high-quality Unity captures.
- Start recording baseline metrics.

---

## Summary
Annotation conversion and dataset-loading pipeline are working end-to-end. Ready to switch to real data and larger Unity exports.


