# Week 3 Progress Log  
**Project:** Improving Rare-Class Object Detection Using Synthetic Data

---

## 1. Objectives
- Replace placeholder data with final COCO subset and Unity-generated images.
- Stabilize the Unity → YOLO conversion pipeline.
- Execute first full synthetic-only training run.

---

## 2. Work Completed
- Standardized Unity export format (`sequence.* / step0.camera.png` + JSON metadata).
- Implemented a reliable `UnityToYOLO.py` converter:
  - Parses JSON
  - Normalizes bounding boxes
  - Handles empty or partial annotations
  - Creates dataset YAML and class files
- Generated a balanced synthetic dataset for target classes.
- Ran synthetic-only YOLOv8 training to validate labels, losses, and predictions.

---

## 3. Issues / Blockers
- A few Unity frames had invalid boxes; added quality filtering.
- Some class names required canonical mapping.

---

## 4. Next Steps
- Regenerate any missing classes for proper balance.
- Complete real COCO subset conversion.
- Build the mixed dataset and begin mixed-model training.
- Standardize metric logging for later comparison.

---

## Summary
Conversion pipeline and synthetic-only training are validated. Ready to scale data volume and progress into mixed-data experiments.


