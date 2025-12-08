# Week 1 Progress Log  
**Project:** Improving Rare-Class Object Detection Using Synthetic Data

---

## 1. Objectives
- Set up codebase and repository.
- Understand COCO class imbalance and identify low-frequency categories.
- Build initial Unity Perception scene and verify export pipeline.
- Validate that Unity → JSON annotations and YOLO training can coexist in the same workflow.

---

## 2. Work Completed
- Created GitHub repository with folders for Unity, Python scripts, and datasets.
- Installed Unity Hub, Unity Editor, and the Perception package.
- Built a minimal Perception scene and generated ~100 test frames.
- Explored COCO dataset and identified target rare classes.
- Set up YOLOv8 environment and ran a smoke test on a tiny toy dataset.

---

## 3. Challenges
- Unity Perception outputs JSON that must be converted to YOLO TXT format.
- Large Unity captures consume storage quickly; added cleanup rules and compression workflows.

---

## 4. Next Steps
- Implement JSON → YOLO converter.
- Prepare real-image dataset subset for baselines.
- Start drafting AP/mAP evaluation scripts.

---

## Summary
Core environment and repo scaffolding are complete. Unity → YOLO integration will be the main focus next week.

