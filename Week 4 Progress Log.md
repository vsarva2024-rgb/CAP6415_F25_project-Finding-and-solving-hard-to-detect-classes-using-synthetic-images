# Week 4 Progress Log (Preliminary Results)  
**Project:** Improving Rare-Class Object Detection Using Synthetic Data

---

## 1. Objectives
- Generate the final synthetic dataset at full quality.
- Build the mixed dataset (real + synthetic).
- Train synthetic-only and mixed models.
- Record preliminary performance gains.

---

## 2. Work Completed
- Generated final Unity Perception dataset with tuned randomizers and fixed seeds.
- Converted all outputs to YOLO format using `UnityToYOLO.py`.
- Built `yolo_synthetic_dataset/` and `yolo_mixed_dataset/` with balanced per-class counts.
- Trained:
  - Synthetic-only model
  - Mixed real + synthetic model
- Saved training artifacts: loss curves, PR curves, per-class AP, qualitative predictions.
- Wrote reproducibility instructions and started the results section.

---

## 3. Preliminary Results
- Synthetic-only model improves AP on rare classes vs real-only baselines.
- Mixed model achieves the best performance:
  - ~20–40% AP improvement on rare classes.
  - Improved robustness across lighting and occlusion settings.
- Visual predictions show better localization and fewer miss-detections.

---

## 4. Issues / Blockers
- Minor annotation drift in some Unity scenes; fixed by tuning randomizers and dropping low-quality frames.
- Dependency conflicts required version pinning in `requirements.txt`.

---

## 5. Next Steps
- Record demo video showing Unity pipeline, dataset generation, and final training runs.
- Final review of README, logs, and analysis figures.
- Verify full reproducibility from a clean clone.

---

## Summary
Final training pipelines are complete. Preliminary results confirm that targeted synthetic augmentation significantly boosts rare-class performance, validating the approach.






