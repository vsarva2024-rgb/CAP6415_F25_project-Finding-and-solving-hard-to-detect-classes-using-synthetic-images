# CAP6415_F25_project-Finding-and-solving-hard-to-detect-classes-using-synthetic-images
CAP6415_F25_project-Finding and solving hard-to-detect classes using synthetic images


Abstract

# 🚀 Training a Vision Model Using Unity-Generated Synthetic Data

**Improving Under-Represented Classes with Synthetic Book Images**

---

## 📄 Overview

This project investigates whether **Unity-generated synthetic images** can reliably improve performance for **under-represented classes** when training YOLOv8.

A real-world dataset was available, but the **“book”** class had very few examples.
To address this, **100 synthetic book images** were generated inside Unity, converted to YOLO format, and **merged into the real training dataset**.

Two models are compared:

| Model               | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| **Pure SOTA Model** | Official YOLOv8n pretrained on COCO (no custom training)          |
| **Trained Model**   | Fine-tuned using the real dataset **+ 100 synthetic book images** |

A custom evaluation pipeline measures improvement in:

* precision
* recall
* F1
* overall accuracy
* per-class behaviour and error trends

An **interactive HTML performance dashboard** summarizes all results.
Source: 

---

# 🧠 Key Components

| Component                           | Purpose                                              |
| ----------------------------------- | ---------------------------------------------------- |
| **Unity Perception**                | Synthetic image generation & 2D bbox labeling        |
| **Perception2YOLO**                 | Conversion of Unity metadata → YOLO format           |
| **YOLOv8 (Ultralytics)**            | Training and inference                               |
| **SOTA_test.py / TRAINING_test.py** | Real-dataset evaluation                              |
| **model_comparison.html**           | Interactive analysis dashboard comparing both models |

---

# 📁 Repository Structure

```
1. unity_project/
   ├─ Assets/
   └─ GeneratedDataset/
        └─ YOLO/                      # created by Perception2YOLO

2. SOTA Training/
   ├─ datasets/
   │   ├─ real_dataset/               # real evaluation images
   │   ├─ TrainingDataset(NonMixed).zip
   │   │     # contains REAL images for all classes
   │   │     # user must paste their YOLO-converted synthetic images here
   │   └─ train_dataset/              # final merged dataset used for training
   ├─ Training.ipynb
   ├─ yolov8n.pt
   └─ runs/                           # YOLO training outputs

3. Test & Results/
   └─ Testing/
       ├─ real_dataset/
       ├─ SOTA_test.py                # Pure SOTA Model evaluation
       ├─ TRAINING_test.py            # Trained Model evaluation
       └─ TrainedModel.pt             # produced by Training.ipynb

model_comparison.html                 # Interactive analysis dashboard
```

---

# 🎲 Unity Synthetic Dataset Generation

### Scene Used

```
Assets/Scenes/SyntheticDataScene.unity
```

### Randomizers (Final Version)

* **PrefabPlacementRandomizer**
* **RotationRandomizer**

No lighting or camera randomizers are used — controlled environment for reproducibility.

### Export location

```
unity_project/GeneratedDataset/
```

Run **Perception2YOLO** to convert into YOLO format:

```
GeneratedDataset/YOLO/
   images/train/
   images/val/
   labels/train/
   labels/val/
   dataset.yaml
```

---

# 📦 Training Dataset (Real + 100 Synthetic Images)

You are provided:

```
TrainingDataset(NonMixed).zip
```

This zip contains **real images for all 5 classes**, including “book”, but the synthetic set must be added manually.

### **How to use it**

1. Unzip:

   ```
   2. SOTA Training/datasets/TrainingDataset(NonMixed)/
   ```
2. Generate synthetic book images → use Perception2YOLO.
3. Paste generated:

   ```
   images/train/   (real + 100 synthetic book)
   labels/train/
   ```

   into the TrainingDataset folder.
4. Rename it as the final training dataset:

   ```
   train_dataset/
   ```

This merged dataset is used for YOLO training.

---

# 🏋️ Training (YOLOv8)

Run:

```
2. SOTA Training/Training.ipynb
```

Set:

```python
DATASET_ROOT = "2. SOTA Training/datasets/train_dataset"
DATA_YAML = f"{DATASET_ROOT}/dataset.yaml"
MODEL = "yolov8n.pt"
```

Outputs:

```
runs/detect/training_run/
   ├─ weights/best.pt
   ├─ results.csv
   ├─ results.png
```

Rename:

```
best.pt → 3. Test & Results/Testing/TrainedModel.pt
```

---

# 📊 Evaluation Pipeline

Located in:

```
3. Test & Results/Testing/
```

### Pure SOTA Model

```bash
python SOTA_test.py
```

### Trained Model (real + synthetic)

```bash
python TRAINING_test.py
```

### Outputs

```
<OUT_DIR>/
   results.csv
   per_class_metrics.csv
   confusion_matrix.html
   summary.json
```

---

# 📈 Final Results (Visual + Interactive)

An interactive dashboard summarizing both models is included:

**`model_comparison.html`**
Contains:

* side-by-side per-class precision, recall, F1
* radar charts
* grouped comparison bars
* F1 deltas
* confusion matrix trends
* raw numeric results
* CSV export

📎 **Cited source:** 

---

# 🧪 Findings (Short Summary)

### **1. Synthetic data dramatically fixes the “book” class.**

Recall increased from **8.8% → 79.4%**, F1 from **0.16 → 0.87**.

### **2. Overall accuracy improves substantially.**

* Pure SOTA Model: **0.681**
* Trained Model: **0.960**

### **3. Cross-class generalization improves.**

Even classes not augmented through synthetic data (cup, chair, laptop) show measurable recall improvements.

### **4. Synthetic variance bridges the domain gap.**

Unity-generated images provided the missing texture/pose diversity needed for the model to properly learn “book”.

---

# 🏁 Conclusion

* Synthetic Unity data is a **practical and effective** method to improve weak classes in real-world datasets.
* Even **100 synthetic images** produced large, measurable gains.
* The Trained Model consistently outperforms the Pure SOTA baseline across all metrics.
* The included dashboard provides an in-depth, visual understanding of the improvements.

---

# 🔗 Files Included

* `model_comparison.html` – Full interactive analytics 
* `TrainingDataset(NonMixed).zip` – real-image training base
* `Perception2YOLO` – converter script
* `SOTA_test.py` / `TRAINING_test.py` – evaluation scripts
* `Training.ipynb` – training workflow


