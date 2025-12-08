
# **REPRODUCIBILITY GUIDE**

This document describes the steps required to fully reproduce model training, synthetic-data generation, and evaluation. All paths and filenames correspond to the repository structure shown in the project.

---

# **1. Environment Setup**

## **1.1 Create virtual environment**

```bash
python -m venv .venv
```

## **1.2 Activate**

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

## **1.3 Install dependencies**

```bash
pip install -r requirements.txt
```

If a requirements file is not used:

```bash
pip install ultralytics pandas numpy scikit-learn tqdm plotly pillow matplotlib
```

---

# **2. Unity Synthetic Dataset Generation**

## **2.1 Open the Unity project**

Open:

```
1. unity_project/
```

## **2.2 Prefab assets**

Book prefabs are located at:

```
Assets/3dModel/book1.prefab
...
Assets/3dModel/book9.prefab
```

## **2.3 Scene configuration**

Load the ``SyntheticDataScene.unity`` scene used for Perception dataset collection.
Ensure the Perception Camera, labelers, and randomizers are active.

## **2.4 Synthetic capture**

Press **Play** in Unity.
Synthetic BOOK images are written to:

```
1. unity_project/GeneratedDataset/
```

## **2.5 Convert Unity output to YOLO format**

Run the provided script:

```
Perception2YOLO
```

This produces a YOLO-formatted dataset:

```
GeneratedDataset/YOLO/
   images/
   labels/
   dataset.yaml
```

## **2.6 Move dataset into training directory**

Copy the converted dataset into:

```
2. SOTA Training/datasets/train_dataset/
```

Final structure:

```
train_dataset/
   images/train/
   images/val/
   labels/train/
   labels/val/
   dataset.yaml
```

---

# **3. Training Pipeline**

All training is performed using:

```
2. SOTA Training/Training.ipynb
```

## **3.1 Configure dataset**

Inside the notebook:

```python
DATASET_ROOT = '2. SOTA Training/datasets/train_dataset'
DATA_YAML = f"{DATASET_ROOT}/dataset.yaml"
MODEL = 'yolov8n.pt'
NUM_EPOCHS = 20
BATCH = 16
IMG_SIZE = 512
DEVICE = 'cpu'
DO_TRAIN = True
```

## **3.2 Run dataset sanity check**

Execute the dataset visualization cell.
A set of samples with bounding boxes will be displayed.

## **3.3 Train**

The notebook executes:

```python
from ultralytics import YOLO
model = YOLO(MODEL)

model.train(
    data=DATA_YAML,
    epochs=NUM_EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=DEVICE,
    project="runs/detect",
    name="training_run"
)
```

## **3.4 Training output**

```
runs/detect/training_run/weights/best.pt
```

Copy this file to:

```
3. Test & Results/Testing/TrainedModel.pt
```

---

# **4. Evaluation on Real Dataset**

Evaluation scripts are located in:

```
3. Test & Results/Testing/
```

Real dataset structure:

```
real_dataset/
   book/
   bottle/
   chair/
   cup/
   laptop/
```

---

## **4.1 Evaluate Pure SOTA Model**

```bash
python "3. Test & Results/Testing/SOTA_test.py" \
  --model "yolov8n.pt" \
  --data "3. Test & Results/Testing/real_dataset" \
  --out "3. Test & Results/Testing/SOTA_eval_results" \
  --device cpu \
  --imgsz 512
```

Outputs:

```
SOTA_eval_results/
   results.csv
   per_class_metrics.csv
   confusion_matrix.html
   summary.json
```

---

## **4.2 Evaluate Trained Model**

```bash
python "3. Test & Results/Testing/TRAINING_test.py" \
  --model "3. Test & Results/Testing/TrainedModel.pt" \
  --data "3. Test & Results/Testing/real_dataset" \
  --out "3. Test & Results/Testing/Training_eval_results" \
  --device cpu \
  --imgsz 512
```

Outputs:

```
Training_eval_results/
   results.csv
   per_class_metrics.csv
   confusion_matrix.html
   summary.json
```

---

# **5. Directory Summary**

```
1. unity_project/
   ├─ Assets/3dModel/
   └─ GeneratedDataset/
        └─ YOLO/

2. SOTA Training/
   ├─ datasets/
   │   └─ train_dataset/
   ├─ Training.ipynb
   └─ yolov8n.pt

3. Test & Results/
   └─ Testing/
       ├─ real_dataset/
       ├─ SOTA_test.py
       ├─ TRAINING_test.py
       └─ TrainedModel.pt
```

---

If you want a **PDF-ready** version (LaTeX compiled formatting) or a **Markdown file** to place directly in the repo, tell me which one and I’ll generate it instantly.
