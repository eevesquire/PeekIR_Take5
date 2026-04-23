# IR Gun Detection Pipeline (YOLO)

This folder contains a full train/infer workflow for IR gun detection with bounding boxes.

## 1) Install dependencies

```bash
pip install -r weapon_detection/requirements-weapon-detection.txt
```

## 2) Annotate gun images (required for real boxes)

You should draw boxes around visible guns in images from:

- `weapon_detection/ir_gun/`

### Fast built-in labeling workflow (recommended)

```bash
python weapon_detection/label_ir_gun_boxes.py
```

Mouse/keyboard controls:

- Left-click + drag: draw a gun box
- `u`: undo last box
- `c`: clear all boxes on current image
- `s`: save current image labels
- `n`: save + next image
- `p`: save + previous image
- `q`: quit

This writes YOLO label files next to images (same basename, `.txt`).

### Alternative labeling tools

You can also use tools that export **YOLO txt** labels (class id + normalized bbox):

- CVAT
- Roboflow
- labelImg (YOLO mode)

Keep labels next to images with the same basename:

- `ir_20260422_005917_131.jpg`
- `ir_20260422_005917_131.txt`

For `ir_nogun/`, no boxes are needed; empty labels are generated automatically.

## 3) Build YOLO dataset structure

```bash
python weapon_detection/prepare_ir_yolo_dataset.py
```

This creates:

- `weapon_detection/dataset/images/{train,val,test}`
- `weapon_detection/dataset/labels/{train,val,test}`
- `weapon_detection/dataset/data.yaml`

## 4) Train detector

```bash
python weapon_detection/train_ir_gun_detector.py --epochs 100 --imgsz 640
```

Model weights are saved in:

- `weapon_detection/runs/detect/.../weights/best.pt`

## 5) Inference (gun/no-gun + bounding boxes)

```bash
python weapon_detection/infer_ir_gun_detector.py ^
  --weights weapon_detection/runs/detect/ir_gun_detector/weights/best.pt ^
  --source weapon_detection/ir_gun ^
  --conf 0.25
```

Outputs:

- Console per image: `GUN` or `NO_GUN` + bbox confidence
- Annotated images saved under `weapon_detection/inference_outputs/`

## Notes

- If gun images are not labeled yet, dataset prep will report missing label files and skip those images.
- You can start by labeling a smaller subset, train quickly, and iterate.
