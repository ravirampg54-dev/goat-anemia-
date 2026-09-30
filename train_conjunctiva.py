from ultralytics import YOLO
import os

DATASET = "dataset_conjunctiva/data.yaml"

if not os.path.exists(DATASET):
    print("ERROR: dataset_conjunctiva/data.yaml not found!")
    exit()

print("=" * 60)
print("AUTOMATIC CONJUNCTIVA DETECTION TRAINING")
print("=" * 60)

model = YOLO("yolo11n.pt")

model.train(
    data=DATASET,
    epochs=50,
    imgsz=640,
    batch=2,
    device="cpu",
    workers=2,
    project="runs",
    name="conjunctiva_detection",
    exist_ok=True,
    patience=10,
    save=True,
    pretrained=True,
    verbose=True
)

print()
print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
print()
print("Best model:")
print("runs/detect/runs/conjunctiva_detection/weights/best.pt")