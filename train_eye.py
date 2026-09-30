from ultralytics import YOLO
import os

DATASET = "dataset/data.yaml"

if not os.path.exists(DATASET):
    print("ERROR: dataset/data.yaml not found!")
    exit()

print("=" * 60)
print("GOAT EYE DETECTION MODEL TRAINING")
print("=" * 60)

print("Dataset:", DATASET)
print("Device: CPU")
print("Model: YOLO11n")

model = YOLO("yolo11n.pt")

model.train(
    data=DATASET,
    epochs=50,
    imgsz=640,
    batch=2,
    device="cpu",
    workers=2,
    project="runs",
    name="goat_eye_detection",
    exist_ok=True,
    patience=10,
    save=True,
    pretrained=True,
    verbose=True
)

print()
print("=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print()
print("Best model:")
print("runs/detect/goat_eye_detection/weights/best.pt")