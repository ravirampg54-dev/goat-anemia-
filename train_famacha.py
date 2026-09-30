from ultralytics import YOLO
import os

DATASET = "dataset_anemia"

if not os.path.exists(DATASET):
    print("ERROR: dataset_anemia folder not found!")
    exit()

print("=" * 60)
print("FAMACHA 1-4 CLASSIFICATION TRAINING")
print("=" * 60)

model = YOLO("yolo11n-cls.pt")

model.train(
    data=DATASET,
    epochs=50,
    imgsz=224,
    batch=8,
    device="cpu",
    workers=2,
    project="runs",
    name="famacha_classification",
    exist_ok=True,
    patience=10,
    save=True,
    pretrained=True,
    verbose=True
)

print("=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)
