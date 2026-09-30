from ultralytics import YOLO
import os

print("=" * 60)
print("GOAT EYE DETECTION - TRAINING")
print("=" * 60)

# Check data.yaml
if not os.path.exists("data.yaml"):
    print("ERROR: data.yaml not found!")
    exit()

# Load pretrained YOLO model
print("\nLoading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")

# Train the model
print("\nStarting training...")
print("This may take some time.\n")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    workers=2,
    project="runs",
    name="goat_eye_model",
    pretrained=True,
    save=True,
    patience=15,
    verbose=True
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED")
print("=" * 60)

print("\nYour trained model:")
print("runs/goat_eye_model/weights/best.pt")

print("\nCopy best.pt to:")
print("model/best.pt")