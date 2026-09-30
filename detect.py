from ultralytics import YOLO
import cv2
import os
import glob

# ==================================================
# SETTINGS
# ==================================================

MODEL_PATH = "model/best.pt"
TEST_FOLDER = "test_images"
OUTPUT_FOLDER = "results"

CONFIDENCE = 0.40

# ==================================================
# CREATE OUTPUT FOLDER
# ==================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ==================================================
# CHECK MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    print("ERROR: best.pt not found!")
    print("Expected location:")
    print(MODEL_PATH)
    exit()

# ==================================================
# FIND TEST IMAGE AUTOMATICALLY
# ==================================================

image_extensions = [
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.JPG",
    "*.JPEG",
    "*.PNG"
]

images = []

for extension in image_extensions:
    images.extend(
        glob.glob(
            os.path.join(TEST_FOLDER, extension)
        )
    )

if len(images) == 0:
    print("ERROR: No test image found!")
    print()
    print("Put a goat image inside:")
    print(TEST_FOLDER)
    exit()

# Use the first image found
IMAGE_PATH = images[0]

print("=" * 60)
print("GOAT EYE DETECTION")
print("=" * 60)

print("\nTest image:")
print(IMAGE_PATH)

# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading trained model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")

# ==================================================
# READ IMAGE
# ==================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not read image!")
    exit()

# ==================================================
# RUN YOLO DETECTION
# ==================================================

print("\nDetecting goat eye...")

results = model.predict(
    source=image,
    conf=CONFIDENCE,
    imgsz=640,
    verbose=False
)

# ==================================================
# VARIABLES
# ==================================================

eye_detected = False
eye_count = 0
highest_confidence = 0.0

# ==================================================
# PROCESS DETECTIONS
# ==================================================

for result in results:

    for box in result.boxes:

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Class name
        class_name = model.names[class_id]

        # ------------------------------------------
        # GOAT EYE DETECTED
        # ------------------------------------------

        if class_name == "goat_eye":

            eye_detected = True
            eye_count += 1

            if confidence > highest_confidence:
                highest_confidence = confidence

            # Draw bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            # Label
            label = f"GOAT EYE {confidence * 100:.1f}%"

            # Text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )

            # Label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 15),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )

            # Label text
            cv2.putText(
                image,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2
            )

# ==================================================
# RESULT
# ==================================================

print("\n" + "=" * 60)

if eye_detected:

    print("RESULT: GOAT EYE DETECTED")
    print(f"Number of eyes detected: {eye_count}")
    print(
        f"Highest confidence: "
        f"{highest_confidence * 100:.2f}%"
    )

    status = (
        f"GOAT EYE DETECTED "
        f"{highest_confidence * 100:.1f}%"
    )

    status_color = (0, 255, 0)

else:

    print("RESULT: GOAT EYE NOT DETECTED")

    status = "GOAT EYE NOT DETECTED"

    status_color = (0, 0, 255)

print("=" * 60)

# ==================================================
# ADD RESULT TO IMAGE
# ==================================================

cv2.putText(
    image,
    status,
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    status_color,
    2
)

# ==================================================
# SAVE OUTPUT
# ==================================================

output_path = os.path.join(
    OUTPUT_FOLDER,
    "goat_eye_result.jpg"
)

cv2.imwrite(
    output_path,
    image
)

print("\nAnnotated image saved to:")
print(output_path)

# ==================================================
# DISPLAY
# ==================================================

cv2.imshow(
    "Goat Eye Detection",
    image
)

print("\nPress any key to close the image.")

cv2.waitKey(0)
cv2.destroyAllWindows()