from ultralytics import YOLO
import cv2
import os

MODEL = "model/conjunctiva_best.pt"
INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results/conjunctiva_test"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL)

print("=" * 60)
print("CONJUNCTIVA MODEL TEST")
print("=" * 60)
print("Model:", MODEL)
print("Classes:", model.names)

images = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

print("Images found:", len(images))
print()

for filename in images:

    input_path = os.path.join(INPUT_FOLDER, filename)
    image = cv2.imread(input_path)

    if image is None:
        print("Could not read:", filename)
        continue

    results = model.predict(
        source=image,
        conf=0.10,
        device="cpu",
        verbose=False
    )

    output = image.copy()
    found = False

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])
            found = True

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            cv2.putText(
                output,
                f"CONJUNCTIVA {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    cv2.imwrite(output_path, output)

    print(filename, "-> Conjunctiva:", found)

print()
print("=" * 60)
print("TEST COMPLETED")
print("=" * 60)
print("Results saved to:", OUTPUT_FOLDER)