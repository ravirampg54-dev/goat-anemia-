from ultralytics import YOLO
import cv2
import os

EYE_MODEL = "model/best.pt"
CONJUNCTIVA_MODEL = "model/conjunctiva_best.pt"

INPUT_FOLDER = "test_images"
OUTPUT_FOLDER = "results/automatic"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

eye_model = YOLO(EYE_MODEL)
conjunctiva_model = YOLO(CONJUNCTIVA_MODEL)

print("=" * 60)
print("GOAT EYE + CONJUNCTIVA AUTOMATIC DETECTION")
print("=" * 60)

images = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

print("Images found:", len(images))
print()

for filename in images:

    input_path = os.path.join(INPUT_FOLDER, filename)

    image = cv2.imread(input_path)

    if image is None:
        print("ERROR:", filename)
        continue

    output = image.copy()

    eye_found = False
    conjunctiva_found = False

    # -------------------------------------------------
    # STEP 1: DETECT GOAT EYE
    # -------------------------------------------------

    eye_results = eye_model.predict(
        source=image,
        conf=0.10,
        device="cpu",
        verbose=False
    )

    for eye_result in eye_results:

        if eye_result.boxes is None:
            continue

        for eye_box in eye_result.boxes:

            x1, y1, x2, y2 = map(
                int,
                eye_box.xyxy[0].tolist()
            )

            confidence = float(eye_box.conf[0])

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            eye_found = True

            # Draw eye box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                output,
                f"GOAT EYE {confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # -------------------------------------------------
            # STEP 2: CROP THE DETECTED EYE
            # -------------------------------------------------

            eye_crop = image[y1:y2, x1:x2]

            if eye_crop.size == 0:
                continue

            # Save eye crop
            base_name = os.path.splitext(filename)[0]

            eye_crop_path = os.path.join(
                OUTPUT_FOLDER,
                base_name + "_EYE_CROP.jpg"
            )

            cv2.imwrite(
                eye_crop_path,
                eye_crop
            )

            # -------------------------------------------------
            # STEP 3: DETECT CONJUNCTIVA INSIDE EYE CROP
            # -------------------------------------------------

            conjunctiva_results = conjunctiva_model.predict(
                source=eye_crop,
                conf=0.05,
                device="cpu",
                verbose=False
            )

            for c_result in conjunctiva_results:

                if c_result.boxes is None:
                    continue

                for c_box in c_result.boxes:

                    cx1, cy1, cx2, cy2 = map(
                        int,
                        c_box.xyxy[0].tolist()
                    )

                    c_confidence = float(
                        c_box.conf[0]
                    )

                    # Convert crop coordinates
                    # back to original image coordinates
                    ox1 = x1 + cx1
                    oy1 = y1 + cy1
                    ox2 = x1 + cx2
                    oy2 = y1 + cy2

                    conjunctiva_found = True

                    # Draw conjunctiva
                    cv2.rectangle(
                        output,
                        (ox1, oy1),
                        (ox2, oy2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        output,
                        f"CONJUNCTIVA {c_confidence:.2f}",
                        (ox1, min(oy1 + 25, image.shape[0] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # Save conjunctiva crop
                    conjunctiva_crop = eye_crop[
                        cy1:cy2,
                        cx1:cx2
                    ]

                    if conjunctiva_crop.size != 0:

                        conjunctiva_path = os.path.join(
                            OUTPUT_FOLDER,
                            base_name + "_CONJUNCTIVA.jpg"
                        )

                        cv2.imwrite(
                            conjunctiva_path,
                            conjunctiva_crop
                        )

    # -------------------------------------------------
    # STATUS
    # -------------------------------------------------

    if eye_found:

        cv2.putText(
            output,
            "GOAT EYE DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            output,
            "GOAT EYE NOT DETECTED",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    if conjunctiva_found:

        cv2.putText(
            output,
            "CONJUNCTIVA DETECTED",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            output,
            "CONJUNCTIVA NOT DETECTED",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # -------------------------------------------------
    # SAVE FINAL IMAGE
    # -------------------------------------------------

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    cv2.imwrite(
        output_path,
        output
    )

    print(
        f"{filename} -> "
        f"Eye: {eye_found} | "
        f"Conjunctiva: {conjunctiva_found}"
    )

print()
print("=" * 60)
print("PROCESS COMPLETED")
print("=" * 60)
print()
print("Results:")
print("results/automatic")