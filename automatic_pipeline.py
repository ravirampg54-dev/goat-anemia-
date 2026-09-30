from ultralytics import YOLO
import cv2
import os

# ==========================================
# MODELS
# ==========================================

EYE_MODEL = "model/best.pt"
CONJUNCTIVA_MODEL = "model/conjunctiva_best.pt"

# ==========================================
# FOLDERS
# ==========================================

INPUT_FOLDER = "test_images"

OUTPUT_FOLDER = "results/pipeline"

EYE_FOLDER = os.path.join(OUTPUT_FOLDER, "eye_crops")
CONJUNCTIVA_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "conjunctiva_crops"
)

ANNOTATED_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "annotated"
)

os.makedirs(EYE_FOLDER, exist_ok=True)
os.makedirs(CONJUNCTIVA_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)

# ==========================================
# LOAD MODELS
# ==========================================

print("=" * 60)
print("AUTOMATIC GOAT EYE PIPELINE")
print("=" * 60)

eye_model = YOLO(EYE_MODEL)
conjunctiva_model = YOLO(CONJUNCTIVA_MODEL)

print("Eye model:", eye_model.names)
print("Conjunctiva model:", conjunctiva_model.names)
print()

# ==========================================
# GET IMAGES
# ==========================================

images = [
    f for f in os.listdir(INPUT_FOLDER)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]

print("Images found:", len(images))
print()

# ==========================================
# PROCESS EACH IMAGE
# ==========================================

for filename in images:

    print("-" * 60)
    print("Processing:", filename)

    image_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Could not read image")
        continue

    annotated = image.copy()

    # ======================================
    # STEP 1: FIND EYE
    # ======================================

    eye_results = eye_model.predict(
        source=image,
        conf=0.10,
        device="cpu",
        verbose=False
    )

    eye_found = False

    for result in eye_results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            eye_found = True

            print(
                f"Eye detected "
                f"(confidence={confidence:.2f})"
            )

            # ==================================
            # DRAW EYE BOX
            # ==================================

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            cv2.putText(
                annotated,
                f"GOAT EYE {confidence:.2f}",
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ==================================
            # STEP 2: CROP EYE
            # ==================================

            eye_crop = image[y1:y2, x1:x2]

            if eye_crop.size == 0:
                print("Invalid eye crop")
                continue

            base_name = os.path.splitext(
                filename
            )[0]

            eye_crop_path = os.path.join(
                EYE_FOLDER,
                base_name + "_eye.jpg"
            )

            cv2.imwrite(
                eye_crop_path,
                eye_crop
            )

            print(
                "Eye crop saved:",
                eye_crop_path
            )

            # ==================================
            # STEP 3: FIND CONJUNCTIVA
            # ==================================

            conjunctiva_results = (
                conjunctiva_model.predict(
                    source=eye_crop,
                    conf=0.05,
                    device="cpu",
                    verbose=False
                )
            )

            conjunctiva_found = False

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

                    conjunctiva_found = True

                    print(
                        f"Conjunctiva detected "
                        f"(confidence={c_confidence:.2f})"
                    )

                    # ==================================
                    # DRAW CONJUNCTIVA ON ORIGINAL
                    # ==================================

                    ox1 = x1 + cx1
                    oy1 = y1 + cy1
                    ox2 = x1 + cx2
                    oy2 = y1 + cy2

                    cv2.rectangle(
                        annotated,
                        (ox1, oy1),
                        (ox2, oy2),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        annotated,
                        f"CONJUNCTIVA {c_confidence:.2f}",
                        (ox1, max(oy1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # ==================================
                    # STEP 4: CROP CONJUNCTIVA
                    # ==================================

                    conjunctiva_crop = eye_crop[
                        cy1:cy2,
                        cx1:cx2
                    ]

                    if conjunctiva_crop.size != 0:

                        conjunctiva_path = os.path.join(
                            CONJUNCTIVA_FOLDER,
                            base_name + "_conjunctiva.jpg"
                        )

                        cv2.imwrite(
                            conjunctiva_path,
                            conjunctiva_crop
                        )

                        print(
                            "Conjunctiva crop saved:",
                            conjunctiva_path
                        )

            if not conjunctiva_found:

                print(
                    "Conjunctiva NOT detected"
                )

    if not eye_found:

        print("Eye NOT detected")

        cv2.putText(
            annotated,
            "GOAT EYE NOT DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ======================================
    # SAVE ANNOTATED ORIGINAL
    # ======================================

    annotated_path = os.path.join(
        ANNOTATED_FOLDER,
        filename
    )

    cv2.imwrite(
        annotated_path,
        annotated
    )

    print(
        "Annotated image saved:",
        annotated_path
    )

print()
print("=" * 60)
print("PIPELINE COMPLETED")
print("=" * 60)

print()
print("Eye crops:")
print(EYE_FOLDER)

print()
print("Conjunctiva crops:")
print(CONJUNCTIVA_FOLDER)

print()
print("Annotated images:")
print(ANNOTATED_FOLDER)