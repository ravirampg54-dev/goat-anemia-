from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import os
import base64

app = Flask(__name__)

# ============================================================
# MODEL PATHS
# ============================================================

EYE_MODEL = "model/best.pt"
CONJUNCTIVA_MODEL = "model/conjunctiva_best.pt"
FAMACHA_MODEL = "model/famacha_best.pt"


# ============================================================
# OUTPUT FOLDERS
# ============================================================

os.makedirs("uploads", exist_ok=True)

os.makedirs(
    "results",
    exist_ok=True
)

os.makedirs(
    "results/eye_crops",
    exist_ok=True
)

os.makedirs(
    "results/eyelid_crops",
    exist_ok=True
)

os.makedirs(
    "results/conjunctiva_crops",
    exist_ok=True
)


# ============================================================
# LOAD EYE MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING GOAT HEALTH AI MODELS")
print("=" * 60)

try:

    eye_model = YOLO(EYE_MODEL)

    print(
        "Eye model loaded successfully:"
    )

    print(
        eye_model.names
    )

except Exception as e:

    eye_model = None

    print(
        "ERROR loading eye model:"
    )

    print(e)


# ============================================================
# LOAD CONJUNCTIVA MODEL
# ============================================================

try:

    conjunctiva_model = YOLO(
        CONJUNCTIVA_MODEL
    )

    print(
        "Conjunctiva model loaded successfully:"
    )

    print(
        conjunctiva_model.names
    )

except Exception as e:

    conjunctiva_model = None

    print(
        "ERROR loading conjunctiva model:"
    )

    print(e)


# ============================================================
# LOAD FAMACHA MODEL
# ============================================================

try:

    famacha_model = YOLO(
        FAMACHA_MODEL
    )

    print(
        "FAMACHA model loaded successfully:"
    )

    print(
        famacha_model.names
    )

except Exception as e:

    famacha_model = None

    print(
        "FAMACHA model not available:"
    )

    print(e)


print("=" * 60)
print()


# ============================================================
# IMAGE TO BASE64
# ============================================================

def image_to_base64(image):

    if image is None:
        return None

    if image.size == 0:
        return None

    success, buffer = cv2.imencode(
        ".png",
        image
    )

    if not success:
        return None

    encoded = base64.b64encode(
        buffer
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# MAKE CLEAR DISPLAY IMAGE
# ============================================================

def make_clear_display(
    image,
    scale=3
):

    if image is None:
        return None

    if image.size == 0:
        return None

    height, width = image.shape[:2]

    new_width = width * scale
    new_height = height * scale

    enlarged = cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_LANCZOS4
    )

    # Mild sharpening
    blurred = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        1.0
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.25,
        blurred,
        -0.25,
        0
    )

    return sharpened


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE IMAGE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    try:

        # ====================================================
        # CHECK UPLOAD
        # ====================================================

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error":
                    "No image was uploaded."
            })


        file = request.files["image"]


        if file.filename == "":

            return jsonify({
                "success": False,
                "error":
                    "Please select an image."
            })


        # ====================================================
        # SAVE ORIGINAL FILE
        # ====================================================

        filename = file.filename

        upload_path = os.path.join(
            "uploads",
            filename
        )

        file.save(
            upload_path
        )


        # ====================================================
        # READ IMAGE
        # ====================================================

        image = cv2.imread(
            upload_path,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return jsonify({
                "success": False,
                "error":
                    "OpenCV could not read the image."
            })


        print()
        print("=" * 60)
        print("NEW GOAT IMAGE")
        print("=" * 60)

        print(
            "Filename:",
            filename
        )

        print(
            "Image size:",
            image.shape
        )


        # ====================================================
        # KEEP ORIGINAL
        # ====================================================

        original = image.copy()

        annotated = image.copy()


        # ====================================================
        # VARIABLES
        # ====================================================

        eye_found = False

        eye_confidence = 0.0

        eye_crop = None

        eye_display = None

        eyelid_crop = None

        eyelid_display = None

        conjunctiva_crop = None

        conjunctiva_display = None

        conjunctiva_found = False

        conjunctiva_confidence = 0.0

        famacha_result = None

        famacha_confidence = None


        # ====================================================
        # EYE MODEL CHECK
        # ====================================================

        if eye_model is None:

            return jsonify({
                "success": False,
                "error":
                    "Eye model could not be loaded."
            })


        # ====================================================
        # FIND EYE
        # ====================================================

        print()
        print(
            "STEP 1: FINDING GOAT EYE..."
        )


        eye_results = eye_model.predict(

            source=image,

            conf=0.01,

            imgsz=1280,

            device="cpu",

            verbose=False

        )


        best_eye = None


        # ====================================================
        # FIND HIGHEST CONFIDENCE EYE
        # ====================================================

        for result in eye_results:

            if result.boxes is None:
                continue


            for box in result.boxes:

                confidence = float(
                    box.conf[0]
                )


                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )


                if (
                    best_eye is None
                    or
                    confidence >
                    best_eye["confidence"]
                ):

                    best_eye = {

                        "x1": x1,

                        "y1": y1,

                        "x2": x2,

                        "y2": y2,

                        "confidence":
                            confidence

                    }


        # ====================================================
        # EYE FOUND
        # ====================================================

        if best_eye is not None:

            eye_found = True


            # Original eye coordinates
            x1 = max(
                0,
                best_eye["x1"]
            )

            y1 = max(
                0,
                best_eye["y1"]
            )

            x2 = min(
                image.shape[1],
                best_eye["x2"]
            )

            y2 = min(
                image.shape[0],
                best_eye["y2"]
            )


            eye_confidence = (
                best_eye["confidence"]
            )


            print(
                "Eye detected."
            )

            print(
                "Eye confidence:",
                round(
                    eye_confidence,
                    4
                )
            )


            print(
                "Eye coordinates:",
                x1,
                y1,
                x2,
                y2
            )


            # =================================================
            # ADD PADDING
            # =================================================

            eye_width = x2 - x1

            eye_height = y2 - y1


            pad_x = int(
                eye_width * 0.25
            )

            pad_y = int(
                eye_height * 0.35
            )


            px1 = max(
                0,
                x1 - pad_x
            )

            py1 = max(
                0,
                y1 - pad_y
            )

            px2 = min(
                image.shape[1],
                x2 + pad_x
            )

            py2 = min(
                image.shape[0],
                y2 + pad_y
            )


            # =================================================
            # EYE CROP
            # =================================================

            eye_crop = image[
                py1:py2,
                px1:px2
            ]


            if eye_crop.size != 0:

                print(
                    "Eye crop created:"
                )

                print(
                    eye_crop.shape
                )


                # Save original-resolution crop
                cv2.imwrite(

                    "results/eye_crops/"
                    "eye_original.png",

                    eye_crop

                )


                # Create clear display version
                eye_display = (
                    make_clear_display(
                        eye_crop,
                        scale=3
                    )
                )


                cv2.imwrite(

                    "results/eye_crops/"
                    "eye_clear.png",

                    eye_display

                )


                # =================================================
                # DRAW EYE BOX
                # =================================================

                cv2.rectangle(

                    annotated,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    5

                )


                cv2.putText(

                    annotated,

                    "GOAT EYE",

                    (
                        x1,
                        max(
                            y1 - 15,
                            40
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    1.1,

                    (0, 255, 0),

                    3

                )


                # =================================================
                # LOWER EYELID CROP
                # =================================================

                print()
                print(
                    "STEP 2: EXTRACTING LOWER EYELID..."
                )


                eye_height_pixels = (
                    eye_crop.shape[0]
                )

                eye_width_pixels = (
                    eye_crop.shape[1]
                )


                # Lower 45 percent of eye crop
                lower_start = int(
                    eye_height_pixels * 0.55
                )


                eyelid_crop = eye_crop[
                    lower_start:
                    eye_height_pixels,

                    0:
                    eye_width_pixels
                ]


                if eyelid_crop.size != 0:

                    print(
                        "Lower eyelid crop created:"
                    )

                    print(
                        eyelid_crop.shape
                    )


                    cv2.imwrite(

                        "results/"
                        "eyelid_crops/"
                        "lower_eyelid_original.png",

                        eyelid_crop

                    )


                    eyelid_display = (
                        make_clear_display(

                            eyelid_crop,

                            scale=3

                        )
                    )


                    cv2.imwrite(

                        "results/"
                        "eyelid_crops/"
                        "lower_eyelid_clear.png",

                        eyelid_display

                    )


                # =================================================
                # CONJUNCTIVA MODEL
                # =================================================

                if conjunctiva_model is not None:

                    print()
                    print(
                        "STEP 3: FINDING CONJUNCTIVA..."
                    )


                    conjunctiva_results = (
                        conjunctiva_model.predict(

                            source=eye_crop,

                            conf=0.01,

                            imgsz=1280,

                            device="cpu",

                            verbose=False

                        )
                    )


                    best_conj = None


                    for result in conjunctiva_results:

                        if result.boxes is None:
                            continue


                        for box in result.boxes:

                            confidence = float(
                                box.conf[0]
                            )


                            cx1, cy1, cx2, cy2 = map(
                                int,
                                box.xyxy[0].tolist()
                            )


                            if (
                                best_conj is None
                                or
                                confidence >
                                best_conj[
                                    "confidence"
                                ]
                            ):

                                best_conj = {

                                    "x1": cx1,

                                    "y1": cy1,

                                    "x2": cx2,

                                    "y2": cy2,

                                    "confidence":
                                        confidence

                                }


                    # =================================================
                    # CONJUNCTIVA FOUND
                    # =================================================

                    if best_conj is not None:

                        cx1 = max(
                            0,
                            best_conj["x1"]
                        )

                        cy1 = max(
                            0,
                            best_conj["y1"]
                        )

                        cx2 = min(
                            eye_crop.shape[1],
                            best_conj["x2"]
                        )

                        cy2 = min(
                            eye_crop.shape[0],
                            best_conj["y2"]
                        )


                        conjunctiva_crop = (
                            eye_crop[
                                cy1:cy2,
                                cx1:cx2
                            ]
                        )


                        if (
                            conjunctiva_crop
                            is not None
                            and
                            conjunctiva_crop.size
                            != 0
                        ):

                            conjunctiva_found = True


                            conjunctiva_confidence = (
                                best_conj[
                                    "confidence"
                                ]
                            )


                            print(
                                "Conjunctiva detected."
                            )


                            print(
                                "Conjunctiva confidence:",
                                round(
                                    conjunctiva_confidence,
                                    4
                                )
                            )


                            # Save original crop
                            cv2.imwrite(

                                "results/"
                                "conjunctiva_crops/"
                                "conjunctiva_original.png",

                                conjunctiva_crop

                            )


                            # Clear display crop
                            conjunctiva_display = (
                                make_clear_display(

                                    conjunctiva_crop,

                                    scale=3

                                )
                            )


                            cv2.imwrite(

                                "results/"
                                "conjunctiva_crops/"
                                "conjunctiva_clear.png",

                                conjunctiva_display

                            )


                            # =================================================
                            # DRAW CONJUNCTIVA BOX
                            # =================================================

                            ox1 = (
                                px1 + cx1
                            )

                            oy1 = (
                                py1 + cy1
                            )

                            ox2 = (
                                px1 + cx2
                            )

                            oy2 = (
                                py1 + cy2
                            )


                            cv2.rectangle(

                                annotated,

                                (ox1, oy1),

                                (ox2, oy2),

                                (0, 0, 255),

                                5

                            )


                            cv2.putText(

                                annotated,

                                "CONJUNCTIVA",

                                (
                                    ox1,
                                    max(
                                        oy1 - 15,
                                        40
                                    )
                                ),

                                cv2.FONT_HERSHEY_SIMPLEX,

                                0.9,

                                (0, 0, 255),

                                3

                            )

                    else:

                        print(
                            "Conjunctiva not detected."
                        )

        else:

            print(
                "Eye was NOT detected."
            )


        # ====================================================
        # FAMACHA
        # ====================================================

        if (
            conjunctiva_found
            and
            conjunctiva_crop is not None
            and
            famacha_model is not None
        ):

            print()
            print(
                "STEP 4: FAMACHA SCREENING..."
            )


            famacha_results = (
                famacha_model.predict(

                    source=conjunctiva_crop,

                    imgsz=224,

                    device="cpu",

                    verbose=False

                )
            )


            for result in famacha_results:

                if result.probs is not None:

                    class_id = int(
                        result.probs.top1
                    )


                    famacha_confidence = float(
                        result.probs.top1conf
                    )


                    famacha_result = (
                        famacha_model.names[
                            class_id
                        ]
                    )


                    print(
                        "FAMACHA:",
                        famacha_result
                    )


                    print(
                        "FAMACHA confidence:",
                        round(
                            famacha_confidence,
                            4
                        )
                    )


        # ====================================================
        # CONVERT IMAGES FOR WEBSITE
        # ====================================================

        original_base64 = (
            image_to_base64(
                original
            )
        )


        annotated_base64 = (
            image_to_base64(
                annotated
            )
        )


        eye_base64 = None

        if eye_display is not None:

            eye_base64 = (
                image_to_base64(
                    eye_display
                )
            )


        eyelid_base64 = None

        if eyelid_display is not None:

            eyelid_base64 = (
                image_to_base64(
                    eyelid_display
                )
            )


        conjunctiva_base64 = None

        if conjunctiva_display is not None:

            conjunctiva_base64 = (
                image_to_base64(
                    conjunctiva_display
                )
            )


        # ====================================================
        # SEND RESULTS TO WEBSITE
        # ====================================================

        response = {

            "success":
                True,

            "original":
                original_base64,

            "annotated":
                annotated_base64,

            "eye_crop":
                eye_base64,

            "eyelid_crop":
                eyelid_base64,

            "conjunctiva_crop":
                conjunctiva_base64,

            "eye_detected":
                eye_found,

            "eye_confidence":
                round(
                    eye_confidence * 100,
                    2
                ),

            "conjunctiva_detected":
                conjunctiva_found,

            "conjunctiva_confidence":
                round(
                    conjunctiva_confidence * 100,
                    2
                ),

            "famacha_result":
                famacha_result,

            "famacha_confidence":
                (
                    round(
                        famacha_confidence * 100,
                        2
                    )
                    if famacha_confidence
                    is not None
                    else None
                )

        }


        print()
        print(
            "ANALYSIS COMPLETE"
        )

        print(
            "Eye:",
            eye_found
        )

        print(
            "Conjunctiva:",
            conjunctiva_found
        )

        print(
            "FAMACHA:",
            famacha_result
        )

        print(
            "=" * 60
        )


        return jsonify(
            response
        )


    except Exception as e:

        print()
        print(
            "=" * 60
        )

        print(
            "SERVER ERROR"
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("🐐 GOAT HEALTH AI")
    print("=" * 60)

    print(
        "Website:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "=" * 60
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )