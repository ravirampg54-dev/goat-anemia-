const imageInput =
    document.getElementById("imageInput");

const previewContainer =
    document.getElementById("previewContainer");

const previewImage =
    document.getElementById("previewImage");


imageInput.addEventListener(
    "change",
    function () {

        const file = this.files[0];

        if (!file) {
            return;
        }

        const reader =
            new FileReader();

        reader.onload = function (event) {

            previewImage.src =
                event.target.result;

            previewContainer
                .classList
                .remove("hidden");
        };

        reader.readAsDataURL(file);
    }
);


async function analyzeImage() {

    const file =
        imageInput.files[0];

    if (!file) {

        alert(
            "Please select a goat image first."
        );

        return;
    }


    const button =
        document.getElementById(
            "analyzeButton"
        );

    const loading =
        document.getElementById(
            "loading"
        );

    const results =
        document.getElementById(
            "results"
        );


    button.disabled = true;

    loading
        .classList
        .remove("hidden");

    results
        .classList
        .add("hidden");


    const formData =
        new FormData();

    formData.append(
        "image",
        file
    );


    try {

        const response =
            await fetch(
                "/analyze",
                {
                    method: "POST",
                    body: formData
                }
            );


        const data =
            await response.json();


        if (!data.success) {

            alert(
                data.error ||
                "Analysis failed."
            );

            return;
        }


        // =====================================
        // ORIGINAL
        // =====================================

        document.getElementById(
            "originalResult"
        ).src =
            previewImage.src;


        // =====================================
        // GOAT
        // =====================================

        document.getElementById(
            "goatStatus"
        ).textContent =
            data.goat_detected
                ? "GOAT DETECTED"
                : "NOT A GOAT";


        // =====================================
        // EYE
        // =====================================

        document.getElementById(
            "eyeStatus"
        ).textContent =
            data.eye_detected
                ? "DETECTED"
                : "NOT DETECTED";


        if (data.eye_confidence !== null) {

            document.getElementById(
                "eyeConfidence"
            ).textContent =
                "Confidence: " +
                data.eye_confidence +
                "%";
        }


        if (data.eye_crop) {

            document.getElementById(
                "eyeResult"
            ).src =
                data.eye_crop;

        } else {

            document.getElementById(
                "eyeResult"
            ).removeAttribute("src");
        }


        // =====================================
        // CONJUNCTIVA
        // =====================================

        document.getElementById(
            "conjunctivaStatus"
        ).textContent =
            data.conjunctiva_detected
                ? "DETECTED"
                : "NOT DETECTED";


        if (
            data.conjunctiva_confidence !==
            null
        ) {

            document.getElementById(
                "conjunctivaConfidence"
            ).textContent =
                "Confidence: " +
                data.conjunctiva_confidence +
                "%";
        }


        if (data.conjunctiva_crop) {

            document.getElementById(
                "conjunctivaResult"
            ).src =
                data.conjunctiva_crop;

        } else {

            document.getElementById(
                "conjunctivaResult"
            ).removeAttribute("src");
        }


        // =====================================
        // ANNOTATED IMAGE
        // =====================================

        document.getElementById(
            "annotatedResult"
        ).src =
            data.annotated_image;


        // =====================================
        // FAMACHA
        // =====================================

        const score =
            document.getElementById(
                "famachaScore"
            );

        const text =
            document.getElementById(
                "famachaText"
            );

        const confidence =
            document.getElementById(
                "famachaConfidence"
            );


        if (data.famacha_result) {

            score.textContent =
                data.famacha_result
                    .replace("_", " ");

            text.textContent =
                "FAMACHA screening result";

            confidence.textContent =
                "Model confidence: " +
                data.famacha_confidence +
                "%";

        } else {

            score.textContent =
                "N/A";

            text.textContent =
                "FAMACHA analysis unavailable";

            confidence.textContent =
                "Conjunctiva was not detected.";
        }


        // =====================================
        // SHOW RESULTS
        // =====================================

        results
            .classList
            .remove("hidden");

    }

    catch (error) {

        console.error(error);

        alert(
            "An error occurred during analysis."
        );

    }

    finally {

        button.disabled = false;

        loading
            .classList
            .add("hidden");
    }
}