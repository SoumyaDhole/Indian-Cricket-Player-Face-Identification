from pathlib import Path

import cv2
import streamlit as st

from face_id import (
    FaceIdentifier,
    FaceIdentificationError,
    ImageValidationError,
    ModelLoadError,
    SUPPORTED_IMAGE_EXTENSIONS,
    decode_uploaded_image,
)
from training import (
    ClassifierArtifactError,
    ModelNotTrainedError,
    TrainedClassifier,
    load_classifier,
    predict_embedding,
)


ROOT = Path(__file__).parent
REFERENCE_DIR = ROOT / "reference_images"
MODEL_DIR = ROOT / "models"


@st.cache_resource(show_spinner="Loading the CPU face models and reference images...")
def get_identifier() -> FaceIdentifier:
    return FaceIdentifier.from_model_directory(MODEL_DIR, REFERENCE_DIR)


@st.cache_resource(show_spinner="Loading the trained SVM classifier...")
def get_trained_classifier() -> TrainedClassifier:
    return load_classifier(MODEL_DIR / "artifacts")


def main() -> None:
    st.set_page_config(page_title="Indian Cricket Player Face Identification", page_icon="🏏", layout="centered")
    st.title("Indian Cricket Player Face Identification")
    st.write(
        "Identify a player using a trained SVM over pretrained SFace embeddings, "
        "or the existing reference matcher until an SVM has been trained."
    )

    with st.expander("Supported players", expanded=True):
        try:
            trained_model = get_trained_classifier()
            player_names = list(trained_model.labels)
            roster_source = "trained SVM"
        except ModelNotTrainedError:
            player_names = sorted(
                path.name.replace("_", " ")
                for path in REFERENCE_DIR.iterdir()
                if path.is_dir()
                and any(
                    file.is_file() and file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                    for file in path.iterdir()
                )
            ) if REFERENCE_DIR.exists() else []
            roster_source = "reference folders (SVM not trained)"
        except ClassifierArtifactError as error:
            player_names = []
            roster_source = "trained SVM unavailable"
            st.error(f"The trained-player roster could not be loaded: {error}")
        if player_names:
            st.caption(f"{len(player_names)} supported classes from {roster_source}")
            st.write(", ".join(player_names))
        else:
            st.info("No labeled reference photos found yet. Add images using the instructions in README.md.")

    uploaded = st.file_uploader(
        "Choose a face image",
        type=["jpg", "jpeg", "png", "webp", "jfif", "bmp"],
        help="Use a clear, front-facing photo containing exactly one face.",
    )
    if uploaded is None:
        st.subheader("Results")
        st.caption("Your uploaded image, detected face, and match result will appear here.")
    elif st.button("Identify player", type="primary"):
        try:
            image = decode_uploaded_image(uploaded.getvalue(), uploaded.name)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded image", width="stretch")
            identifier = get_identifier()
            result = identifier.identify(image)
            try:
                classifier = get_trained_classifier()
            except ModelNotTrainedError:
                classifier = None
                st.info(
                    "The trained SVM has not been created yet. This result uses the existing "
                    "SFace cosine reference matcher. Run `python train_classifier.py`, then restart the app."
                )
            except ClassifierArtifactError as error:
                st.error(f"The trained SVM artifact could not be loaded: {error}")
                return
            if result.aligned_face is not None:
                st.image(
                    cv2.cvtColor(result.aligned_face, cv2.COLOR_BGR2RGB),
                    caption="Detected and aligned face",
                    width=220,
                )
            st.subheader("Result")
            if classifier is not None:
                prediction = predict_embedding(classifier, result.embedding)
                if prediction.player_name is None:
                    st.warning(f"Unknown player (closest SVM candidate: {prediction.candidate_name})")
                else:
                    st.success(prediction.player_name)
                st.metric("SVM decision margin", f"{prediction.margin:.3f}")
                st.caption(
                    f"Margin threshold: {classifier.rejection_threshold:.3f} "
                    f"({classifier.rejection_method}). This is not a probability."
                )
            else:
                if result.player_name is None:
                    st.warning("Unknown player")
                else:
                    st.success(result.player_name)
                if result.similarity is not None:
                    st.metric("Cosine similarity", f"{result.similarity:.3f}")
                    st.caption("Similarity is not a calibrated probability or a guarantee of identity.")
            if result.reference_warnings:
                with st.expander("Reference image notes"):
                    for warning in result.reference_warnings:
                        st.write(f"- {warning}")
        except ImageValidationError as error:
            st.error(str(error))
        except FaceIdentificationError as error:
            st.error(str(error))
        except ModelLoadError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Identification failed unexpectedly: {error}")

    st.subheader("Limitations")
    st.write(
        "The SVM is trained only on the labeled photos you provide. Evaluation is unreliable with "
        "very small or near-duplicate-heavy datasets; the rejection margin is not an unknown-person "
        "probability. Photo quality, lighting, pose, class coverage, and dataset licensing affect results."
    )
    st.caption("Images are processed locally by this app and are not sent to a paid service.")


if __name__ == "__main__":
    main()