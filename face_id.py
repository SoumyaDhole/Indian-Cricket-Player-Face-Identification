from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".jfif", ".bmp"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
COSINE_SIMILARITY_THRESHOLD = 0.363
YUNET_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"


class ImageValidationError(ValueError):
    """The uploaded file is not a supported, decodable image."""


class FaceIdentificationError(ValueError):
    """The uploaded image does not contain exactly one detectable face."""


class ModelLoadError(RuntimeError):
    """A required pretrained OpenCV model could not be loaded."""


@dataclass(frozen=True)
class MatchResult:
    player_name: str | None
    similarity: float | None


@dataclass(frozen=True)
class IdentificationResult:
    player_name: str | None
    similarity: float | None
    aligned_face: np.ndarray | None
    reference_warnings: tuple[str, ...]
    embedding: np.ndarray


def decode_uploaded_image(image_bytes: bytes, filename: str) -> np.ndarray:
    """Validate a supported image upload and decode it to an OpenCV BGR array."""
    if Path(filename).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageValidationError("Unsupported image format. Upload a JPG, JPEG, PNG, WEBP, JFIF, or BMP image.")
    if not image_bytes:
        raise ImageValidationError("The uploaded file is empty or could not be read as an image.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError("The image is larger than the 10 MB upload limit.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageValidationError("The uploaded file is empty or could not be read as an image.")
    return image


def match_embedding(
    query_embedding: np.ndarray,
    reference_embeddings: dict[str, list[np.ndarray]],
    threshold: float = COSINE_SIMILARITY_THRESHOLD,
) -> MatchResult:
    """Return the closest labeled reference only when it meets the threshold."""
    query = _normalized_embedding(query_embedding)
    best_name: str | None = None
    best_similarity: float | None = None
    for player_name, embeddings in reference_embeddings.items():
        for embedding in embeddings:
            reference = _normalized_embedding(embedding)
            if query.shape != reference.shape:
                continue
            similarity = float(np.dot(query, reference))
            if best_similarity is None or similarity > best_similarity:
                best_name = player_name
                best_similarity = similarity

    if best_similarity is None:
        return MatchResult(None, None)
    if best_similarity < threshold:
        return MatchResult(None, best_similarity)
    return MatchResult(best_name, best_similarity)


def _normalized_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Face embeddings must be finite, non-zero vectors.")
    return vector / norm


class FaceIdentifier:
    def __init__(self, detector: cv2.FaceDetectorYN, recognizer: cv2.FaceRecognizerSF,
                 reference_embeddings: dict[str, list[np.ndarray]], warnings: tuple[str, ...] = ()) -> None:
        self.detector = detector
        self.recognizer = recognizer
        self.reference_embeddings = reference_embeddings
        self.reference_warnings = warnings

    @classmethod
    def from_model_directory(cls, model_dir: Path, reference_dir: Path) -> FaceIdentifier:
        detector, recognizer = load_face_models(model_dir)
        embeddings, warnings = _load_reference_embeddings(detector, recognizer, reference_dir)
        return cls(detector, recognizer, embeddings, warnings)

    def identify(self, image: np.ndarray) -> IdentificationResult:
        aligned, embedding = extract_single_face_embedding(self.detector, self.recognizer, image)
        match = match_embedding(embedding, self.reference_embeddings)
        return IdentificationResult(
            match.player_name, match.similarity, aligned, self.reference_warnings, embedding
        )


def load_face_models(model_dir: Path) -> tuple[cv2.FaceDetectorYN, cv2.FaceRecognizerSF]:
    detector_path = model_dir / YUNET_MODEL_FILENAME
    recognizer_path = model_dir / SFACE_MODEL_FILENAME
    missing = [path.name for path in (detector_path, recognizer_path) if not path.is_file()]
    if missing:
        raise ModelLoadError(
            "Missing model file(s): " + ", ".join(missing) + ". See README.md for the official download steps."
        )
    try:
        detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.9, 0.3, 5000)
        recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    except (cv2.error, AttributeError) as error:
        raise ModelLoadError(
            f"Could not load the face models: {error}. Install the required opencv-contrib-python-headless package."
        ) from error
    return detector, recognizer

def extract_single_face_embedding(
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise FaceIdentificationError("No face was detected. Try a clear, well-lit, front-facing image.")
    if len(faces) > 1:
        raise FaceIdentificationError("More than one face was detected. Upload an image with exactly one face.")
    aligned = recognizer.alignCrop(image, faces[0])
    embedding = recognizer.feature(aligned).reshape(-1)
    return aligned, embedding


def _load_reference_embeddings(
    detector: cv2.FaceDetectorYN,
    recognizer: cv2.FaceRecognizerSF,
    reference_dir: Path,
) -> tuple[dict[str, list[np.ndarray]], tuple[str, ...]]:
    embeddings: dict[str, list[np.ndarray]] = {}
    warnings: list[str] = []
    if not reference_dir.is_dir():
        return embeddings, (f"Reference directory not found: {reference_dir}",)

    for player_dir in sorted(path for path in reference_dir.iterdir() if path.is_dir()):
        player_embeddings: list[np.ndarray] = []
        for image_path in sorted(player_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                warnings.append(f"Skipped unsupported reference format: {image_path.relative_to(reference_dir)}")
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                warnings.append(f"Skipped unreadable reference image: {image_path.relative_to(reference_dir)}")
                continue
            try:
                _, embedding = extract_single_face_embedding(detector, recognizer, image)
            except FaceIdentificationError:
                warnings.append(
                    f"Skipped reference image without exactly one face: {image_path.relative_to(reference_dir)}"
                )
                continue
            player_embeddings.append(embedding)
        if player_embeddings:
            embeddings[player_dir.name.replace("_", " ")] = player_embeddings
        elif any(path.is_file() for path in player_dir.iterdir()):
            warnings.append(f"No usable face references found for {player_dir.name}.")
    return embeddings, tuple(warnings)