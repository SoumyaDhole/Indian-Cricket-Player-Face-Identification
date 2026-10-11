import cv2
import numpy as np
import pytest

from face_id import (
    COSINE_SIMILARITY_THRESHOLD,
    FaceIdentificationError,
    FaceIdentifier,
    ImageValidationError,
    _load_reference_embeddings,
    decode_uploaded_image,
    match_embedding,
)
from validate_setup import collect_setup_status


def encoded_png() -> bytes:
    success, encoded = cv2.imencode(".png", np.zeros((12, 12, 3), dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_decode_uploaded_png() -> None:
    image = decode_uploaded_image(encoded_png(), "portrait.PNG")
    assert image.shape == (12, 12, 3)


@pytest.mark.parametrize(
    ("content", "filename", "message"),
    [
        (b"not an image", "portrait.jpg", "could not be read"),
        (encoded_png(), "portrait.gif", "Unsupported image format"),
        (b"", "portrait.png", "empty"),
    ],
)
def test_reject_invalid_uploads(content: bytes, filename: str, message: str) -> None:
    with pytest.raises(ImageValidationError, match=message):
        decode_uploaded_image(content, filename)


def test_match_embedding_returns_labeled_match_above_threshold() -> None:
    result = match_embedding(
        np.array([1.0, 0.0]),
        {"Virat Kohli": [np.array([0.99, 0.01])]},
    )
    assert result.player_name == "Virat Kohli"
    assert result.similarity is not None
    assert result.similarity >= COSINE_SIMILARITY_THRESHOLD


def test_match_embedding_returns_unknown_below_threshold() -> None:
    result = match_embedding(
        np.array([1.0, 0.0]),
        {"Virat Kohli": [np.array([0.0, 1.0])]},
    )
    assert result.player_name is None
    assert result.similarity == pytest.approx(0.0)


def test_match_embedding_without_references_returns_unknown() -> None:
    result = match_embedding(np.array([1.0, 0.0]), {})
    assert result.player_name is None
    assert result.similarity is None


class FakeDetector:
    def __init__(self, faces: np.ndarray | None) -> None:
        self.faces = faces
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_sizes.append(size)

    def detect(self, _image: np.ndarray) -> tuple[None, np.ndarray | None]:
        return None, self.faces


class FakeRecognizer:
    def __init__(self) -> None:
        self.aligned_faces = 0
        self.extracted_features = 0

    def alignCrop(self, _image: np.ndarray, face: np.ndarray) -> np.ndarray:
        self.aligned_faces += 1
        return face

    def feature(self, _aligned: np.ndarray) -> np.ndarray:
        self.extracted_features += 1
        return np.array([[1.0, 0.0]], dtype=np.float32)


@pytest.mark.parametrize("faces", [None, np.array([[1.0], [2.0]], dtype=np.float32)])
def test_identify_rejects_zero_or_multiple_faces(faces: np.ndarray | None) -> None:
    identifier = FaceIdentifier(FakeDetector(faces), FakeRecognizer(), {})  # type: ignore[arg-type]
    with pytest.raises(FaceIdentificationError):
        identifier.identify(np.zeros((8, 10, 3), dtype=np.uint8))


def test_identify_single_face_uses_unknown_when_references_are_missing() -> None:
    detector = FakeDetector(np.array([[1.0]], dtype=np.float32))
    identifier = FaceIdentifier(detector, FakeRecognizer(), {})  # type: ignore[arg-type]
    result = identifier.identify(np.zeros((8, 10, 3), dtype=np.uint8))
    assert result.player_name is None
    assert result.aligned_face is not None
    assert detector.input_sizes == [(10, 8)]


def test_setup_validation_reports_missing_models_and_references(tmp_path) -> None:
    messages, ready = collect_setup_status(tmp_path / "models", tmp_path / "reference_images")
    assert ready is False
    assert sum(message.startswith("[MISSING] Model file:") for message in messages) == 2
    assert any("No decodable labeled reference photos" in message for message in messages)


def test_setup_validation_counts_decodable_reference_photos(tmp_path) -> None:
    references = tmp_path / "reference_images"
    player_folder = references / "Virat_Kohli"
    player_folder.mkdir(parents=True)
    (player_folder / "portrait.png").write_bytes(encoded_png())

    messages, ready = collect_setup_status(tmp_path / "models", references)
    assert ready is False
    assert any("Virat_Kohli: 1 decodable reference image(s)" in message for message in messages)


def test_reference_and_uploaded_faces_share_preprocessing(tmp_path) -> None:
    reference_dir = tmp_path / "reference_images" / "Virat_Kohli"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "portrait.png"
    reference_path.write_bytes(encoded_png())
    detector = FakeDetector(np.array([[1.0]], dtype=np.float32))
    recognizer = FakeRecognizer()

    reference_embeddings, warnings = _load_reference_embeddings(
        detector, recognizer, reference_dir.parent  # type: ignore[arg-type]
    )
    identifier = FaceIdentifier(detector, recognizer, reference_embeddings, warnings)  # type: ignore[arg-type]
    result = identifier.identify(cv2.imdecode(np.frombuffer(encoded_png(), dtype=np.uint8), cv2.IMREAD_COLOR))

    assert result.player_name == "Virat Kohli"
    assert detector.input_sizes == [(12, 12), (12, 12)]
    assert recognizer.aligned_faces == 2
    assert recognizer.extracted_features == 2