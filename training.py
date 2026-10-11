from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from face_id import (
    FaceIdentificationError,
    ModelLoadError,
    SUPPORTED_IMAGE_EXTENSIONS,
    extract_single_face_embedding,
    load_face_models,
)


DEFAULT_SPLIT_SEED = 42
NEAR_DUPLICATE_HAMMING_DISTANCE = 12
CROSS_LABEL_DUPLICATE_SFACE_SIMILARITY = 0.965
CLASSIFIER_FILENAME = "player_svm.joblib"
METADATA_FILENAME = "metadata.json"
EMBEDDING_CACHE_SCHEMA_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1

_WORKER_DETECTOR: cv2.FaceDetectorYN | None = None
_WORKER_RECOGNIZER: cv2.FaceRecognizerSF | None = None


class DatasetValidationError(ValueError):
    """The labeled image set cannot support a leakage-safe evaluation split."""


class ClassifierArtifactError(RuntimeError):
    """The saved classifier or its metadata is missing, invalid, or incompatible."""


class ModelNotTrainedError(ClassifierArtifactError):
    """No trained SVM artifact has been created yet."""


@dataclass(frozen=True)
class EmbeddingRecord:
    image_path: Path
    player_name: str
    embedding: np.ndarray
    duplicate_group: int


@dataclass(frozen=True)
class DataSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class ClassifierPrediction:
    player_name: str | None
    candidate_name: str
    margin: float


@dataclass(frozen=True)
class TrainedClassifier:
    estimator: OneVsRestClassifier
    labels: tuple[str, ...]
    rejection_threshold: float
    rejection_method: str
    split_seed: int


def perceptual_hash(image: np.ndarray) -> int:
    """Compute a compact pHash used to keep within-class near-duplicates together."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    reduced = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequencies = cv2.dct(reduced)[:8, :8].reshape(-1)[1:]
    median = float(np.median(low_frequencies))
    value = 0
    for bit in low_frequencies >= median:
        value = (value << 1) | int(bit)
    return value


def assign_duplicate_groups(
    fingerprints: Sequence[int], max_hamming_distance: int = NEAR_DUPLICATE_HAMMING_DISTANCE
) -> list[int]:
    """Cluster exact and perceptually similar fingerprints deterministically."""
    if max_hamming_distance < 0:
        raise ValueError("max_hamming_distance must be zero or greater.")
    parents = list(range(len(fingerprints)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left in range(len(fingerprints)):
        for right in range(left + 1, len(fingerprints)):
            distance = (int(fingerprints[left]) ^ int(fingerprints[right])).bit_count()
            if distance <= max_hamming_distance:
                left_root = find(left)
                right_root = find(right)
                if left_root != right_root:
                    parents[max(left_root, right_root)] = min(left_root, right_root)

    roots: dict[int, int] = {}
    groups: list[int] = []
    for index in range(len(fingerprints)):
        root = find(index)
        roots.setdefault(root, len(roots))
        groups.append(roots[root])
    return groups


def _initialize_embedding_worker(model_dir: str) -> None:
    global _WORKER_DETECTOR, _WORKER_RECOGNIZER
    cv2.setNumThreads(1)
    _WORKER_DETECTOR, _WORKER_RECOGNIZER = load_face_models(Path(model_dir))


def _extract_training_image(
    task: tuple[str, str],
) -> tuple[tuple[str, str, np.ndarray, int, str] | None, str | None]:
    path_text, label = task
    path = Path(path_text)
    try:
        image_bytes = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error) as error:
        return None, f"Skipped unreadable image: {path} ({error})"
    if image is None or image.size == 0:
        return None, f"Skipped unreadable image: {path}"
    if _WORKER_DETECTOR is None or _WORKER_RECOGNIZER is None:
        return None, "Face models were not initialized in the extraction worker."
    try:
        _, embedding = extract_single_face_embedding(_WORKER_DETECTOR, _WORKER_RECOGNIZER, image)
    except FaceIdentificationError as error:
        return None, f"Skipped image with no single detectable face ({error}): {path}"
    except cv2.error as error:
        return None, f"Skipped image after OpenCV error ({error}): {path}"
    row = (str(path), label, embedding, perceptual_hash(image), hashlib.sha256(image_bytes).hexdigest())
    return row, None


def _embedding_cache_signature(reference_dir: Path, model_dir: Path, source_files: Sequence[Path]) -> str:
    details: list[str] = [str(Path(reference_dir).resolve()), cv2.__version__]
    for model_name in ("face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx"):
        model_path = Path(model_dir) / model_name
        if not model_path.is_file():
            raise ModelLoadError(f"Missing model file: {model_path}")
        stat = model_path.stat()
        details.append(f"{model_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
    for path in source_files:
        stat = path.stat()
        details.append(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("\n".join(details).encode("utf-8")).hexdigest()


def _group_fingerprints_by_label(
    labels: Sequence[str], fingerprints: Sequence[int], max_hamming_distance: int
) -> list[int]:
    if len(labels) != len(fingerprints):
        raise DatasetValidationError("Each labeled image must have one perceptual fingerprint.")
    group_ids = [-1] * len(labels)
    next_group_id = 0
    for label in sorted(set(labels)):
        indices = [index for index, name in enumerate(labels) if name == label]
        local_groups = assign_duplicate_groups(
            [fingerprints[index] for index in indices], max_hamming_distance
        )
        remap = {group: next_group_id + offset for offset, group in enumerate(sorted(set(local_groups)))}
        for index, local_group in zip(indices, local_groups, strict=True):
            group_ids[index] = remap[local_group]
        next_group_id += len(remap)
    return group_ids


def exclude_cross_label_exact_duplicates(
    rows: Sequence[tuple[str, str, np.ndarray, int, str]],
) -> tuple[list[tuple[str, str, np.ndarray, int, str]], list[str]]:
    """Exclude every copy of a byte-identical image if folders assign different labels."""
    labels_by_hash: dict[str, set[str]] = {}
    for _, label, _, _, image_hash in rows:
        labels_by_hash.setdefault(image_hash, set()).add(label)
    conflicting_hashes = {
        image_hash for image_hash, labels in labels_by_hash.items() if len(labels) > 1
    }
    if not conflicting_hashes:
        return list(rows), []
    conflicts = [row for row in rows if row[4] in conflicting_hashes]
    retained = [row for row in rows if row[4] not in conflicting_hashes]
    warnings = [
        f"Skipped byte-identical image with conflicting folder label '{label}': {path}"
        for path, label, _, _, image_hash in conflicts
        if image_hash in conflicting_hashes
    ]
    warnings.insert(
        0,
        f"Skipped {len(conflicts)} image files from {len(conflicting_hashes)} byte-identical "
        "cross-label duplicate group(s).",
    )
    return retained, warnings


def exclude_cross_label_near_duplicates(
    rows: Sequence[tuple[str, str, np.ndarray, int, str]],
    max_hamming_distance: int = NEAR_DUPLICATE_HAMMING_DISTANCE,
    minimum_sface_similarity: float = CROSS_LABEL_DUPLICATE_SFACE_SIMILARITY,
) -> tuple[list[tuple[str, str, np.ndarray, int, str]], list[str]]:
    """Exclude likely cross-label photo copies only when pHash and SFace both agree."""
    normalized = [
        np.asarray(row[2], dtype=np.float32).reshape(-1)
        / np.linalg.norm(np.asarray(row[2], dtype=np.float32).reshape(-1))
        for row in rows
    ]
    excluded_indices: set[int] = set()
    pairs: list[tuple[int, int, float]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            if rows[left][1] == rows[right][1]:
                continue
            if (int(rows[left][3]) ^ int(rows[right][3])).bit_count() > max_hamming_distance:
                continue
            if normalized[left].shape != normalized[right].shape:
                continue
            similarity = float(np.dot(normalized[left], normalized[right]))
            if similarity >= minimum_sface_similarity:
                excluded_indices.update((left, right))
                pairs.append((left, right, similarity))
    retained = [row for index, row in enumerate(rows) if index not in excluded_indices]
    warnings = []
    if pairs:
        warnings.append(
            f"Skipped {len(excluded_indices)} image files from {len(pairs)} cross-label near-duplicate "
            f"pair(s) (pHash <= {max_hamming_distance}, SFace cosine >= {minimum_sface_similarity:.3f})."
        )
        warnings.extend(
            f"Skipped cross-label near-duplicate (SFace cosine {similarity:.4f}): "
            f"{rows[left][1]}:{rows[left][0]} <> {rows[right][1]}:{rows[right][0]}"
            for left, right, similarity in pairs
        )
    return retained, warnings


def extract_training_records(
    reference_dir: Path,
    model_dir: Path,
    duplicate_distance: int = NEAR_DUPLICATE_HAMMING_DISTANCE,
    cache_path: Path | None = None,
    workers: int = 2,
) -> tuple[list[EmbeddingRecord], list[str]]:
    """Extract labeled embeddings, caching by image/model file metadata."""
    if not reference_dir.is_dir():
        raise DatasetValidationError(f"Reference directory does not exist: {reference_dir}")
    source_files = [
        path
        for player_dir in sorted(path for path in reference_dir.iterdir() if path.is_dir())
        for path in sorted(child for child in player_dir.rglob("*") if child.is_file())
    ]
    image_files = [path for path in source_files if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    unsupported_files = [path for path in source_files if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS]
    signature = _embedding_cache_signature(reference_dir, model_dir, source_files)
    warnings: list[str] = []
    cache_message: str | None = None
    cache_rows: list[tuple[str, str, np.ndarray, int, str]] | None = None

    if cache_path is not None and Path(cache_path).is_file():
        try:
            cached = joblib.load(cache_path)
            if (
                cached.get("schema_version") == EMBEDDING_CACHE_SCHEMA_VERSION
                and cached.get("signature") == signature
            ):
                cache_rows = cached["rows"]
                warnings = list(cached["warnings"])
                cache_message = f"[CACHE] Reused {len(cache_rows)} embeddings from {cache_path}"
            else:
                cache_message = "[CACHE] Dataset or model files changed; re-extracting embeddings."
        except Exception as error:
            cache_message = f"[CACHE] Could not read embedding cache ({error}); re-extracting embeddings."

    if cache_rows is None:
        tasks = [
            (str(path), path.relative_to(reference_dir).parts[0].replace("_", " "))
            for path in image_files
        ]
        worker_count = max(1, min(int(workers), len(tasks) or 1))
        if worker_count == 1:
            _initialize_embedding_worker(str(model_dir))
            extracted = list(map(_extract_training_image, tasks))
        else:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_embedding_worker,
                initargs=(str(model_dir),),
            ) as pool:
                extracted = list(pool.map(_extract_training_image, tasks, chunksize=8))

        cache_rows = []
        for (path_text, _), (row, warning) in zip(tasks, extracted, strict=True):
            if warning:
                warnings.append(warning)
            if row is not None:
                cache_rows.append(row)
        warnings.extend(f"Skipped unsupported image: {path}" for path in unsupported_files)
        if cache_path is not None:
            cache_path = Path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_name(cache_path.name + ".tmp")
            joblib.dump(
                {
                    "schema_version": EMBEDDING_CACHE_SCHEMA_VERSION,
                    "signature": signature,
                    "rows": cache_rows,
                    "warnings": warnings,
                },
                temporary_path,
            )
            temporary_path.replace(cache_path)
            cache_message = f"[CACHE] Saved {len(cache_rows)} embeddings to {cache_path}"

    if cache_message:
        warnings.insert(0, cache_message)
    cache_rows, duplicate_warnings = exclude_cross_label_exact_duplicates(cache_rows or [])
    cache_rows, near_duplicate_warnings = exclude_cross_label_near_duplicates(cache_rows)
    warnings.extend(duplicate_warnings)
    warnings.extend(near_duplicate_warnings)
    if not cache_rows:
        raise DatasetValidationError(f"No usable single-face reference images found under {reference_dir}.")

    labels = [row[1] for row in cache_rows]
    fingerprints = [row[3] for row in cache_rows]
    groups = _group_fingerprints_by_label(labels, fingerprints, duplicate_distance)
    records = [
        EmbeddingRecord(Path(path), label, embedding, group)
        for (path, label, embedding, _, _), group in zip(cache_rows, groups, strict=True)
    ]
    validate_records(records)
    return records, warnings


def validate_records(records: Sequence[EmbeddingRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not records:
        raise DatasetValidationError("No embedding samples were provided.")
    labels = [record.player_name for record in records]
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise DatasetValidationError("At least two labeled player folders with usable faces are required.")
    group_labels: dict[int, set[str]] = {}
    for record in records:
        group_labels.setdefault(record.duplicate_group, set()).add(record.player_name)
    conflicts = [group for group, names in group_labels.items() if len(names) > 1]
    if conflicts:
        raise DatasetValidationError(
            "A duplicate group has conflicting player labels. Review groups "
            f"{conflicts[:10]} before training."
        )

    vectors = [np.asarray(record.embedding, dtype=np.float32).reshape(-1) for record in records]
    dimensions = {vector.size for vector in vectors}
    if len(dimensions) != 1 or 0 in dimensions:
        raise DatasetValidationError("All face embeddings must have the same non-zero feature dimension.")
    if any(not np.all(np.isfinite(vector)) for vector in vectors):
        raise DatasetValidationError("Face embeddings contain NaN or infinite values.")
    if any(float(np.linalg.norm(vector)) == 0 for vector in vectors):
        raise DatasetValidationError("Face embeddings must not be zero vectors.")
    matrix = np.vstack([vector / np.linalg.norm(vector) for vector in vectors]).astype(np.float32)
    return matrix, np.asarray(labels), np.asarray([record.duplicate_group for record in records]), classes


def records_fingerprint(records: Sequence[EmbeddingRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.image_path.resolve()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.player_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.duplicate_group).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(record.embedding, dtype=np.float32).reshape(-1).tobytes())
    return digest.hexdigest()


def make_grouped_split(
    labels: Sequence[str], groups: Sequence[int], random_state: int = DEFAULT_SPLIT_SEED
) -> DataSplit:
    """Make deterministic per-class splits without separating duplicate groups."""
    label_array = np.asarray(labels)
    group_array = np.asarray(groups)
    if label_array.size != group_array.size or label_array.size == 0:
        raise DatasetValidationError("Labels and duplicate groups must have the same non-zero length.")
    classes = sorted(set(label_array.tolist()))
    groups_by_class: dict[str, set[int]] = {}
    group_labels: dict[int, set[str]] = {}
    for label, group in zip(label_array, group_array, strict=True):
        groups_by_class.setdefault(str(label), set()).add(int(group))
        group_labels.setdefault(int(group), set()).add(str(label))
    cross_label_groups = [group for group, names in group_labels.items() if len(names) > 1]
    if cross_label_groups:
        raise DatasetValidationError(
            "Duplicate groups cannot cross player labels; review groups "
            f"{cross_label_groups[:10]} before training."
        )
    insufficient = {label: len(groups) for label, groups in groups_by_class.items() if len(groups) < 4}
    if insufficient:
        details = ", ".join(f"{label}: {count} group(s)" for label, count in insufficient.items())
        raise DatasetValidationError(
            "A leakage-safe train/validation/test split needs at least four distinct photo groups per player; "
            f"found {details}."
        )

    random = np.random.default_rng(random_state)
    assigned: dict[str, set[int]] = {"train": set(), "validation": set(), "test": set()}
    for label in classes:
        class_groups = np.asarray(sorted(groups_by_class[str(label)]), dtype=np.int64)
        shuffled = random.permutation(class_groups)
        test_count = max(1, int(round(len(shuffled) * 0.2)))
        validation_count = max(1, int(round(len(shuffled) * 0.2)))
        while test_count + validation_count > len(shuffled) - 2:
            if test_count > validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                break
        assigned["test"].update(int(group) for group in shuffled[:test_count])
        assigned["validation"].update(
            int(group) for group in shuffled[test_count : test_count + validation_count]
        )
        assigned["train"].update(int(group) for group in shuffled[test_count + validation_count :])

    splits = {
        name: np.asarray(
            [index for index, group in enumerate(group_array) if int(group) in group_set], dtype=np.int64
        )
        for name, group_set in assigned.items()
    }
    return DataSplit(np.sort(splits["train"]), np.sort(splits["validation"]), np.sort(splits["test"]))


def filter_trainable_records(
    records: Sequence[EmbeddingRecord], minimum_groups_per_class: int = 4
) -> tuple[list[EmbeddingRecord], list[str]]:
    groups_by_class: dict[str, set[int]] = {}
    for record in records:
        groups_by_class.setdefault(record.player_name, set()).add(record.duplicate_group)
    excluded = {
        label: len(groups)
        for label, groups in groups_by_class.items()
        if len(groups) < minimum_groups_per_class
    }
    eligible = [record for record in records if record.player_name not in excluded]
    warnings = [
        f"Excluded {label}: {count} distinct photo groups; at least {minimum_groups_per_class} are required."
        for label, count in sorted(excluded.items())
    ]
    if len({record.player_name for record in eligible}) < 2:
        raise DatasetValidationError(
            "Fewer than two player classes have enough distinct photo groups for train/validation/test evaluation."
        )
    return eligible, warnings


def _decision_matrix(estimator: Any, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(estimator.decision_function(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    if scores.ndim != 2 or scores.shape[1] != len(classes):
        raise ClassifierArtifactError("Classifier returned decision scores incompatible with its class metadata.")
    return scores, classes


def _margins(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    winners = np.argmax(scores, axis=1)
    ordered = np.sort(scores, axis=1)
    return winners, ordered[:, -1] - ordered[:, -2]


def select_rejection_threshold(
    estimator: Any, validation_features: np.ndarray, validation_labels: Sequence[str]
) -> tuple[float, str]:
    scores, classes = _decision_matrix(estimator, validation_features)
    winner_indices, margins = _margins(scores)
    predictions = classes[winner_indices]
    correct_margins = margins[predictions == np.asarray(validation_labels)]
    if correct_margins.size == 0:
        return 0.0, "zero-margin-heuristic-no-correct-validation-examples"
    threshold = max(0.0, float(np.quantile(correct_margins, 0.10)) * 0.95)
    return threshold, "validation-correct-margin-q10-heuristic"


def predict_embedding(model: TrainedClassifier, embedding: np.ndarray) -> ClassifierPrediction:
    vector = np.array(embedding, dtype=np.float32, copy=True).reshape(1, -1)
    if vector.shape[1] != int(model.estimator.n_features_in_):
        raise ClassifierArtifactError("The face embedding dimension does not match the trained classifier.")
    if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) == 0:
        raise ClassifierArtifactError("The face embedding must be finite and non-zero.")
    vector /= np.linalg.norm(vector)
    scores, classes = _decision_matrix(model.estimator, vector)
    winner_indices, margins = _margins(scores)
    candidate = str(classes[winner_indices[0]])
    margin = float(margins[0])
    if margin <= model.rejection_threshold:
        return ClassifierPrediction(None, candidate, margin)
    return ClassifierPrediction(candidate, candidate, margin)


def _calculate_metrics(
    estimator: Any,
    model_threshold: float,
    features: np.ndarray,
    labels: Sequence[str],
    classes: Sequence[str],
) -> dict[str, Any]:
    scores, estimator_classes = _decision_matrix(estimator, features)
    winner_indices, margins = _margins(scores)
    predictions = estimator_classes[winner_indices].astype(object)
    predictions[margins <= model_threshold] = "Unknown player"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, labels=list(classes), average="macro", zero_division=0
    )
    matrix_labels = list(classes) + ["Unknown player"]
    matrix = confusion_matrix(labels, predictions, labels=matrix_labels)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": {"labels": matrix_labels, "counts": matrix.tolist()},
    }


def train_and_evaluate(
    records: Sequence[EmbeddingRecord],
    artifact_dir: Path,
    random_state: int = DEFAULT_SPLIT_SEED,
    dataset_source: Path | None = None,
) -> tuple[TrainedClassifier, dict[str, Any]]:
    input_count = len(records)
    records, excluded_classes = filter_trainable_records(records)
    data_fingerprint = records_fingerprint(records)
    features, labels, groups, classes = validate_records(records)
    split = make_grouped_split(labels, groups, random_state)
    estimator = OneVsRestClassifier(
        LinearSVC(class_weight="balanced", random_state=random_state, max_iter=20000)
    )
    estimator.fit(features[split.train], labels[split.train])
    threshold, threshold_method = select_rejection_threshold(
        estimator, features[split.validation], labels[split.validation]
    )
    metrics = _calculate_metrics(estimator, threshold, features[split.test], labels[split.test], classes)
    model = TrainedClassifier(estimator, tuple(classes), threshold, threshold_method, random_state)
    report: dict[str, Any] = {
        "test_split_valid": True,
        "test_metrics": metrics,
        "samples": {
            "input": int(input_count),
            "used": int(len(records)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "per_class_used": dict(sorted(Counter(labels.tolist()).items())),
        "duplicate_groups": int(len(set(groups.tolist()))),
        "excluded_classes": excluded_classes,
        "dataset_source": str(dataset_source) if dataset_source is not None else None,
        "dataset_fingerprint": data_fingerprint,
        "split_seed": random_state,
    }
    save_classifier(model, artifact_dir, report)
    return model, report


def save_classifier(model: TrainedClassifier, artifact_dir: Path, evaluation: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / CLASSIFIER_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    joblib.dump(model.estimator, model_path)
    metadata = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "labels": list(model.labels),
        "feature_count": int(model.estimator.n_features_in_),
        "rejection_threshold": model.rejection_threshold,
        "rejection_method": model.rejection_method,
        "split_seed": model.split_seed,
        "evaluation": evaluation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if "confusion_matrix" in evaluation.get("test_metrics", {}):
        write_confusion_matrix_csv(evaluation["test_metrics"]["confusion_matrix"], artifact_dir / "confusion_matrix.csv")


def write_confusion_matrix_csv(matrix: dict[str, Any], output_path: Path) -> None:
    labels = matrix["labels"]
    counts = matrix["counts"]
    with Path(output_path).open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["actual \\ predicted", *labels])
        for label, row in zip(labels, counts, strict=True):
            writer.writerow([label, *row])


def load_classifier(artifact_dir: Path) -> TrainedClassifier:
    model_path = artifact_dir / CLASSIFIER_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not model_path.exists() and not metadata_path.exists():
        raise ModelNotTrainedError(
            f"No trained SVM found in {artifact_dir}. Run `python train_classifier.py` first."
        )
    if not model_path.is_file() or not metadata_path.is_file():
        raise ClassifierArtifactError(
            f"Incomplete SVM artifact in {artifact_dir}; expected both {CLASSIFIER_FILENAME} and {METADATA_FILENAME}."
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        estimator = joblib.load(model_path)
        labels = tuple(metadata["labels"])
        threshold = float(metadata["rejection_threshold"])
        seed = int(metadata["split_seed"])
        method = str(metadata["rejection_method"])
        if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported artifact schema version")
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("invalid rejection threshold")
        if tuple(str(label) for label in estimator.classes_) != labels:
            raise ValueError("classifier labels do not match metadata")
        if int(estimator.n_features_in_) != int(metadata["feature_count"]):
            raise ValueError("classifier feature count does not match metadata")
        if not hasattr(estimator, "decision_function"):
            raise ValueError("classifier does not provide decision scores")
    except Exception as error:
        raise ClassifierArtifactError(f"Could not load SVM artifact from {artifact_dir}: {error}") from error
    return TrainedClassifier(estimator, labels, threshold, method, seed)


def evaluate_saved_classifier(
    records: Sequence[EmbeddingRecord], artifact_dir: Path, dataset_source: Path | None = None
) -> dict[str, Any]:
    model = load_classifier(artifact_dir)
    input_count = len(records)
    records, excluded_classes = filter_trainable_records(records)
    features, labels, groups, classes = validate_records(records)
    metadata = json.loads((artifact_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    saved_fingerprint = metadata.get("evaluation", {}).get("dataset_fingerprint")
    current_fingerprint = records_fingerprint(records)
    if not saved_fingerprint or current_fingerprint != saved_fingerprint:
        raise DatasetValidationError(
            "The dataset embeddings do not match the dataset used to train this artifact. "
            "Train again before reporting held-out metrics."
        )
    split = make_grouped_split(labels, groups, model.split_seed)
    if tuple(classes) != model.labels:
        raise DatasetValidationError(
            f"Current player labels {classes} do not match trained artifact labels {list(model.labels)}."
        )
    metrics = _calculate_metrics(
        model.estimator, model.rejection_threshold, features[split.test], labels[split.test], classes
    )
    return {
        "test_split_valid": True,
        "test_metrics": metrics,
        "samples": {
            "input": int(input_count),
            "used": int(len(records)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "per_class_used": dict(sorted(Counter(labels.tolist()).items())),
        "duplicate_groups": int(len(set(groups.tolist()))),
        "excluded_classes": excluded_classes,
        "dataset_source": str(dataset_source) if dataset_source is not None else None,
        "split_seed": model.split_seed,
    }


def print_evaluation(report: dict[str, Any]) -> None:
    metrics = report["test_metrics"]
    print("Held-out test metrics (computed from the current dataset):")
    print(f"  accuracy:         {metrics['accuracy']:.4f}")
    print(f"  macro precision:  {metrics['macro_precision']:.4f}")
    print(f"  macro recall:     {metrics['macro_recall']:.4f}")
    print(f"  macro F1:         {metrics['macro_f1']:.4f}")
    matrix_labels = metrics["confusion_matrix"]["labels"]
    print(f"  confusion matrix: {len(matrix_labels)}x{len(matrix_labels)} (saved as confusion_matrix.csv)")
    print(f"  sample counts:    {report['samples']}")
    print(f"  player classes:   {len(report['per_class_used'])}")
    print(f"  duplicate groups: {report['duplicate_groups']}")