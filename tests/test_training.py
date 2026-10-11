from pathlib import Path

import numpy as np
import pytest

from training import (
    ClassifierArtifactError,
    DatasetValidationError,
    EmbeddingRecord,
    TrainedClassifier,
    assign_duplicate_groups,
    evaluate_saved_classifier,
    exclude_cross_label_exact_duplicates,
    exclude_cross_label_near_duplicates,
    filter_trainable_records,
    _group_fingerprints_by_label,
    load_classifier,
    make_grouped_split,
    predict_embedding,
    train_and_evaluate,
    validate_records,
)


PLAYER_LABELS = ("Jasprit Bumrah", "Rohit Sharma", "Virat Kohli")


def synthetic_records(samples_per_player: int = 6) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for player_index, player_name in enumerate(PLAYER_LABELS):
        for sample_index in range(samples_per_player):
            embedding = np.zeros(len(PLAYER_LABELS) + 2, dtype=np.float32)
            embedding[player_index] = 1.0
            embedding[-2] = sample_index * 0.003
            embedding[-1] = player_index * 0.002
            records.append(
                EmbeddingRecord(
                    Path(f"{player_name.replace(' ', '_')}/{sample_index}.jpg"),
                    player_name,
                    embedding,
                    len(records),
                )
            )
    return records


def test_duplicate_fingerprints_are_grouped() -> None:
    fingerprints = [0b101010, 0b101011, 0b000000]
    groups = assign_duplicate_groups(fingerprints, max_hamming_distance=1)
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_default_duplicate_tolerance_groups_12_bit_near_duplicate() -> None:
    within_threshold = assign_duplicate_groups([0, (1 << 12) - 1])
    above_threshold = assign_duplicate_groups([0, (1 << 13) - 1])
    assert within_threshold[0] == within_threshold[1]
    assert above_threshold[0] != above_threshold[1]


def test_perceptual_hash_groups_are_scoped_to_player_label() -> None:
    groups = _group_fingerprints_by_label(["Player A", "Player B"], [1234, 1234], 12)
    assert groups[0] != groups[1]


def test_conflicting_exact_duplicate_files_are_excluded_and_reported() -> None:
    shared_embedding = np.array([1.0, 0.0], dtype=np.float32)
    rows = [
        ("A/face.jpg", "Player A", shared_embedding, 1, "shared-hash"),
        ("B/copy.jpg", "Player B", shared_embedding, 2, "shared-hash"),
        ("A/other.jpg", "Player A", shared_embedding, 3, "unique-a"),
    ]

    retained, warnings = exclude_cross_label_exact_duplicates(rows)

    assert [row[0] for row in retained] == ["A/other.jpg"]
    assert any("Skipped 2 image files" in warning for warning in warnings)
    assert any("Player A" in warning and "A/face.jpg" in warning for warning in warnings)
    assert any("Player B" in warning and "B/copy.jpg" in warning for warning in warnings)


def test_cross_label_near_duplicate_requires_matching_face_embedding() -> None:
    rows = [
        ("A/face.jpg", "Player A", np.array([1.0, 0.0]), 100, "hash-a"),
        ("B/crop.jpg", "Player B", np.array([0.999, 0.01]), 101, "hash-b"),
        ("C/background.jpg", "Player C", np.array([0.0, 1.0]), 100, "hash-c"),
    ]

    retained, warnings = exclude_cross_label_near_duplicates(rows)

    assert {row[0] for row in retained} == {"C/background.jpg"}
    assert any("cross-label near-duplicate pair(s)" in warning for warning in warnings)


def test_grouped_split_keeps_groups_disjoint_and_all_labels_present() -> None:
    records = synthetic_records()
    duplicate = records[0]
    records[1] = EmbeddingRecord(
        records[1].image_path, records[1].player_name, records[1].embedding, duplicate.duplicate_group
    )
    _, labels, groups, classes = validate_records(records)
    split = make_grouped_split(labels, groups)
    train_groups = set(groups[split.train])
    validation_groups = set(groups[split.validation])
    test_groups = set(groups[split.test])

    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups
    assert set(labels[split.train]) == set(classes)
    assert set(labels[split.validation]) == set(classes)
    assert set(labels[split.test]) == set(classes)


def test_training_rejects_duplicate_group_with_conflicting_labels() -> None:
    records = synthetic_records()
    first = records[0]
    second = records[-1]
    records[0] = EmbeddingRecord(first.image_path, first.player_name, first.embedding, 123)
    records[-1] = EmbeddingRecord(second.image_path, second.player_name, second.embedding, 123)
    with pytest.raises(DatasetValidationError, match="conflicting player labels"):
        validate_records(records)


def test_invalid_split_does_not_produce_test_metrics() -> None:
    labels = ["A", "A", "B", "B"]
    groups = [1, 2, 3, 4]
    with pytest.raises(DatasetValidationError, match="at least four distinct photo groups"):
        make_grouped_split(labels, groups)


def test_training_does_not_save_artifacts_when_classes_are_insufficient(tmp_path) -> None:
    records = synthetic_records(samples_per_player=3)
    artifact_dir = tmp_path / "artifacts"
    with pytest.raises(DatasetValidationError, match="Fewer than two player classes"):
        train_and_evaluate(records, artifact_dir)
    assert not artifact_dir.exists()


def test_filter_trainable_records_excludes_under_sampled_class() -> None:
    records = synthetic_records()
    for sample_index in range(3):
        records.append(
            EmbeddingRecord(
                Path(f"Under Sampled/{sample_index}.jpg"),
                "Under Sampled",
                np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                100 + sample_index,
            )
        )

    eligible, warnings = filter_trainable_records(records)

    assert "Under Sampled" not in {record.player_name for record in eligible}
    assert any("Excluded Under Sampled: 3 distinct photo groups" in warning for warning in warnings)


def test_train_save_load_predict_and_reject_unknown(tmp_path) -> None:
    artifact_dir = tmp_path / "models" / "artifacts"
    trained, report = train_and_evaluate(synthetic_records(), artifact_dir)

    assert report["test_split_valid"] is True
    assert set(report["test_metrics"]) == {
        "accuracy", "macro_precision", "macro_recall", "macro_f1", "confusion_matrix"
    }
    confusion = report["test_metrics"]["confusion_matrix"]
    assert confusion["labels"] == [*PLAYER_LABELS, "Unknown player"]
    assert len(confusion["counts"]) == len(PLAYER_LABELS) + 1
    assert (artifact_dir / "player_svm.joblib").is_file()
    assert (artifact_dir / "metadata.json").is_file()
    repeated = evaluate_saved_classifier(synthetic_records(), artifact_dir)
    assert repeated["test_metrics"] == report["test_metrics"]

    loaded = load_classifier(artifact_dir)
    query = np.array([2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    original_query = query.copy()
    known = predict_embedding(loaded, query)
    assert known.player_name == "Jasprit Bumrah"
    assert known.margin > 0
    np.testing.assert_array_equal(query, original_query)

    strict_rejection = TrainedClassifier(
        loaded.estimator,
        loaded.labels,
        rejection_threshold=1_000_000.0,
        rejection_method="test-only-strict-threshold",
        split_seed=loaded.split_seed,
    )
    rejected = predict_embedding(
        strict_rejection, np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    assert rejected.player_name is None
    assert rejected.candidate_name == "Jasprit Bumrah"


def test_evaluate_only_rejects_changed_dataset_for_saved_artifact(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    train_and_evaluate(synthetic_records(), artifact_dir)
    changed_records = synthetic_records()
    changed_records[0] = EmbeddingRecord(
        Path("Jasprit_Bumrah/changed.jpg"),
        changed_records[0].player_name,
        changed_records[0].embedding,
        changed_records[0].duplicate_group,
    )
    with pytest.raises(DatasetValidationError, match="do not match the dataset used to train"):
        evaluate_saved_classifier(changed_records, artifact_dir)


def test_model_loader_reports_missing_or_incomplete_artifacts(tmp_path) -> None:
    with pytest.raises(ClassifierArtifactError, match="No trained SVM"):
        load_classifier(tmp_path)

    (tmp_path / "player_svm.joblib").touch()
    with pytest.raises(ClassifierArtifactError, match="Incomplete SVM artifact"):
        load_classifier(tmp_path)