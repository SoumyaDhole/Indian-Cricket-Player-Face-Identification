from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from face_id import ModelLoadError
from training import (
    DEFAULT_SPLIT_SEED,
    NEAR_DUPLICATE_HAMMING_DISTANCE,
    DatasetValidationError,
    ClassifierArtifactError,
    evaluate_saved_classifier,
    extract_training_records,
    print_evaluation,
    train_and_evaluate,
    write_confusion_matrix_csv,
)


ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SFace embeddings, train a grouped-split SVM, and report held-out metrics."
    )
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "reference_images")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="folder-labeled dataset root; when supplied it takes precedence over --reference-dir",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "models" / "artifacts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--workers", type=int, default=2, help="CPU image-extraction processes (default: 2)"
    )
    parser.add_argument(
        "--duplicate-distance",
        type=int,
        default=NEAR_DUPLICATE_HAMMING_DISTANCE,
        help="pHash Hamming distance grouped as near-duplicate",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="re-evaluate the saved SVM on the deterministic grouped test split without retraining",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.dataset_dir or args.reference_dir
    try:
        records, warnings = extract_training_records(
            source_dir,
            args.model_dir,
            args.duplicate_distance,
            cache_path=args.artifact_dir / "embedding_cache.joblib",
            workers=args.workers,
        )
        cache_notes = [warning for warning in warnings if warning.startswith("[CACHE]")]
        for note in cache_notes:
            print(note)
        duplicate_summaries = [
            warning
            for warning in warnings
            if warning.startswith("Skipped ")
            and ("byte-identical cross-label duplicate group(s)" in warning
                 or "cross-label near-duplicate pair(s)" in warning)
        ]
        for summary in duplicate_summaries:
            print(f"[SKIPPED] {summary}")
        skipped = [
            warning
            for warning in warnings
            if not warning.startswith("[CACHE]") and warning not in duplicate_summaries
        ]
        skip_categories = Counter(
            "unsupported format" if warning.startswith("Skipped unsupported image")
            else "unreadable image" if warning.startswith("Skipped unreadable image")
            else "no single detectable face" if warning.startswith("Skipped image with no single detectable face")
            else "conflicting exact duplicate labels" if warning.startswith("Skipped byte-identical image")
            else "cross-label near-duplicate images" if warning.startswith("Skipped cross-label near-duplicate")
            else "OpenCV image error"
            for warning in skipped
        )
        for category, count in sorted(skip_categories.items()):
            print(f"[SKIPPED] {count} image(s): {category}")
        for warning in skipped[:10]:
            print(f"  {warning}")
        if len(skipped) > 10:
            print(f"  ... {len(skipped) - 10} more; see dataset_inventory.csv for file-level scan results")
        print(f"Embedding source: {source_dir}")
        print(
            f"Extracted {len(records)} usable face embeddings from "
            f"{len(set(record.player_name for record in records))} labeled players."
        )
        if args.evaluate_only:
            report = evaluate_saved_classifier(records, args.artifact_dir, source_dir)
            write_confusion_matrix_csv(
                report["test_metrics"]["confusion_matrix"], args.artifact_dir / "confusion_matrix.csv"
            )
            print_evaluation(report)
            return 0

        model, report = train_and_evaluate(
            records, args.artifact_dir, args.seed, dataset_source=source_dir
        )
        for warning in report["excluded_classes"]:
            print(f"[EXCLUDED] {warning}")
        print(f"Saved classifier and metadata to {args.artifact_dir}")
        print(
            "Unknown-player rejection uses threshold "
            f"{model.rejection_threshold:.6f} ({model.rejection_method})."
        )
        print_evaluation(report)
        print(
            "Note: the rejection margin is a heuristic derived from known validation players; "
            "there are no unknown-person examples in this dataset."
        )
        return 0
    except (ModelLoadError, DatasetValidationError, ClassifierArtifactError, ValueError) as error:
        print(f"Training/evaluation could not complete: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
