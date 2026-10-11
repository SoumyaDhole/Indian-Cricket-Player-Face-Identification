from __future__ import annotations

import argparse
from pathlib import Path

from dataset import DatasetStructureError, discover_dataset, write_inventory_csv


ROOT = Path(__file__).parent


def default_dataset_root() -> Path:
    for name in ("indian_cricketer", "indian cricketer"):
        candidate = ROOT / name
        if candidate.is_dir():
            return candidate
    return ROOT / "indian_cricketer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect folder-labeled player images and export counts.")
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_root())
    parser.add_argument("--csv", type=Path, default=ROOT / "dataset_inventory.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inventory = discover_dataset(args.dataset_dir)
    except DatasetStructureError as error:
        print(f"Dataset inspection failed: {error}")
        return 1

    print(f"Dataset root: {inventory.root}")
    print(f"Player folders/classes: {inventory.class_count}")
    print(f"Files inside class folders: {inventory.file_count - len(inventory.root_level_files)}")
    print(f"Readable supported images: {inventory.readable_supported_count}")
    print(f"Unreadable files: {len(inventory.unreadable_files)}")
    print(f"Unsupported files: {len(inventory.unsupported_files)}")
    print(f"Empty class folders: {len(inventory.empty_folders)}")
    print(f"Files at dataset root (not assigned to a class): {len(inventory.root_level_files)}")
    print("Extensions:")
    for extension, count in inventory.extension_counts.items():
        print(f"  {extension or '<no extension>'}: {count}")
    print("Per-player counts (total / readable supported):")
    for player in inventory.players:
        print(f"  {player.label}: {player.file_count} / {player.readable_supported_count}")

    write_inventory_csv(inventory, args.csv)
    print(f"Full per-player inventory written to: {args.csv}")
    if inventory.unreadable_files:
        print("Unreadable file paths:")
        for path in inventory.unreadable_files:
            print(f"  {path}")
    if inventory.empty_folders:
        print("Empty class folders:")
        for path in inventory.empty_folders:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
