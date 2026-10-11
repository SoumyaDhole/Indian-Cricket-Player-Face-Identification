from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from face_id import SUPPORTED_IMAGE_EXTENSIONS


class DatasetStructureError(ValueError):
    """The dataset root does not contain class directories."""


@dataclass(frozen=True)
class PlayerImageSummary:
    label: str
    folder: Path
    file_count: int
    supported_count: int
    readable_supported_count: int
    unreadable_supported_count: int
    unsupported_count: int


@dataclass(frozen=True)
class DatasetInventory:
    root: Path
    players: tuple[PlayerImageSummary, ...]
    extension_counts: dict[str, int]
    unreadable_files: tuple[Path, ...]
    unsupported_files: tuple[Path, ...]
    empty_folders: tuple[Path, ...]
    root_level_files: tuple[Path, ...]

    @property
    def class_count(self) -> int:
        return len(self.players)

    @property
    def file_count(self) -> int:
        return sum(player.file_count for player in self.players) + len(self.root_level_files)

    @property
    def readable_supported_count(self) -> int:
        return sum(player.readable_supported_count for player in self.players)


def dataset_label(folder_name: str) -> str:
    """Convert the existing underscore naming convention without inventing labels."""
    return folder_name.replace("_", " ")


def discover_dataset(root: Path) -> DatasetInventory:
    """Inspect immediate player folders, recursively counting their image files."""
    root = Path(root)
    if not root.is_dir():
        raise DatasetStructureError(f"Dataset directory does not exist: {root}")
    player_folders = sorted(path for path in root.iterdir() if path.is_dir())
    if not player_folders:
        raise DatasetStructureError(f"No player class folders found under dataset root: {root}")

    summaries: list[PlayerImageSummary] = []
    extensions: Counter[str] = Counter()
    unreadable: list[Path] = []
    unsupported: list[Path] = []
    empty_folders: list[Path] = []
    for folder in player_folders:
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        if not files:
            empty_folders.append(folder)
        supported_count = 0
        readable_supported_count = 0
        unreadable_supported_count = 0
        unsupported_count = 0
        for path in files:
            extension = path.suffix.lower()
            extensions[extension] += 1
            if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                unsupported.append(path)
                if cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                    unreadable.append(path)
                continue
            supported_count += 1
            if cv2.imread(str(path), cv2.IMREAD_COLOR) is None:
                unreadable.append(path)
                unreadable_supported_count += 1
            else:
                readable_supported_count += 1
        unsupported_count = len(files) - supported_count
        summaries.append(
            PlayerImageSummary(
                dataset_label(folder.name), folder, len(files), supported_count,
                readable_supported_count, unreadable_supported_count, unsupported_count,
            )
        )

    root_level_files = tuple(sorted(path for path in root.iterdir() if path.is_file()))
    return DatasetInventory(
        root, tuple(summaries), dict(sorted(extensions.items())), tuple(unreadable),
        tuple(unsupported), tuple(empty_folders), root_level_files,
    )


def write_inventory_csv(inventory: DatasetInventory, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["player_label", "folder", "total_files", "supported_files", "readable_supported_files",
             "unreadable_supported_files", "unsupported_files"]
        )
        for player in inventory.players:
            writer.writerow(
                [player.label, str(player.folder), player.file_count, player.supported_count,
                 player.readable_supported_count, player.unreadable_supported_count, player.unsupported_count]
            )
