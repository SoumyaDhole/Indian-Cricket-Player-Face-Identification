import cv2
import numpy as np
import pytest

from dataset import DatasetStructureError, discover_dataset, write_inventory_csv


def encoded_png() -> bytes:
    success, encoded = cv2.imencode(".png", np.full((8, 8, 3), 127, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def test_discover_dataset_uses_actual_folder_names_and_counts_files(tmp_path) -> None:
    root = tmp_path / "dataset"
    player_folder = root / "Player One"
    (player_folder / "nested").mkdir(parents=True)
    (player_folder / "nested" / "face.png").write_bytes(encoded_png())
    (player_folder / "broken.jpg").write_bytes(b"not an image")
    (root / "Empty Player").mkdir()
    (root / "Other_Player").mkdir()
    (root / "Other_Player" / "still.bmp").write_bytes(encoded_png())

    inventory = discover_dataset(root)

    assert inventory.class_count == 3
    assert inventory.file_count == 3
    assert inventory.readable_supported_count == 2
    assert {player.label for player in inventory.players} == {"Player One", "Empty Player", "Other Player"}
    player_one = next(player for player in inventory.players if player.label == "Player One")
    assert player_one.file_count == 2
    assert player_one.readable_supported_count == 1
    assert player_one.unreadable_supported_count == 1
    assert inventory.empty_folders == (root / "Empty Player",)


def test_inventory_reports_missing_root_and_missing_class_directories(tmp_path) -> None:
    with pytest.raises(DatasetStructureError, match="does not exist"):
        discover_dataset(tmp_path / "missing")

    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(DatasetStructureError, match="No player class folders"):
        discover_dataset(empty_root)


def test_inventory_csv_contains_per_player_counts(tmp_path) -> None:
    root = tmp_path / "dataset"
    player_folder = root / "Actual Label"
    player_folder.mkdir(parents=True)
    (player_folder / "face.png").write_bytes(encoded_png())
    inventory = discover_dataset(root)

    output = tmp_path / "report.csv"
    write_inventory_csv(inventory, output)

    contents = output.read_text(encoding="utf-8-sig")
    assert "player_label,folder,total_files,supported_files,readable_supported_files" in contents
    assert "Actual Label" in contents
    assert ",1,1,1,0,0" in contents
