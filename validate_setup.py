from pathlib import Path

import cv2

from face_id import (
    SFACE_MODEL_FILENAME,
    SUPPORTED_IMAGE_EXTENSIONS,
    YUNET_MODEL_FILENAME,
)


ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
REFERENCE_DIR = ROOT / "reference_images"


def collect_setup_status(model_dir: Path, reference_dir: Path) -> tuple[list[str], bool]:
    messages: list[str] = []
    ready = True
    for filename in (YUNET_MODEL_FILENAME, SFACE_MODEL_FILENAME):
        model_path = model_dir / filename
        if model_path.is_file() and model_path.stat().st_size > 0:
            messages.append(f"[OK] Model: {model_path} ({model_path.stat().st_size:,} bytes)")
        else:
            messages.append(f"[MISSING] Model file: {model_path}")
            ready = False

    usable_reference_count = 0
    player_folders = sorted(path for path in reference_dir.iterdir() if path.is_dir()) if reference_dir.is_dir() else []
    for player_dir in player_folders:
        decodable_count = 0
        for image_path in sorted(player_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            if cv2.imread(str(image_path), cv2.IMREAD_COLOR) is None:
                messages.append(f"[INVALID] Reference image cannot be decoded: {image_path}")
            else:
                decodable_count += 1
        if decodable_count:
            usable_reference_count += decodable_count
            messages.append(f"[OK] {player_dir.name}: {decodable_count} decodable reference image(s)")
        else:
            messages.append(f"[MISSING] No decodable JPG, JPEG, PNG, or WEBP reference photos in: {player_dir}")

    if usable_reference_count == 0:
        messages.append(
            f"[MISSING] No decodable labeled reference photos found under {reference_dir}. "
            "Create reference_images/<Player_Name>/ and add permitted photos."
        )
        ready = False
    messages.append("[NOTE] This checks file presence and decoding only; model loading and face detection are not tested.")
    return messages, ready


def main() -> int:
    messages, ready = collect_setup_status(MODEL_DIR, REFERENCE_DIR)
    print("Setup validation:")
    print("\n".join(messages))
    print("\nSetup files are present." if ready else "\nSetup incomplete; resolve the [MISSING] entries above.")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())