
import random
import shutil
from pathlib import Path

random.seed(42)

source_dir = Path("data")
output_dir = source_dir / "eye_state_split"

classes = {
    "open": source_dir / "Open_Eyes",
    "closed": source_dir / "Closed_Eyes",
}

splits = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

for class_name, folder in classes.items():
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]

    if not images:
        raise ValueError(f"No images found in {folder}")

    random.shuffle(images)

    train_end = int(len(images) * 0.70)
    val_end = train_end + int(len(images) * 0.15)

    groups = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    for split_name, files in groups.items():
        destination = output_dir / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image in files:
            shutil.copy2(image, destination / image.name)

        print(f"{split_name}/{class_name}: {len(files)} images")

print("\nDataset preparation completed!")
print(f"Output folder: {output_dir.resolve()}")
