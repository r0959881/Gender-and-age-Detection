import os
import cv2
from pathlib import Path
import shutil
SRC_DIR = Path("data/UTKFace")
OUT_GENDER = Path("data/gender")
OUT_AGE = Path("data/age")

AGE_BUCKETS = [
    (0,2),(3,6),(7,12),(13,19),(20,26),(27,33),(34,40),(41,47),(48,54),(55,61),(62,100)
]

def age_to_bucket(age):
    for lo, hi in AGE_BUCKETS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    return "unknown"

def ensure_dirs():
    (OUT_GENDER / "male").mkdir(parents=True, exist_ok=True)
    (OUT_GENDER / "female").mkdir(parents=True, exist_ok=True)
    for lo, hi in AGE_BUCKETS:
        (OUT_AGE / f"{lo}-{hi}").mkdir(parents=True, exist_ok=True)

def process_images():
    ensure_dirs()
    images = list(SRC_DIR.glob("*.jpg.chip.jpg"))
    for img_path in images:
        parts = img_path.name.split("_")
        if len(parts) < 4:
            continue
        try:
            age = int(parts[0])
            gender = int(parts[1])
        except ValueError:
            continue
        gender_dir = "male" if gender == 0 else "female"
        age_bucket = age_to_bucket(age)

        # Read and resize image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_resized = cv2.resize(img, (128, 128))

        # Save to gender folder
        cv2.imwrite(str(OUT_GENDER / gender_dir / img_path.name), img_resized)
        # Save to age bucket folder
        cv2.imwrite(str(OUT_AGE / age_bucket / img_path.name), img_resized)

    print("Images organized and resized.")

if __name__ == "__main__":
    process_images()