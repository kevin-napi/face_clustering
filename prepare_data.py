import csv
from pathlib import Path
from PIL import Image
from tqdm import tqdm

DATA_DIR   = Path("data/anime_sample")
OUTPUT_DIR = Path("outputs")
MANIFEST   = OUTPUT_DIR / "manifest.csv"

def find_images(root):
    if not root.exists():
        raise FileNotFoundError(f"Could not find {root}\nMake sure your images are in data/anime_sample/")
    entries = []
    for img_path in sorted(root.glob("*")):
        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            entries.append((img_path, root.name))
    return entries

def verify_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Scanning {DATA_DIR} ...")
    entries = find_images(DATA_DIR)
    print(f"Found {len(entries)} images")
    valid = []
    skipped = 0
    for path, title in tqdm(entries, desc="Verifying images"):
        if verify_image(path):
            valid.append((str(path), title))
        else:
            skipped += 1
    print(f"\n{len(valid)} valid images ({skipped} skipped)")
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "manga_title"])
        writer.writerows(valid)
    print(f"Manifest saved → {MANIFEST}")

if __name__ == "__main__":
    main()
