import json
import random
import os
import shutil
from pathlib import Path
from collections import defaultdict, Counter

# This script assumes the DocLayNet dataset is downloaded and extracted into the 'data' directory.
# Expected structure:
#   data/DocLayNet/
#     ├─ images/
#     │   ├─ train/
#     │   ├─ val/
#     │   └─ test/
#     └─ coco_annotations.json
#
# Run: python3 utils/data_preparation.py

# Define paths relative to the project root
DATA_ROOT = Path('./data/DocLayNet')
OUTPUT_ROOT = Path('doclaynet_yolo')

# Allowed image extensions to probe if the annotated extension doesn't match what's on disk
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def _normalize_fname(fname: str) -> str:
    """
    Normalize a COCO 'file_name' to use forward slashes and strip any leading slashes.
    Also remove a leading 'images/' if present so we can safely join with DATA_ROOT/'images'.
    """
    f = fname.replace('\\', '/').lstrip('/')
    if f.startswith('images/'):
        f = f[len('images/'):]
    # Collapse accidental 'images/images/' duplication if it exists
    if f.startswith('images/images/'):
        f = f[len('images/'):]
    return f


def _find_image_path(dataset_root: Path, file_name: str) -> Path | None:
    """
    Given the dataset root and an image 'file_name' from the COCO JSON,
    try to resolve the actual file on disk with some robustness:
      - Normalize slashes, drop a leading 'images/' if present
      - Probe common image extensions if the exact path doesn't exist
    """
    f = _normalize_fname(file_name)
    candidate = (dataset_root / 'images' / f)

    # 1) Exact match
    if candidate.exists():
        return candidate

    # 2) Try alternative extensions (same base name)
    base = candidate.with_suffix('')
    for ext in IMG_EXTS:
        alt = base.with_suffix(ext)
        if alt.exists():
            return alt

    # 3) Last-ditch: look in split folders by basename only
    basename = Path(f).name
    for split in ('train', 'val', 'test'):
        for ext in IMG_EXTS:
            alt2 = dataset_root / 'images' / split / (Path(basename).with_suffix(ext).name)
            if alt2.exists():
                return alt2

    return None


def prepare_data(dataset_path: Path):
    print(f"Preparing data from {dataset_path}...")

    coco_annotations_path = dataset_path / 'coco_annotations.json'
    if not coco_annotations_path.exists():
        print(f"Error: {coco_annotations_path} not found. Please ensure the dataset is extracted correctly.")
        return

    with open(coco_annotations_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Categories -> YOLO class indices [0..N-1] in sorted order of category id (stable)
    categories_sorted = sorted(coco['categories'], key=lambda x: x['id'])
    cat_id_to_idx = {c['id']: i for i, c in enumerate(categories_sorted)}
    names = [c['name'] for c in categories_sorted]

    # Images and annotations
    img_meta = {im['id']: im for im in coco['images']}
    ann_by_img = defaultdict(list)
    for a in coco['annotations']:
        ann_by_img[a['image_id']].append(a)

    # Reproducible split# Reproducible split
    random.seed(42)
    all_image_ids = list(img_meta.keys())
    random.shuffle(all_image_ids)

    total_images = len(all_image_ids)

    # Use the entire dataset: 80% train, 10% val, 10% test (no caps)
    train_size = int(total_images * 0.8)
    val_size = int(total_images * 0.1)
    test_size = total_images - train_size - val_size  # ensures all images are used

    train_ids = all_image_ids[:train_size]
    val_ids = all_image_ids[train_size:train_size + val_size]
    test_ids = all_image_ids[train_size + val_size:train_size + val_size + test_size]

    print(f"Selected {len(train_ids)} training images, {len(val_ids)} validation images, {len(test_ids)} test images.")

    # Create YOLO folder structure
    for split in ["train", "val", "test"]:
        (OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)

    missing_counter = Counter()

    def write_split(ids, split):
        split_missing = 0
        for img_id in ids:
            im = img_meta[img_id]

            # Resolve the image path robustly
            original_img_path = _find_image_path(dataset_path, im['file_name'])
            if original_img_path is None:
                split_missing += 1
                # Capture a few example missing names, but don't spam the console
                if split_missing <= 10:
                    print(f"Warning: Image for file_name='{im['file_name']}' not found. (img_id={img_id}) Skipping.")
                continue

            # Destination image path: keep original basename & extension for safety with YOLO
            dest_img_name = original_img_path.name
            dest_img_path = OUTPUT_ROOT / 'images' / split / dest_img_name
            shutil.copy2(original_img_path, dest_img_path)

            # Write YOLO label(s): same basename as image (but .txt)
            W, H = im['width'], im['height']
            dest_label_path = OUTPUT_ROOT / 'labels' / split / (Path(dest_img_name).with_suffix('.txt').name)
            with open(dest_label_path, "w", encoding="utf-8") as lf:
                for a in ann_by_img[img_id]:
                    x, y, w, h = a["bbox"]  # COCO format: [x_min, y_min, width, height]
                    cx = (x + w / 2) / W
                    cy = (y + h / 2) / H
                    nw = w / W
                    nh = h / H
                    cls = cat_id_to_idx[a["category_id"]]
                    lf.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        missing_counter[split] = split_missing

    write_split(train_ids, "train")
    write_split(val_ids, "val")
    write_split(test_ids, "test")

    # Create data.yaml (plain string, no PyYAML dependency)
    names_yaml = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    data_yaml_content = (
        f"path: {OUTPUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names: {names_yaml}\n"
    )
    with open("doclaynet.yaml", "w", encoding="utf-8") as f:
        f.write(data_yaml_content)

    print("Data preparation complete. YOLO-formatted data and doclaynet.yaml created.")
    if sum(missing_counter.values()):
        print(
            f"Note: skipped {sum(missing_counter.values())} images "
            f"(train={missing_counter['train']}, val={missing_counter['val']}, test={missing_counter['test']})."
        )


if __name__ == "__main__":
    prepare_data(DATA_ROOT)
