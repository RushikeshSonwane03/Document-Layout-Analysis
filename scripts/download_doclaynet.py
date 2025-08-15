import json, os, shutil
from pathlib import Path

# >>> CHANGE THIS if your path differs
ROOT = Path(r"F:\DL\document_layout_analysis\data\DocLayNet")

PNG_DIR   = ROOT / "PNG"
COCO_DIR  = ROOT / "COCO"
OUT_IMG   = ROOT / "images"
OUT_JSON  = ROOT / "coco_annotations.json"

SPLITS = {
    "train": COCO_DIR / "train.json",
    "val":   COCO_DIR / "val.json",
    "test":  COCO_DIR / "test.json",
}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def hardlink_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if dst.exists():  # idempotent
        return
    try:
        os.link(src, dst)  # NTFS hardlink (same drive)
    except Exception:
        shutil.copy2(src, dst)

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # sanity
    assert PNG_DIR.exists(), f"Missing {PNG_DIR}"
    for s, p in SPLITS.items():
        assert p.exists(), f"Missing {p}"

    # Prepare output dirs
    for s in SPLITS.keys():
        ensure_dir(OUT_IMG / s)

    all_images = []
    all_anns   = []
    categories = None
    next_img_id = 1
    next_ann_id = 1

    for split, ann_path in SPLITS.items():
        data = load_json(ann_path)

        # categories should be identical across splits
        if categories is None:
            categories = data.get("categories", [])
        else:
            assert data.get("categories", []) == categories, "Categories mismatch across splits"

        old2new = {}

        # images
        for img in data["images"]:
            file_name = img["file_name"]   # e.g. "000abc...png"
            src = PNG_DIR / file_name
            dst = OUT_IMG / split / file_name

            if not src.exists():
                raise FileNotFoundError(f"Image not found: {src}")

            hardlink_or_copy(src, dst)

            new_img = dict(img)
            new_img["id"] = next_img_id
            new_img["file_name"] = str(Path("images") / split / file_name)
            old2new[img["id"]] = next_img_id
            all_images.append(new_img)
            next_img_id += 1

        # annotations
        for ann in data["annotations"]:
            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = old2new[ann["image_id"]]
            all_anns.append(new_ann)
            next_ann_id += 1

    merged = {
        "images": all_images,
        "annotations": all_anns,
        "categories": categories,
    }

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False)

    print("✅ Done")
    print("Images placed under:", OUT_IMG)
    print("Merged COCO JSON   :", OUT_JSON)
    print("Counts -> images:", len(all_images), "annotations:", len(all_anns), "categories:", len(categories))

if __name__ == "__main__":
    main()
