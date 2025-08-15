# utils/prepare_yolo_labels.py
import os, json, re
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(r"F:\DL\document_layout_analysis\data\DocLayNet")
COCO_JSON = DATASET_ROOT / "coco_annotations.json"
IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"

def main():
    assert COCO_JSON.exists(), f"Missing {COCO_JSON}"
    for s in ["train","val","test"]:
        assert (IMAGES_DIR/s).exists(), f"Missing {IMAGES_DIR/s}"
        (LABELS_DIR/s).mkdir(parents=True, exist_ok=True)

    with COCO_JSON.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = sorted(coco["categories"], key=lambda c: c["id"])
    cat_id2idx = {c["id"]: i for i,c in enumerate(cats)}
    img_meta = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    n_imgs, n_lines = 0, 0
    for img_id, im in img_meta.items():
        # im["file_name"] like "images/train/xxxx.png"
        file_rel = im["file_name"].replace("\\","/")
        m = re.match(r"images/(train|val|test)/(.+)$", file_rel)
        if not m:
            continue
        split, fname = m.group(1), m.group(2)
        stem = Path(fname).stem
        W, H = int(im["width"]), int(im["height"])
        lines = []
        for a in anns_by_img[img_id]:
            x, y, w, h = a["bbox"]
            cx, cy = (x + w/2)/W, (y + h/2)/H
            nw, nh = w/W, h/H
            cls = cat_id2idx[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (LABELS_DIR/split/f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_imgs += 1
        n_lines += len(lines)

    Path("doclaynet.yaml").write_text(
        f"""# DocLayNet YOLO config
path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val
test: images/test
names: {[c['name'] for c in cats]}
""",
        encoding="utf-8",
    )
    print("✅ labels ->", LABELS_DIR)
    print("✅ yaml   ->", Path("doclaynet.yaml").resolve())
    print("counts   -> images:", n_imgs, "boxes:", n_lines)

if __name__ == "__main__":
    main()
