"""
ParkVision AI - preprocess.py (memory-safe version, capped at 2000/class)
"""

import json
import os
import cv2
import random
import shutil
import time

RAW_SPLITS = ["train", "valid", "test"]
DATA_DIR = "data"
TEMP_DIR = "data_temp"
OUTPUT_DIR = "data_processed"
IMG_SIZE = 224
MAX_PER_CLASS = 2000
random.seed(42)


def crop_and_save_temp():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(os.path.join(TEMP_DIR, "occupied"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_DIR, "empty"), exist_ok=True)

    counts = {"occupied": 0, "empty": 0}

    for split in RAW_SPLITS:
        if counts["occupied"] >= MAX_PER_CLASS and counts["empty"] >= MAX_PER_CLASS:
            break

        ann_path = os.path.join(DATA_DIR, split, "_annotations.coco.json")
        if not os.path.exists(ann_path):
            print(f"Skipping {split} (no annotation file found)")
            continue

        t0 = time.time()
        with open(ann_path) as f:
            coco = json.load(f)

        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        img_map = {img["id"]: img["file_name"] for img in coco["images"]}

        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        total_images = len(anns_by_image)
        print(f"{split}: {total_images} images available")

        for count, (image_id, anns) in enumerate(anns_by_image.items(), 1):
            if counts["occupied"] >= MAX_PER_CLASS and counts["empty"] >= MAX_PER_CLASS:
                print("Reached cap for both classes, stopping early.")
                break

            img_file = img_map.get(image_id)
            if img_file is None:
                continue
            img_path = os.path.join(DATA_DIR, split, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            for ann in anns:
                cat_name = cat_map.get(ann["category_id"], "")
                if cat_name not in ("space-empty", "space-occupied"):
                    continue

                label = "occupied" if cat_name == "space-occupied" else "empty"
                if counts[label] >= MAX_PER_CLASS:
                    continue

                x, y, w, h = [int(v) for v in ann["bbox"]]
                x, y = max(0, x), max(0, y)
                crop = img[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

                out_path = os.path.join(TEMP_DIR, label, f"{label}_{counts[label]:05d}.jpg")
                cv2.imwrite(out_path, crop)
                counts[label] += 1

            if count % 200 == 0 or count == total_images:
                elapsed = time.time() - t0
                print(f"  {split}: {count}/{total_images} images done ({elapsed:.1f}s) -> occupied={counts['occupied']}, empty={counts['empty']}")

        print(f"Finished {split} in {time.time()-t0:.1f}s. Totals -> occupied={counts['occupied']}, empty={counts['empty']}")

    return counts


def augment_and_save(src_path, dst_path):
    img = cv2.imread(src_path)
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    factor = random.uniform(0.7, 1.3)
    img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
    cv2.imwrite(dst_path, img)


def split_and_organize(label):
    src_dir = os.path.join(TEMP_DIR, label)
    files = sorted(os.listdir(src_dir))
    random.shuffle(files)

    n = len(files)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    splits = {
        "train": files[:train_end],
        "val": files[train_end:val_end],
        "test": files[val_end:]
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(OUTPUT_DIR, split_name, label)
        os.makedirs(out_dir, exist_ok=True)
        for fname in split_files:
            shutil.copy(os.path.join(src_dir, fname), os.path.join(out_dir, fname))
        if split_name == "train":
            for fname in split_files:
                aug_name = "aug_" + fname
                augment_and_save(os.path.join(src_dir, fname), os.path.join(out_dir, aug_name))

    return {k: len(v) for k, v in splits.items()}


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    print("Step 1: cropping and saving slots to disk (memory-safe, capped at", MAX_PER_CLASS, "per class)...")
    counts = crop_and_save_temp()
    print(f"\nCrop totals -> occupied={counts['occupied']}, empty={counts['empty']}")

    print("\nStep 2: splitting into train/val/test and augmenting training data...")
    occ_split = split_and_organize("occupied")
    emp_split = split_and_organize("empty")

    print("\nDone. Final structure saved under 'data_processed/':")
    print(f"  train: occupied~{occ_split['train']}(x2 with aug), empty~{emp_split['train']}(x2 with aug)")
    print(f"  val:   occupied={occ_split['val']}, empty={emp_split['val']}")
    print(f"  test:  occupied={occ_split['test']}, empty={emp_split['test']}")

    shutil.rmtree(TEMP_DIR)
    print("Cleaned up temp files.")


if __name__ == "__main__":
    main()
