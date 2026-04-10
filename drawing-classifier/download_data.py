"""
AR-BOOK Drawing Classifier — Data Downloader

Downloads and merges two data sources:
1. Quick Draw (.npy → inverted PNG, black-on-white)
2. ImageNet-Sketch (HuggingFace → PNG, various sketch styles)

Both are organized into ImageFolder structure: data/{train,val}/{class}/
"""

import os
import shutil
import urllib.request
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download

import config


# ============================================================
# Quick Draw
# ============================================================

def download_quickdraw_npy(category: str, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{category}.npy"
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        print(f"    [skip] {filename}")
        return filepath
    url_name = category.replace(" ", "%20")
    url = f"{config.QUICKDRAW_URL}/{url_name}.npy"
    print(f"    [download] {url}")
    urllib.request.urlretrieve(url, filepath)
    return filepath


def load_quickdraw():
    """Load Quick Draw data and return (images, labels) arrays."""
    raw_dir = os.path.join(config.DATA_DIR, "raw")
    all_images, all_labels = [], []

    for super_cls, sub_map in config.QUICKDRAW_MAP.items():
        print(f"\n[Quick Draw — {super_cls}]")
        for sub, samples in sub_map.items():
            print(f"  {sub}:")
            npy_path = download_quickdraw_npy(sub, raw_dir)
            data = np.load(npy_path)
            n = min(samples, len(data))
            data = data[:n]
            all_images.append(data)
            all_labels.extend([super_cls] * n)
            print(f"    loaded {n} → '{super_cls}'")

    return np.concatenate(all_images, axis=0), all_labels


def save_quickdraw_images(images, labels, split, offset=0):
    """Save Quick Draw numpy arrays as inverted PNGs."""
    counter = {}
    for img_array, label in zip(images, labels):
        class_dir = os.path.join(config.DATA_DIR, split, label)
        os.makedirs(class_dir, exist_ok=True)
        idx = counter.get(label, 0) + offset
        counter[label] = counter.get(label, 0) + 1
        inverted = 255 - img_array.reshape(28, 28)
        img = Image.fromarray(inverted, mode="L")
        img.save(os.path.join(class_dir, f"qd_{idx:05d}.png"))
    return counter


# ============================================================
# ImageNet-Sketch
# ============================================================

def download_imagenet_sketch():
    """Load ImageNet-Sketch images from local cache or HuggingFace."""
    print("\n[ImageNet-Sketch]")

    sketch_dir = os.path.join(config.DATA_DIR, "imagenet_sketch_raw", "sketch")
    all_images = []  # list of (PIL.Image, super_class)

    for super_cls, wnid_map in config.IMAGENET_SKETCH_MAP.items():
        print(f"\n  [{super_cls}]")
        for wnid, name in wnid_map.items():
            local_dir = os.path.join(sketch_dir, wnid)

            if os.path.isdir(local_dir):
                # Load from local cache
                count = 0
                for fname in sorted(os.listdir(local_dir)):
                    if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        try:
                            img = Image.open(os.path.join(local_dir, fname)).convert("RGB")
                            all_images.append((img, super_cls))
                            count += 1
                        except Exception:
                            continue
                print(f"    {name} ({wnid}): {count} images [local] → '{super_cls}'")
            else:
                # Fallback: download from HuggingFace
                try:
                    from huggingface_hub import list_repo_tree
                    prefix = f"data/{wnid}"
                    files = [f for f in list_repo_tree("songweig/imagenet_sketch", repo_type="dataset")
                             if hasattr(f, 'rfilename') and f.rfilename.startswith(prefix)
                             and f.rfilename.endswith(('.JPEG', '.jpg', '.png', '.jpeg'))]

                    if not files:
                        print(f"    {name} ({wnid}): no files found, skipping")
                        continue

                    count = 0
                    for f in files:
                        try:
                            local_path = hf_hub_download(
                                "songweig/imagenet_sketch",
                                f.rfilename,
                                repo_type="dataset",
                            )
                            img = Image.open(local_path).convert("RGB")
                            all_images.append((img, super_cls))
                            count += 1
                        except Exception:
                            continue

                    print(f"    {name} ({wnid}): {count} images [HF] → '{super_cls}'")
                except Exception as e:
                    print(f"    {name} ({wnid}): error - {e}")

    return all_images


def save_imagenet_sketch_images(images_with_labels, split, offset_counters=None):
    """Save ImageNet-Sketch PIL images as PNGs."""
    if offset_counters is None:
        offset_counters = {}
    counter = {}
    for img, label in images_with_labels:
        class_dir = os.path.join(config.DATA_DIR, split, label)
        os.makedirs(class_dir, exist_ok=True)
        base = offset_counters.get(label, 0)
        idx = counter.get(label, 0) + base
        counter[label] = counter.get(label, 0) + 1
        # Resize to reasonable size, save as RGB
        img_resized = img.resize((224, 224))
        img_resized.save(os.path.join(class_dir, f"is_{idx:05d}.png"))
    return counter


# ============================================================
# Main
# ============================================================

def main():
    # Clean previous splits (preserve train/animal — manually curated)
    PRESERVE = [("train", "animal")]
    for split in ["train", "val"]:
        split_dir = os.path.join(config.DATA_DIR, split)
        if os.path.exists(split_dir):
            for cls_name in os.listdir(split_dir):
                if (split, cls_name) in PRESERVE:
                    print(f"Preserved: {split}/{cls_name}")
                    continue
                cls_path = os.path.join(split_dir, cls_name)
                if os.path.isdir(cls_path):
                    shutil.rmtree(cls_path)
                    print(f"Cleaned: {split}/{cls_name}")

    # --- Quick Draw ---
    qd_images, qd_labels = load_quickdraw()

    # --- ImageNet-Sketch ---
    is_data = download_imagenet_sketch()

    # Use all Quick Draw data (no balancing — synth_augment handles domain gap)

    # Split Quick Draw
    qd_train_imgs, qd_val_imgs, qd_train_labels, qd_val_labels = train_test_split(
        qd_images, qd_labels,
        test_size=config.VAL_RATIO, stratify=qd_labels, random_state=42,
    )
    print(f"\nQuick Draw (balanced): train={len(qd_train_imgs)}, val={len(qd_val_imgs)}")

    # Filter out train/animal (preserved manually curated data)
    skip_set = {(split, cls) for split, cls in PRESERVE}
    train_mask = [i for i, l in enumerate(qd_train_labels) if ("train", l) not in skip_set]
    qd_train_imgs_filtered = qd_train_imgs[train_mask]
    qd_train_labels_filtered = [qd_train_labels[i] for i in train_mask]

    qd_train_counts = save_quickdraw_images(qd_train_imgs_filtered, qd_train_labels_filtered, "train")
    qd_val_counts = save_quickdraw_images(qd_val_imgs, qd_val_labels, "val")

    # Save ImageNet-Sketch
    if is_data:
        is_images, is_labels = zip(*is_data)
        is_images, is_labels = list(is_images), list(is_labels)
        is_train, is_val, is_train_labels, is_val_labels = train_test_split(
            list(zip(is_images, is_labels)),
            [l for l in is_labels],
            test_size=config.VAL_RATIO, stratify=is_labels, random_state=42,
        )
        # Filter out train/animal (preserved)
        is_train_filtered = [(img, lbl) for img, lbl in is_train if ("train", lbl) not in skip_set]
        print(f"\nImageNet-Sketch: train={len(is_train_filtered)} (excl. preserved), val={len(is_val)}")
        save_imagenet_sketch_images(is_train_filtered, "train", qd_train_counts)
        save_imagenet_sketch_images(is_val, "val", qd_val_counts)
    else:
        print("\nWARNING: No ImageNet-Sketch images downloaded")

    # --- Summary ---
    print("\n=== Summary ===")
    total = 0
    for split in ["train", "val"]:
        for cls in config.CLASSES:
            d = os.path.join(config.DATA_DIR, split, cls)
            count = len(os.listdir(d)) if os.path.exists(d) else 0
            total += count
            print(f"  {split}/{cls}: {count}")
    print(f"\nTotal: {total} images")
    print("Done.")


if __name__ == "__main__":
    main()
