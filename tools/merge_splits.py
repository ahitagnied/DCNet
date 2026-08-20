#!/usr/bin/env python3
"""Merge COCO splits into one directory of symlinks plus a combined json.

    python tools/merge_splits.py /scratch/ad158/gollum/gollum-v4 \
        --splits valid test --out valtest

Image and annotation ids are re-indexed so they stay unique across sources, and
file names are prefixed with their split when two splits collide.
"""

import argparse
import json
import os
import shutil


def merge(root, splits, out_name):
    out_dir = os.path.join(root, out_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    images, annotations = [], []
    categories = None
    seen_names = {}
    next_img, next_ann = 1, 1

    for split in splits:
        src = os.path.join(root, split)
        with open(os.path.join(src, "_annotations.coco.json")) as f:
            data = json.load(f)

        if categories is None:
            categories = data["categories"]
        elif [c["name"] for c in categories] != [c["name"] for c in data["categories"]]:
            raise SystemExit(f"category mismatch between {splits[0]} and {split}")

        by_img = {}
        for ann in data["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)

        for im in data["images"]:
            name = im["file_name"]
            if name in seen_names:
                name = f"{split}_{name}"
            seen_names[name] = split

            link = os.path.join(out_dir, name)
            target = os.path.abspath(os.path.join(src, im["file_name"]))
            if not os.path.exists(target):
                continue
            os.symlink(target, link)

            old_id = im["id"]
            im = dict(im, id=next_img, file_name=name)
            images.append(im)

            for ann in by_img.get(old_id, []):
                annotations.append(dict(ann, id=next_ann, image_id=next_img))
                next_ann += 1
            next_img += 1

    merged = {"images": images, "annotations": annotations, "categories": categories}
    with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
        json.dump(merged, f)

    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    p.add_argument("--out", default="valtest")
    args = p.parse_args()

    m = merge(args.root, args.splits, args.out)
    cats = {c["id"]: c["name"] for c in m["categories"]}
    counts = {}
    for ann in m["annotations"]:
        name = cats[ann["category_id"]]
        counts[name] = counts.get(name, 0) + 1

    labeled = len({a["image_id"] for a in m["annotations"]})
    print(f"{args.out}: {len(m['images'])} images ({labeled} labeled), "
          f"{len(m['annotations'])} annotations")
    for name in sorted(counts):
        print(f"  {name[:34]:<36}{counts[name]:>6}")


if __name__ == "__main__":
    main()
