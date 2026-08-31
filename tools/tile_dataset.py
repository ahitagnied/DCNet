#!/usr/bin/env python3
"""Tile every image into a 3x3 grid, re-cut annotations per tile, and
re-split into train/valid/test by source frame.

    python tools/tile_dataset.py /scratch/ad158/gollum-v4 \
        --out /scratch/ad158/gollum-v4-tiled

Pools images from all input splits (train + valid), tiles each into 9 crops,
and decodes each RLE segmentation once per source image, then crops the mask
per tile and re-traces it into polygons (DCNet's dataset mapper needs
polygons, not RLE -- see tools/rle_to_polygon.py). Tiny fragments left behind
by the cut (< MIN_POLY_AREA px) are dropped.

The 80/10/10 split is done by source frame (the `extra.name` field Roboflow
stamps on every image), not by individual image or tile. Roboflow already
expanded each frame into up to 3 augmented copies, and tiling further expands
each image into 9 tiles; grouping by frame keeps every augmented copy and
every tile of the same frame on one side of the split so no near-duplicate
content leaks between train/val/test.
"""

import argparse
import json
import os
import random
import shutil
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

MIN_POLY_AREA = 4.0
GRID = 3
SPLITS = ("train", "valid", "test")
RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


def rle_to_mask(segm, h, w):
    rle = segm
    if isinstance(rle.get("counts"), list):
        rle = mask_util.frPyObjects(rle, h, w)
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m.any(axis=2).astype(np.uint8)
    return m


def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < MIN_POLY_AREA:
            continue
        polys.append(c.reshape(-1).astype(float).tolist())
    return polys


def polygon_area_and_bbox(polys):
    total_area = 0.0
    xs, ys = [], []
    for p in polys:
        x = np.asarray(p[0::2])
        y = np.asarray(p[1::2])
        total_area += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        xs.append(x)
        ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
    return float(total_area), [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def tile_edges(size, n=GRID):
    e = np.linspace(0, size, n + 1).round().astype(int).tolist()
    return list(zip(e[:-1], e[1:]))


def load_pool(root, in_splits):
    """Return list of image dicts (with _path, _anns, _split) and categories."""
    images = []
    categories = None
    for split in in_splits:
        jf = os.path.join(root, split, "_annotations.coco.json")
        with open(jf) as f:
            d = json.load(f)
        if categories is None:
            categories = d["categories"]
        elif [c["name"] for c in categories] != [c["name"] for c in d["categories"]]:
            raise SystemExit(f"category mismatch between {in_splits[0]} and {split}")

        by_img = defaultdict(list)
        for a in d["annotations"]:
            by_img[a["image_id"]].append(a)

        for im in d["images"]:
            images.append({
                **im,
                "_path": os.path.join(root, split, im["file_name"]),
                "_anns": by_img.get(im["id"], []),
                "_frame": im.get("extra", {}).get("name", im["file_name"]),
            })
    return images, categories


def assign_splits(images, seed):
    """Group images by source frame, greedily balance groups across splits."""
    groups = defaultdict(list)
    for im in images:
        groups[im["_frame"]].append(im)

    frames = list(groups.keys())
    random.Random(seed).shuffle(frames)

    counts = {s: 0 for s in SPLITS}
    total = 0
    assignment = {}
    for frame in frames:
        n = len(groups[frame])
        total += n
        best = max(SPLITS, key=lambda s: RATIOS[s] * total - counts[s])
        counts[best] += n
        assignment[frame] = best

    out = {s: [] for s in SPLITS}
    for frame, split in assignment.items():
        out[split].extend(groups[frame])
    return out


def tile_one_image(im, out_dir, next_img_id, next_ann_id, resize=None):
    img = Image.open(im["_path"]).convert("RGB")
    w, h = im["width"], im["height"]
    xb, yb = tile_edges(w), tile_edges(h)

    decoded = [(a, rle_to_mask(a["segmentation"], h, w) if isinstance(a["segmentation"], dict)
                else None) for a in im["_anns"]]

    stem, ext = os.path.splitext(im["file_name"])
    new_images, new_anns = [], []

    for row, (y0, y1) in enumerate(yb):
        for col, (x0, x1) in enumerate(xb):
            tw, th = x1 - x0, y1 - y0
            out_w, out_h = (resize, resize) if resize else (tw, th)
            sx, sy = out_w / tw, out_h / th
            tile_name = f"{stem}_t{row}{col}{ext}"
            tile_img = img.crop((x0, y0, x1, y1))
            if resize:
                tile_img = tile_img.resize((out_w, out_h), Image.LANCZOS)
            tile_img.save(os.path.join(out_dir, tile_name), quality=95)

            img_id = next_img_id()
            new_images.append({
                "id": img_id,
                "license": im.get("license", 1),
                "file_name": tile_name,
                "height": out_h,
                "width": out_w,
                "date_captured": im.get("date_captured", ""),
                "extra": {"name": im["_frame"], "tile": [row, col], "source_image": im["file_name"]},
            })

            for a, mask in decoded:
                if mask is None:
                    continue
                crop = mask[y0:y1, x0:x1]
                if not crop.any():
                    continue
                polys = mask_to_polygons(np.ascontiguousarray(crop))
                if not polys:
                    continue
                if resize:
                    polys = [[v * sx if k % 2 == 0 else v * sy for k, v in enumerate(p)] for p in polys]
                area, bbox = polygon_area_and_bbox(polys)
                new_anns.append({
                    "id": next_ann_id(),
                    "image_id": img_id,
                    "category_id": a["category_id"],
                    "segmentation": polys,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
    return new_images, new_anns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset root containing train/ valid/")
    p.add_argument("--out", required=True, help="output dataset root")
    p.add_argument("--in-splits", nargs="+", default=["train", "valid"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize", type=int, default=None,
                    help="resize each tile to a resize x resize square (default: no resize)")
    args = p.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} already exists; remove it or pick another --out")

    images, categories = load_pool(args.root, args.in_splits)
    by_split = assign_splits(images, args.seed)

    for split in SPLITS:
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir)

        next_img = iter_id()
        next_ann = iter_id()
        all_images, all_anns = [], []
        for im in by_split[split]:
            imgs, anns = tile_one_image(im, out_dir, next_img, next_ann, resize=args.resize)
            all_images.extend(imgs)
            all_anns.extend(anns)

        with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
            json.dump({"images": all_images, "annotations": all_anns, "categories": categories}, f)

        cats = {c["id"]: c["name"] for c in categories}
        counts = defaultdict(int)
        for a in all_anns:
            counts[cats.get(a["category_id"], a["category_id"])] += 1
        n_frames = len({im["_frame"] for im in by_split[split]})
        labeled = len({a["image_id"] for a in all_anns})
        print(f"\n=== {split}: {n_frames} frames -> {len(by_split[split])} images -> "
              f"{len(all_images)} tiles ({labeled} labeled), {len(all_anns)} annotations ===")
        for name in sorted(counts):
            print(f"  {name[:34]:<36}{counts[name]:>6}")


def iter_id():
    state = {"n": 0}
    def _next():
        v = state["n"]
        state["n"] += 1
        return v
    return _next


if __name__ == "__main__":
    main()
