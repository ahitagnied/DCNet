#!/usr/bin/env python3
"""Convert RLE segmentations in a COCO json to polygons.

DCNet's dataset mapper builds PolygonMasks unconditionally and reads
`.polygons`, so RLE annotations cannot be loaded. This rewrites them in place
(originals saved as *.bak_rle) and also recomputes `area` from the mask, since
Roboflow exports store bbox area there.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import cv2
import numpy as np
from pycocotools import mask as mask_util

MIN_POLY_AREA = 4.0


def rle_to_polygons(segm, height, width):
    """Return (polygons, mask_area). Polygons are COCO-style flat coord lists."""
    rle = segm
    if isinstance(rle.get("counts"), list):
        rle = mask_util.frPyObjects(rle, height, width)
    mask = mask_util.decode(rle)
    if mask.ndim == 3:
        mask = mask.any(axis=2).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3 or cv2.contourArea(c) < MIN_POLY_AREA:
            continue
        polys.append(c.reshape(-1).astype(float).tolist())
    return polys, float(mask.sum())


def polygon_area(polys):
    total = 0.0
    for p in polys:
        if len(p) < 6:
            continue
        x = np.asarray(p[0::2])
        y = np.asarray(p[1::2])
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(total)


def convert(json_file, fix_area=True):
    backup = json_file + ".bak_rle"
    if not os.path.exists(backup):
        shutil.copy2(json_file, backup)

    with open(backup) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    sizes = {im["id"]: (im["height"], im["width"]) for im in data["images"]}

    kept, dropped = [], defaultdict(int)
    delta = defaultdict(list)
    converted = defaultdict(int)

    for ann in data["annotations"]:
        name = cats.get(ann["category_id"], str(ann["category_id"]))
        segm = ann.get("segmentation")
        h, w = sizes[ann["image_id"]]

        if isinstance(segm, dict):
            polys, mask_area = rle_to_polygons(segm, h, w)
            if not polys:
                dropped[name] += 1
                continue
            ann["segmentation"] = polys
            converted[name] += 1
            poly_area = polygon_area(polys)
            if mask_area > 0:
                delta[name].append(poly_area / mask_area)
            if fix_area:
                ann["area"] = poly_area
        elif isinstance(segm, list):
            polys = [p for p in segm if len(p) >= 6]
            if not polys:
                dropped[name] += 1
                continue
            ann["segmentation"] = polys
            if fix_area:
                ann["area"] = polygon_area(polys)
        else:
            dropped[name] += 1
            continue

        kept.append(ann)

    data["annotations"] = kept
    with open(json_file, "w") as f:
        json.dump(data, f)

    return converted, dropped, delta, len(kept)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="dataset root containing train/ valid/ test/")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--keep-area", action="store_true", help="do not recompute area")
    args = p.parse_args()

    for split in args.splits:
        jf = os.path.join(args.root, split, "_annotations.coco.json")
        if not os.path.exists(jf):
            print(f"{split}: missing {jf}")
            continue
        converted, dropped, delta, n = convert(jf, fix_area=not args.keep_area)
        print(f"\n=== {split} -> {n} annotations ===")
        for name in sorted(set(converted) | set(dropped)):
            ratios = delta.get(name, [])
            ratio = f"  poly/mask area x{np.median(ratios):.3f}" if ratios else ""
            print(
                f"  {name[:34]:<36} rle->poly {converted.get(name, 0):>5}"
                f"  dropped {dropped.get(name, 0):>3}{ratio}"
            )
        # stale sidecar from register_cis
        side = jf.replace(".json", ".filtered.json")
        if os.path.exists(side):
            os.remove(side)
            print("  removed stale", os.path.basename(side))


if __name__ == "__main__":
    main()
