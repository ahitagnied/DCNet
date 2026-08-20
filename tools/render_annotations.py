#!/usr/bin/env python3
"""Render annotation overlays to a folder of JPGs.

Useful when a browser-based viewer is not reachable: open the output folder in
any file browser and step through the images with arrow keys.

    python tools/render_annotations.py /scratch/ad158/gollum/gollum-v4 \
        --split train --out /scratch/ad158/gollum/gollum-v4/annotated_train
"""

import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_util

# BGR, matched to the html viewer's palette order
COLORS = [(107, 59, 255), (238, 211, 34), (53, 230, 163), (11, 158, 245)]


def polygons_of(ann, h, w):
    segm = ann.get("segmentation")
    if isinstance(segm, dict):
        rle = segm
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, h, w)
        m = mask_util.decode(rle)
        if m.ndim == 3:
            m = m.any(axis=2).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2).astype(np.int32) for c in cnts if len(c) >= 3]
    if isinstance(segm, list):
        return [np.asarray(p, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
                for p in segm if len(p) >= 6]
    return []


def poly_area(pts):
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="0 = all labeled images")
    p.add_argument("--include-empty", action="store_true")
    args = p.parse_args()

    split_dir = os.path.join(args.root, args.split)
    with open(os.path.join(split_dir, "_annotations.coco.json")) as f:
        data = json.load(f)

    names = [c["name"] for c in data["categories"] if c["name"].lower() != "gollum"]
    idx = {c["id"]: names.index(c["name"]) for c in data["categories"]
           if c["name"] in names}

    by_img = {}
    for ann in data["annotations"]:
        if ann["category_id"] in idx:
            by_img.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(args.out, exist_ok=True)
    for stale in os.listdir(args.out):
        os.remove(os.path.join(args.out, stale))

    records = [im for im in data["images"]
               if args.include_empty or by_img.get(im["id"])]
    records.sort(key=lambda im: -len(by_img.get(im["id"], [])))
    if args.limit:
        records = records[:args.limit]

    s = args.scale
    written = 0
    for rank, im in enumerate(records):
        img = cv2.imread(os.path.join(split_dir, im["file_name"]))
        if img is None:
            continue
        if s != 1:
            img = cv2.resize(img, (img.shape[1] * s, img.shape[0] * s),
                             interpolation=cv2.INTER_CUBIC)
        overlay = img.copy()
        anns = by_img.get(im["id"], [])
        for ann in anns:
            k = idx[ann["category_id"]]
            col = COLORS[k % len(COLORS)]
            for pts in polygons_of(ann, im["height"], im["width"]):
                area = poly_area(pts)
                sp = pts * s
                cv2.fillPoly(overlay, [sp], col)
                cv2.polylines(img, [sp], True, col, 1, cv2.LINE_AA)
                x, y = sp[:, 0].min(), sp[:, 1].min()
                tag = f"{names[k].split(' -')[0]} {area:.0f}px"
                cv2.putText(img, tag, (int(x), max(10, int(y) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, (0, 0, 0), 3)
                cv2.putText(img, tag, (int(x), max(10, int(y) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35 * s, col, 1, cv2.LINE_AA)
        out = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        cv2.putText(out, f"{rank + 1}/{len(records)}  {len(anns)} inst",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                    cv2.LINE_AA)
        name = f"{rank + 1:05d}_{len(anns):02d}inst_{im['file_name'][:40]}"
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            name += ".jpg"
        cv2.imwrite(os.path.join(args.out, name), out)
        written += 1

    print(f"wrote {written} images -> {args.out}")


if __name__ == "__main__":
    main()
