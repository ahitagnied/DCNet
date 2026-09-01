#!/usr/bin/env python3
"""Drop 'blob' annotations that smear a class over most of the frame.

Some Roboflow masks cover a large fraction of the image (background painted as
icw/stainless-steel/etc. rather than the actual object). Tiling multiplies the
damage -- a full-frame blob becomes up to 9 tiles that are entirely one class --
so these are removed before tiling.

Only the offending annotation is dropped; other instances on the same image are
kept. Originals are backed up next to each JSON as *.orig.

    python tools/clean_blob_annotations.py /scratch/ad158/gollum --max-frac 0.10
"""
import argparse
import json
import os
import shutil
from collections import Counter

from pycocotools import mask as mask_util

SPLITS = ("train", "valid", "test")


def mask_fraction(segm, h, w):
    """Fraction of the frame covered by this annotation's mask."""
    rle = segm
    if isinstance(segm, dict):
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, h, w)
    else:
        rle = mask_util.merge(mask_util.frPyObjects(segm, h, w))
    return float(mask_util.area(rle)) / (h * w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--max-frac", type=float, default=0.10,
                   help="drop annotations covering more than this fraction of the frame")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    grand_dropped = Counter()
    grand_kept = 0

    for split in SPLITS:
        jf = os.path.join(args.root, split, "_annotations.coco.json")
        if not os.path.exists(jf):
            continue
        with open(jf) as f:
            d = json.load(f)

        cats = {c["id"]: c["name"] for c in d["categories"]}
        dims = {im["id"]: (im["height"], im["width"]) for im in d["images"]}

        keep, dropped = [], []
        for a in d["annotations"]:
            h, w = dims[a["image_id"]]
            frac = mask_fraction(a["segmentation"], h, w)
            if frac > args.max_frac:
                dropped.append((frac, cats[a["category_id"]]))
            else:
                keep.append(a)

        counts = Counter(n for _, n in dropped)
        grand_dropped.update(counts)
        grand_kept += len(keep)

        print(f"{split}: {len(d['annotations'])} -> {len(keep)} "
              f"({len(dropped)} dropped)")
        for frac, name in sorted(dropped, reverse=True):
            print(f"    drop {name:<16} {frac*100:6.2f}% of frame")

        if args.dry_run:
            continue

        backup = jf + ".orig"
        if not os.path.exists(backup):
            shutil.copy2(jf, backup)
        d["annotations"] = keep
        with open(jf, "w") as f:
            json.dump(d, f)

        # Stale sidecar from register_cis would otherwise shadow the clean file.
        side = jf.replace(".json", ".filtered.json")
        if os.path.exists(side):
            os.remove(side)

    print(f"\ntotal dropped: {sum(grand_dropped.values())} -> {dict(grand_dropped)}")
    print(f"total kept:    {grand_kept}")
    if args.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
