#!/usr/bin/env python3
"""Drop annotations whose mask area falls below a per-class threshold.

    python tools/filter_small_instances.py /scratch/ad158/gollum/gollum-v4 \
        --min-area "ICW=300" "Bare Wire=180"

Class names may be prefixes, so "ICW" matches "ICW -Insulated Copper Wire-".
Originals are saved once as *.bak_prefilter; rerunning always re-filters from
that backup, so thresholds can be changed without compounding.
"""

import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np


def polygon_area(segm):
    total = 0.0
    if not isinstance(segm, list):
        return total
    for p in segm:
        if len(p) < 6:
            continue
        x = np.asarray(p[0::2])
        y = np.asarray(p[1::2])
        total += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(total)


def resolve(name, thresholds):
    for key, val in thresholds.items():
        if name == key or name.startswith(key):
            return val
    return 0.0


def filter_split(json_file, thresholds):
    backup = json_file + ".bak_prefilter"
    if not os.path.exists(backup):
        shutil.copy2(json_file, backup)

    with open(backup) as f:
        data = json.load(f)

    cats = {c["id"]: c["name"] for c in data["categories"]}
    before, after = Counter(), Counter()
    imgs_before, imgs_after = set(), set()
    kept = []

    for ann in data["annotations"]:
        name = cats.get(ann["category_id"], "")
        before[name] += 1
        imgs_before.add(ann["image_id"])
        if polygon_area(ann.get("segmentation")) < resolve(name, thresholds):
            continue
        after[name] += 1
        imgs_after.add(ann["image_id"])
        kept.append(ann)

    data["annotations"] = kept
    with open(json_file, "w") as f:
        json.dump(data, f)

    side = json_file.replace(".json", ".filtered.json")
    if os.path.exists(side):
        os.remove(side)

    return before, after, len(data["images"]), len(imgs_before), len(imgs_after)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--min-area", nargs="+", required=True,
                   help='e.g. "ICW=300" "Bare Wire=180"')
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    args = p.parse_args()

    thresholds = {}
    for item in args.min_area:
        key, _, val = item.rpartition("=")
        thresholds[key.strip()] = float(val)
    print("thresholds:", {k: int(v) for k, v in thresholds.items()})

    totals = Counter()
    kept_totals = Counter()
    for split in args.splits:
        jf = os.path.join(args.root, split, "_annotations.coco.json")
        if not os.path.exists(jf):
            print(f"{split}: missing")
            continue
        before, after, n_img, lab_b, lab_a = filter_split(jf, thresholds)
        totals.update(before)
        kept_totals.update(after)
        print(f"\n=== {split} === images {n_img}: "
              f"labeled {lab_b} -> {lab_a} (empty {n_img-lab_b} -> {n_img-lab_a})")
        print(f"{'class':<32}{'before':>8}{'kept':>8}{'dropped':>9}{'kept%':>7}")
        for name in sorted(before):
            if not before[name]:
                continue
            b, a = before[name], after[name]
            print(f"{name[:31]:<32}{b:>8}{a:>8}{b-a:>9}{100*a/b:>6.0f}%")

    print("\n=== TOTAL ===")
    print(f"{'class':<32}{'before':>8}{'kept':>8}{'dropped':>9}{'kept%':>7}")
    for name in sorted(totals):
        b, a = totals[name], kept_totals[name]
        if b:
            print(f"{name[:31]:<32}{b:>8}{a:>8}{b-a:>9}{100*a/b:>6.0f}%")


if __name__ == "__main__":
    main()
