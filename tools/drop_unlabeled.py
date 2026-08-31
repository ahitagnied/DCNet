#!/usr/bin/env python3
"""Randomly drop most images that have no annotations.

    python tools/drop_unlabeled.py /scratch/ad158/gollum-v4-tiled --keep-frac 0.05

Applied independently per split so labeled/unlabeled ratios stay comparable
across train/valid/test. Originals are saved once as *.bak_unlabeled;
rerunning always re-filters from that backup, so --keep-frac can be changed
without compounding. Only the json is edited -- image files on disk are left
alone (they're cheap to keep and the whole dataset is reproducible via
tools/tile_dataset.py anyway).
"""

import argparse
import json
import os
import random
import shutil


def filter_split(json_file, keep_frac, seed):
    backup = json_file + ".bak_unlabeled"
    if not os.path.exists(backup):
        shutil.copy2(json_file, backup)

    with open(backup) as f:
        data = json.load(f)

    labeled_ids = {a["image_id"] for a in data["annotations"]}
    labeled = [im for im in data["images"] if im["id"] in labeled_ids]
    unlabeled = [im for im in data["images"] if im["id"] not in labeled_ids]

    rng = random.Random(seed)
    rng.shuffle(unlabeled)
    n_keep = round(len(unlabeled) * keep_frac)
    kept_unlabeled = unlabeled[:n_keep]

    keep_ids = {im["id"] for im in labeled} | {im["id"] for im in kept_unlabeled}
    data["images"] = [im for im in data["images"] if im["id"] in keep_ids]

    with open(json_file, "w") as f:
        json.dump(data, f)

    return len(labeled), len(unlabeled), len(kept_unlabeled)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--keep-frac", type=float, default=0.05,
                    help="fraction of unlabeled images to keep (default 0.05 = drop 95%%)")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    for split in args.splits:
        jf = os.path.join(args.root, split, "_annotations.coco.json")
        if not os.path.exists(jf):
            print(f"{split}: missing {jf}")
            continue
        n_lab, n_unlab, n_kept_unlab = filter_split(jf, args.keep_frac, args.seed)
        n_before = n_lab + n_unlab
        n_after = n_lab + n_kept_unlab
        print(f"{split}: {n_before} -> {n_after} images "
              f"({n_lab} labeled + {n_kept_unlab}/{n_unlab} unlabeled kept, "
              f"dropped {n_unlab - n_kept_unlab})")


if __name__ == "__main__":
    main()
