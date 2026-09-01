#!/usr/bin/env python3
"""Remove a category (and any of its annotations) from a COCO dataset.

Roboflow stamps the project name as a placeholder category with no annotations
("ss-al" here). Leaving it in inflates num_classes and permanently wastes an
object-query slot on a class that can never appear.

Remaining category ids are left as-is; detectron2 maps them to a contiguous
range at load time.

    python tools/drop_category.py /scratch/ad158/gollum-tiled-v8 --name ss-al
"""
import argparse
import json
import os
import shutil

SPLITS = ("train", "valid", "test")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--name", required=True, help="category name to remove")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for split in SPLITS:
        jf = os.path.join(args.root, split, "_annotations.coco.json")
        if not os.path.exists(jf):
            continue
        with open(jf) as f:
            d = json.load(f)

        victims = {c["id"] for c in d["categories"] if c["name"] == args.name}
        if not victims:
            print(f"{split}: '{args.name}' not present")
            continue

        n_ann = sum(1 for a in d["annotations"] if a["category_id"] in victims)
        d["categories"] = [c for c in d["categories"] if c["id"] not in victims]
        d["annotations"] = [a for a in d["annotations"] if a["category_id"] not in victims]

        print(f"{split}: dropped '{args.name}' (ids {sorted(victims)}) "
              f"and {n_ann} annotations -> "
              f"{[c['name'] for c in d['categories']]}")

        if args.dry_run:
            continue

        backup = jf + ".bak_category"
        if not os.path.exists(backup):
            shutil.copy2(jf, backup)
        with open(jf, "w") as f:
            json.dump(d, f)

        side = jf.replace(".json", ".filtered.json")
        if os.path.exists(side):
            os.remove(side)

    if args.dry_run:
        print("(dry run -- nothing written)")


if __name__ == "__main__":
    main()
