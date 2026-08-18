import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

# Unpacked Roboflow COCO export lives directly under scratch/gollum
ROOT = os.environ.get("GOLLUM_DATA_ROOT", "/scratch/ad158/gollum")
TRAIN_PATH = os.path.join(ROOT, "train")
VAL_PATH = os.path.join(ROOT, "valid")
TEST_PATH = os.path.join(ROOT, "test")

# Drop Roboflow project-name placeholder (no annotations)
DROP_CATEGORY_NAMES = {"gollum", "Gollum-GM"}


def _filtered_json(json_file: str) -> str:
    """Write sidecar JSON without dummy categories; return path to load.

    Also remaps remaining category ids to contiguous 1..K (and rewrites
    annotation category_id). Detectron2 only builds an id→[0..K) map when ids
    are *not* already ``1..K``; without that remap, GT labels stay as 1..K while
    ``num_classes=K``, so class 0 is unused and category K collides with the
    no-object logit (Aluminium disappeared; every label shifted).
    """
    out = json_file.replace(".json", ".filtered.json")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(json_file):
        return out
    with open(json_file) as f:
        data = json.load(f)
    keep = sorted(
        (
            c
            for c in data["categories"]
            if c["name"] not in DROP_CATEGORY_NAMES
        ),
        key=lambda c: c["id"],
    )
    old_to_new = {c["id"]: i + 1 for i, c in enumerate(keep)}
    data["categories"] = [
        {**c, "id": old_to_new[c["id"]]} for c in keep
    ]
    data["annotations"] = [
        {**a, "category_id": old_to_new[a["category_id"]]}
        for a in data["annotations"]
        if a["category_id"] in old_to_new
    ]
    with open(out, "w") as f:
        json.dump(data, f)
    return out


def _splits():
    """Resolve train/val/test paths lazily — never at import time.

    Import-time filesystem access breaks on-device inference (Jetson has no
    /scratch/... Roboflow tree); training still calls register_dataset().
    """
    return {
        "gollum_train": (
            TRAIN_PATH,
            _filtered_json(os.path.join(TRAIN_PATH, "_annotations.coco.json")),
        ),
        "gollum_val": (
            VAL_PATH,
            _filtered_json(os.path.join(VAL_PATH, "_annotations.coco.json")),
        ),
        "gollum_test": (
            TEST_PATH,
            _filtered_json(os.path.join(TEST_PATH, "_annotations.coco.json")),
        ),
    }


def register_dataset():
    for key, (image_root, json_file) in _splits().items():
        register_dataset_instances(name=key, json_file=json_file, image_root=image_root)


def register_dataset_instances(name, json_file, image_root):
    DatasetCatalog.register(
        name, lambda j=json_file, r=image_root, n=name: load_coco_json(j, r, n)
    )
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco"
    )
    # Populate thing_classes before model build
    DatasetCatalog.get(name)
