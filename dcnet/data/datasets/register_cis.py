import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

# Tiled (3x3, 448x448) + resplit COCO dataset produced by tools/tile_dataset.py.
# Unlike the earlier gollum-v4 export, this dataset's valid/ and test/ splits
# each have full class coverage on their own, so they're used directly instead
# of being merged into a valtest/ split.
ROOT = os.environ.get("GOLLUM_DATA_ROOT", "/scratch/ad158/gollum-tiled")
TRAIN_PATH = os.path.join(ROOT, "train")
VAL_PATH = os.path.join(ROOT, "valid")
TEST_PATH = os.path.join(ROOT, "test")

# Drop Roboflow project-name placeholder (no annotations)
DROP_CATEGORY_NAMES = {"gollum", "Gollum-GM"}


def _filtered_json(json_file: str) -> str:
    """Write sidecar JSON without dummy categories; return path to load."""
    out = json_file.replace(".json", ".filtered.json")
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(json_file):
        return out
    with open(json_file) as f:
        data = json.load(f)
    keep_ids = {
        c["id"]
        for c in data["categories"]
        if c["name"] not in DROP_CATEGORY_NAMES
    }
    data["categories"] = [c for c in data["categories"] if c["id"] in keep_ids]
    data["annotations"] = [
        a for a in data["annotations"] if a["category_id"] in keep_ids
    ]
    with open(out, "w") as f:
        json.dump(data, f)
    return out


TRAIN_JSON = _filtered_json(os.path.join(TRAIN_PATH, "_annotations.coco.json"))
VAL_JSON = _filtered_json(os.path.join(VAL_PATH, "_annotations.coco.json"))
TEST_JSON = _filtered_json(os.path.join(TEST_PATH, "_annotations.coco.json"))

PREDEFINED_SPLITS_DATASET = {
    "gollum_train": (TRAIN_PATH, TRAIN_JSON),
    "gollum_val": (VAL_PATH, VAL_JSON),
    "gollum_test": (TEST_PATH, TEST_JSON),
}


def register_dataset():
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
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
