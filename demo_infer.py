#!/usr/bin/env python3
"""Run DCNet on raw_data: strip blue rails, center-square crop, resize 448, demos + conf stats."""

import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from tqdm import tqdm


def largest_component_mask(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (drops soft-mask speckles)."""
    binary = (mask > 0.5).astype(np.uint8)
    if binary.max() == 0:
        return binary.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, dtype=bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i


def mask_aabb(mask: np.ndarray):
    """Tight AABB from mask pixels: inclusive (x0,y0,x1,y1), or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _put_label(img, text, x, y, color):
    """Outlined text anchored above-left of (x,y); no wide filled chip."""
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = int(np.clip(x, 0, img.shape[1] - tw - 1))
    ty = int(np.clip(y - 3, th + 1, img.shape[0] - 1))
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def draw_instances(bgr: np.ndarray, masks: np.ndarray, labels, alpha: float = 0.55) -> np.ndarray:
    """Overlay binary masks + AABB boxes that touch min/max mask pixels."""
    base = bgr.copy()
    tinted = bgr.copy()
    rng = np.random.RandomState(0)
    drawn = []
    for m, lab in zip(masks, labels):
        m = np.asarray(m, dtype=bool)
        if not m.any():
            continue
        color = tuple(int(x) for x in rng.randint(40, 255, size=3))
        tinted[m] = color
        drawn.append((mask_aabb(m), lab, color, m))
    out = cv2.addWeighted(tinted, alpha, base, 1.0 - alpha, 0)
    for box, lab, color, m in drawn:
        x0, y0, x1, y1 = box
        # 1px box exactly on mask extrema (inclusive)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, lineType=cv2.LINE_8)
        # mask outline so the box/mask relationship is obvious
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 1, lineType=cv2.LINE_8)
        _put_label(out, lab, x0, y0, color)
    return out


def side_by_side(raw_bgr: np.ndarray, pred_bgr: np.ndarray) -> np.ndarray:
    """Left=raw, right=pred, with captions."""
    h, w = raw_bgr.shape[:2]
    gap = 8
    canvas = np.zeros((h + 28, w * 2 + gap, 3), dtype=np.uint8)
    canvas[:] = 32
    canvas[28 : 28 + h, :w] = raw_bgr
    canvas[28 : 28 + h, w + gap :] = pred_bgr
    cv2.putText(canvas, "raw (crop+448)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "pred (mask+box)",
        (w + gap + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return canvas


from dcnet import add_dcnet_config, register_dataset
from dcnet.data.dataset_mappers.mapper_with_Fourier_amplitude import img_cut

CLASS_SHORT = {
    "Bare Wire": "Bare Wire",
    "ICW -Insulated Copper Wire-": "ICW",
    "SS-AL": "SS-AL",
}
THING_CLASSES = [
    "Bare Wire",
    "ICW -Insulated Copper Wire-",
    "SS-AL",
]


def setup(config_file, weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_dcnet_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DCNET.NUM_OBJECT_QUERIES = 30
    cfg.TEST.DETECTIONS_PER_IMAGE = 30
    cfg.freeze()
    default_setup(cfg, None)
    return cfg


def short_name(name):
    return CLASS_SHORT.get(name, name)


def remove_blue_rails_and_center_square(bgr: np.ndarray) -> np.ndarray:
    """Drop left/right blue rails, then take a centered square crop."""
    rgb = bgr[:, :, ::-1]
    R = rgb[:, :, 0].astype(np.int16)
    G = rgb[:, :, 1].astype(np.int16)
    B = rgb[:, :, 2].astype(np.int16)
    blue = (B > R + 35) & (B > G + 35) & (B > 80)
    col_f = blue.mean(axis=0)

    h, w = bgr.shape[:2]
    left = 0
    for i in range(w):
        if col_f[i] < 0.1:
            left = i
            break
    right = w - 1
    for i in range(w - 1, -1, -1):
        if col_f[i] < 0.1:
            right = i
            break

    # pad margins a bit so residual rail paint is gone
    pad = int(0.02 * w)
    left = min(left + pad, w // 3)
    right = max(right - pad, w - w // 3)
    if right <= left + 64:
        left, right = int(0.16 * w), int(0.84 * w)

    belt = bgr[:, left : right + 1]
    bh, bw = belt.shape[:2]
    side = min(bh, bw)
    y0 = (bh - side) // 2
    x0 = (bw - side) // 2
    return belt[y0 : y0 + side, x0 : x0 + side]


def prepare_model_input(bgr_448: np.ndarray):
    """RGB tensor + Fourier amplitude twin, CHW float."""
    image = bgr_448[:, :, ::-1].astype(np.float32)  # RGB
    fre = np.fft.fft2(image, axes=(0, 1))
    fre_a = np.abs(fre)
    fre_p = np.angle(fre)
    constant = fre_p.mean()
    fre_ = fre_a * np.e ** (1j * constant)
    img_a = np.abs(np.fft.ifft2(fre_, axes=(0, 1)))
    img_a = img_cut(img_a)

    image_t = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    image_a_t = torch.as_tensor(np.ascontiguousarray(img_a.transpose(2, 0, 1)))
    return {
        "image": image_t,
        "image_a": image_a_t,
        "height": 448,
        "width": 448,
    }


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="configs/CIS-R50.yaml")
    p.add_argument(
        "--weights",
        default="/scratch/ad158/gollum/output/gollum_r50-208006/model_final.pth",
    )
    p.add_argument("--raw-dir", default="raw_data")
    p.add_argument("--out-dir", default="demo")
    p.add_argument("--num-demos", type=int, default=50)
    p.add_argument("--score-thresh", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = setup(args.config_file, args.weights)
    register_dataset()
    for name in ("gollum_train", "gollum_val", "gollum_test"):
        MetadataCatalog.get(name).thing_classes = list(THING_CLASSES)

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)
    paths = sorted(
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    random.seed(args.seed)
    demo_paths = list(paths)
    random.shuffle(demo_paths)
    demo_paths = set(demo_paths[: args.num_demos])

    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for f in os.listdir(img_dir):
        os.remove(os.path.join(img_dir, f))
    print(f"Writing demos to {img_dir}", flush=True)

    conf_by_class = defaultdict(list)
    n_images = 0
    n_dets = 0
    demos_saved = 0

    with torch.no_grad():
        for path in tqdm(paths, desc="raw infer"):
            bgr = cv2.imread(path)
            if bgr is None:
                continue
            crop = remove_blue_rails_and_center_square(bgr)
            bgr448 = cv2.resize(crop, (448, 448), interpolation=cv2.INTER_LINEAR)
            inp = prepare_model_input(bgr448)
            outputs = model([inp])[0]["instances"].to("cpu")

            scores = outputs.scores.numpy()
            classes = outputs.pred_classes.numpy()
            keep = scores >= args.score_thresh
            scores = scores[keep]
            classes = classes[keep]
            masks = outputs.pred_masks.numpy()[keep] if outputs.has("pred_masks") else None

            n_images += 1
            n_dets += len(scores)
            for s, c in zip(scores, classes):
                cname = THING_CLASSES[int(c)] if int(c) < len(THING_CLASSES) else str(c)
                conf_by_class[cname].append(float(s))

            if path in demo_paths and demos_saved < args.num_demos:
                raw_panel = bgr448.copy()
                if len(scores):
                    if masks is not None:
                        mh, mw = masks.shape[-2:]
                        if (mh, mw) != (448, 448):
                            m = torch.as_tensor(masks).float().unsqueeze(1)
                            m = torch.nn.functional.interpolate(
                                m, size=(448, 448), mode="bilinear", align_corners=False
                            )
                            masks = (m.squeeze(1) > 0.5).numpy()
                        else:
                            masks = masks > 0.5
                        masks = np.stack([largest_component_mask(m) for m in masks])
                    else:
                        masks = np.zeros((0, 448, 448), dtype=bool)

                    labels = [
                        f"{short_name(THING_CLASSES[int(c)])} {s:.2f}"
                        for c, s in zip(classes, scores)
                    ]
                    pred_panel = draw_instances(bgr448, masks, labels, alpha=0.45)
                else:
                    pred_panel = bgr448.copy()
                    cv2.putText(
                        pred_panel,
                        "no dets",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                vis_img = side_by_side(raw_panel, pred_panel)
                base = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(img_dir, f"{demos_saved:02d}_{base}.jpg")
                if not cv2.imwrite(out_path, vis_img):
                    raise RuntimeError(f"cv2.imwrite failed: {out_path}")
                demos_saved += 1

    stats = {
        "weights": args.weights,
        "raw_dir": raw_dir,
        "preprocess": "remove blue rails -> center square -> resize 448x448",
        "score_thresh": args.score_thresh,
        "num_images": n_images,
        "num_detections": n_dets,
        "demos_saved": demos_saved,
        "per_class": {},
    }
    lines = [
        f"weights: {args.weights}",
        f"raw_dir: {raw_dir}",
        "preprocess: blue-rail crop -> center square -> 448x448",
        f"images: {n_images}  detections@{args.score_thresh}: {n_dets}  demos: {demos_saved}",
        "",
        f"{'class':<32} {'n':>6} {'mean':>7} {'median':>7} {'p25':>7} {'p75':>7} {'min':>7} {'max':>7}",
    ]
    for cname in THING_CLASSES:
        xs = np.array(conf_by_class.get(cname, []), dtype=np.float64)
        if len(xs) == 0:
            entry = {"n": 0}
            lines.append(f"{short_name(cname):<32} {0:6d}")
        else:
            entry = {
                "n": int(len(xs)),
                "mean": float(xs.mean()),
                "median": float(np.median(xs)),
                "p25": float(np.percentile(xs, 25)),
                "p75": float(np.percentile(xs, 75)),
                "min": float(xs.min()),
                "max": float(xs.max()),
            }
            lines.append(
                f"{short_name(cname):<32} {entry['n']:6d} {entry['mean']:7.3f} "
                f"{entry['median']:7.3f} {entry['p25']:7.3f} {entry['p75']:7.3f} "
                f"{entry['min']:7.3f} {entry['max']:7.3f}"
            )
        stats["per_class"][cname] = entry

    with open(os.path.join(out_dir, "confidence_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(out_dir, "confidence_stats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote demos -> {img_dir} ({len(os.listdir(img_dir))} files)")


if __name__ == "__main__":
    main()
