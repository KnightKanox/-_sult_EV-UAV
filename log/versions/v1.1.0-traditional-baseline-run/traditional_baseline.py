#!/usr/bin/env python3
import argparse
from pathlib import Path


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "evisseg_evuav.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional non-model baseline for EV-UAV")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to yaml config")
    parser.add_argument("--mode", "--split", dest="mode", default="test", choices=("train", "val", "test"), help="dataset split")
    parser.add_argument("--data-root", default=None, help="override DATA.root in config")
    parser.add_argument("--threshold", type=float, default=2.0, help="density threshold multiplier over positive cell mean")
    parser.add_argument("--min-area", type=int, default=3, help="minimum connected component area in pixels")
    parser.add_argument("--limit", type=int, default=None, help="maximum number of npz samples to evaluate")
    return parser.parse_args()


def load_runtime_dependencies():
    import numpy as np
    import yaml

    return np, yaml


def load_config(path, yaml_module):
    with open(path, "r", encoding="utf-8") as f:
        return yaml_module.safe_load(f)


def connected_components_with_stats(mask, np_module):
    try:
        import cv2

        return cv2.connectedComponentsWithStats(mask, connectivity=8)
    except ImportError:
        from scipy import ndimage

        structure = np_module.ones((3, 3), dtype=np_module.uint8)
        labels, num_labels = ndimage.label(mask, structure=structure)
        stats = np_module.zeros((num_labels + 1, 5), dtype=np_module.int64)
        centroids = np_module.zeros((num_labels + 1, 2), dtype=np_module.float64)
        for label_id in range(1, num_labels + 1):
            ys, xs = np_module.where(labels == label_id)
            if ys.size == 0:
                continue
            stats[label_id] = [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, ys.size]
            centroids[label_id] = [xs.mean(), ys.mean()]
        return num_labels + 1, labels, stats, centroids


def event_xy(data, res, np):
    width, height = int(res[0]), int(res[1])
    n_events = len(data["evs_norm"])

    if "ev" in data and getattr(data["ev"].dtype, "names", None):
        ev = data["ev"]
        if "x" in ev.dtype.names and "y" in ev.dtype.names:
            return clip_xy(ev["x"], ev["y"], width, height, np)

    if "ev_loc" in data and data["ev_loc"].shape[0] == n_events:
        loc = data["ev_loc"]
        # ev_loc is used by the model as 3D coordinates. In exported samples the
        # first two columns align with the image plane; clipping keeps this robust
        # for integer grid coordinates near the configured resolution.
        return clip_xy(loc[:, 0], loc[:, 1], width, height, np)

    evs_norm = data["evs_norm"]
    return normalized_xy(evs_norm[:, 0], evs_norm[:, 1], width, height, np)


def clip_xy(x_values, y_values, width, height, np):
    x = np.rint(np.asarray(x_values, dtype=np.float64)).astype(np.int64)
    y = np.rint(np.asarray(y_values, dtype=np.float64)).astype(np.int64)
    return np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)


def normalized_xy(x_values, y_values, width, height, np):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if np.nanmax(x) <= 1.5 and np.nanmin(x) >= -0.5:
        x = x * (width - 1)
    if np.nanmax(y) <= 1.5 and np.nanmin(y) >= -0.5:
        y = y * (height - 1)
    return clip_xy(x, y, width, height, np)


def predict_sample(npz_path, res, threshold, min_area, np):
    with np.load(npz_path, allow_pickle=False) as data:
        evs_norm = data["evs_norm"]
        seg_label = (evs_norm[:, 4] > 0).astype(np.uint8)
        x, y = event_xy(data, res, np)

    height, width = int(res[1]), int(res[0])
    density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (y, x), 1.0)

    positive_density = density[density > 0]
    if positive_density.size == 0:
        return np.zeros_like(seg_label), seg_label

    cutoff = max(1.0, float(positive_density.mean()) * threshold)
    mask = (density >= cutoff).astype(np.uint8)

    num_labels, labels, stats, _ = connected_components_with_stats(mask, np)
    kept = np.zeros_like(mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, 4] >= min_area:
            kept[labels == label_id] = 1

    seg_pred = kept[y, x].astype(np.uint8)
    return seg_pred, seg_label


def metrics_for_counts(tp, fp, fn, gt_positive, pred_positive, total):
    union = tp + fp + fn
    return {
        "iou": tp / union if union else 0.0,
        "seg_acc": tp / gt_positive if gt_positive else 0.0,
        "recall": tp / gt_positive if gt_positive else 0.0,
        "precision": tp / pred_positive if pred_positive else 0.0,
        "pred_positive": int(pred_positive),
        "gt_positive": int(gt_positive),
        "total_events": int(total),
    }


def evaluate_file(npz_path, res, threshold, min_area, np):
    pred, label = predict_sample(npz_path, res, threshold, min_area, np)
    pred_bool = pred.astype(bool)
    label_bool = label.astype(bool)
    tp = int(np.logical_and(pred_bool, label_bool).sum())
    fp = int(np.logical_and(pred_bool, ~label_bool).sum())
    fn = int(np.logical_and(~pred_bool, label_bool).sum())
    return metrics_for_counts(tp, fp, fn, int(label_bool.sum()), int(pred_bool.sum()), len(label_bool))


def main():
    args = parse_args()
    np, yaml_module = load_runtime_dependencies()
    cfg = load_config(args.config, yaml_module)
    data_root = Path(args.data_root or cfg["DATA"]["root"])
    res = cfg.get("DATA", {}).get("res", [346, 260])
    split_dir = data_root / args.mode
    files = sorted(split_dir.glob("*.npz"))
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    totals = {"tp": 0, "fp": 0, "fn": 0, "gt_positive": 0, "pred_positive": 0, "total": 0}
    for index, npz_path in enumerate(files, 1):
        pred, label = predict_sample(npz_path, res, args.threshold, args.min_area, np)
        pred_bool = pred.astype(bool)
        label_bool = label.astype(bool)
        tp = int(np.logical_and(pred_bool, label_bool).sum())
        fp = int(np.logical_and(pred_bool, ~label_bool).sum())
        fn = int(np.logical_and(~pred_bool, label_bool).sum())
        gt_positive = int(label_bool.sum())
        pred_positive = int(pred_bool.sum())
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["gt_positive"] += gt_positive
        totals["pred_positive"] += pred_positive
        totals["total"] += len(label_bool)

        sample_metrics = metrics_for_counts(tp, fp, fn, gt_positive, pred_positive, len(label_bool))
        print(
            f"[{index}/{len(files)}] {npz_path.name}: "
            f"IoU={sample_metrics['iou']:.6f}, "
            f"seg_acc={sample_metrics['seg_acc']:.6f}, "
            f"precision={sample_metrics['precision']:.6f}, "
            f"pred_pos={sample_metrics['pred_positive']}, "
            f"gt_pos={sample_metrics['gt_positive']}"
        )

    overall = metrics_for_counts(
        totals["tp"],
        totals["fp"],
        totals["fn"],
        totals["gt_positive"],
        totals["pred_positive"],
        totals["total"],
    )
    print("\nOverall")
    print(f"samples: {len(files)}")
    print(f"IoU: {overall['iou']:.6f}")
    print(f"seg_acc/recall: {overall['seg_acc']:.6f}")
    print(f"precision: {overall['precision']:.6f}")
    print(f"pred_positive: {overall['pred_positive']}")
    print(f"gt_positive: {overall['gt_positive']}")
    print(f"total_events: {overall['total_events']}")


if __name__ == "__main__":
    main()
