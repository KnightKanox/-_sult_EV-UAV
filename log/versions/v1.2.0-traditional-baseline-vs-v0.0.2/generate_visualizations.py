#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = ROOT.parent
OUT_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import traditional_baseline as tb

RES = [346, 260]
THRESHOLD = 2.0
MIN_AREA = 3
SAMPLE = DATA_ROOT / "train" / "train_000.npz"


def sample_arrays(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        evs_norm = data["evs_norm"]
        labels = (evs_norm[:, 4] > 0).astype(np.uint8)
        x, y = tb.event_xy(data, RES, np)

    height, width = RES[1], RES[0]
    density = np.zeros((height, width), dtype=np.float32)
    gt_density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (y, x), 1.0)
    np.add.at(gt_density, (y[labels.astype(bool)], x[labels.astype(bool)]), 1.0)

    positive_density = density[density > 0]
    cutoff = max(1.0, float(positive_density.mean()) * THRESHOLD) if positive_density.size else 1.0
    raw_mask = (density >= cutoff).astype(np.uint8)
    num_labels, cc_labels, stats, _ = tb.connected_components_with_stats(raw_mask, np)
    kept = np.zeros_like(raw_mask, dtype=np.uint8)
    for label_id in range(1, num_labels):
        if stats[label_id, 4] >= MIN_AREA:
            kept[cc_labels == label_id] = 1

    pred = kept[y, x].astype(np.uint8)
    pred_density = np.zeros((height, width), dtype=np.float32)
    np.add.at(pred_density, (y[pred.astype(bool)], x[pred.astype(bool)]), 1.0)

    pred_bool = pred.astype(bool)
    label_bool = labels.astype(bool)
    tp = int(np.logical_and(pred_bool, label_bool).sum())
    fp = int(np.logical_and(pred_bool, ~label_bool).sum())
    fn = int(np.logical_and(~pred_bool, label_bool).sum())
    metrics = tb.metrics_for_counts(tp, fp, fn, int(label_bool.sum()), int(pred_bool.sum()), len(label_bool))
    metrics.update({"tp": tp, "fp": fp, "fn": fn, "cutoff": cutoff, "components_raw": int(num_labels - 1), "components_kept": int(kept.max() > 0 and len(np.unique(cc_labels[kept.astype(bool)])))})
    return density, gt_density, pred_density, raw_mask, kept, cc_labels, metrics


def save_metrics_csv(metrics):
    rows = [
        {"item": "traditional_baseline_train_000", "metric": "IoU", "value": metrics["iou"], "note": "event-level, train_000.npz"},
        {"item": "traditional_baseline_train_000", "metric": "seg_acc/recall", "value": metrics["seg_acc"], "note": "event-level, train_000.npz"},
        {"item": "traditional_baseline_train_000", "metric": "precision", "value": metrics["precision"], "note": "event-level, train_000.npz"},
        {"item": "v0.0.2_leaky_relu_history", "metric": "epoch_mean_loss_start", "value": 0.234, "note": "from 修改日志.md"},
        {"item": "v0.0.2_leaky_relu_history", "metric": "epoch_mean_loss_end", "value": 0.023, "note": "from 修改日志.md"},
        {"item": "v0.0.2_leaky_relu_history", "metric": "batch_loss_records", "value": 9950, "note": "from 修改日志.md"},
        {"item": "v0.0.2_val_history", "metric": "IoU", "value": 0.351, "note": "from v0.0.3 log, val 24 samples"},
        {"item": "v0.0.2_val_history", "metric": "seg_acc", "value": 0.7308, "note": "from v0.0.1/v0.0.3 deltas"},
        {"item": "v0.0.2_val_history", "metric": "PD", "value": 0.8418, "note": "from v0.0.1/v0.0.3 deltas"},
        {"item": "v0.0.2_val_history", "metric": "FA", "value": 0.0013, "note": "from 修改日志.md"},
    ]
    with (OUT_DIR / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


def plot_baseline_metrics(metrics):
    names = ["IoU", "seg_acc/recall", "precision"]
    values = [metrics["iou"], metrics["seg_acc"], metrics["precision"]]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    bars = ax.bar(names, values, color=["#386cb0", "#7fc97f", "#fdb462"], width=0.6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Traditional baseline event-level metrics on train_000.npz")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "traditional_baseline_metrics_bar.png")
    plt.close(fig)


def plot_event_visualization(density, gt_density, pred_density, raw_mask, kept, cc_labels, metrics):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=160)
    panels = [
        (np.log1p(density), "All event density", "magma"),
        (gt_density > 0, "GT positive mask", "Greens"),
        (pred_density > 0, "Predicted positive mask", "Blues"),
        (raw_mask, f"Raw density mask cutoff={metrics['cutoff']:.2f}", "gray"),
        (kept, f"Kept components min_area={MIN_AREA}", "gray"),
        (cc_labels * kept, "Connected components kept", "tab20"),
    ]
    for ax, (arr, title, cmap) in zip(axes.flat, panels):
        ax.imshow(arr, cmap=cmap, origin="upper", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Traditional baseline on train_000.npz: IoU={metrics['iou']:.3f}, recall={metrics['seg_acc']:.3f}, precision={metrics['precision']:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "traditional_baseline_event_visualization.png")
    plt.close(fig)


def plot_v002_summary(metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)
    ax = axes[0]
    metric_names = ["IoU", "seg_acc", "PD/recall", "FA"]
    traditional = [metrics["iou"], metrics["seg_acc"], metrics["seg_acc"], np.nan]
    v002 = [0.351, 0.7308, 0.8418, 0.0013]
    x = np.arange(len(metric_names))
    width = 0.36
    ax.bar(x - width / 2, traditional, width, label="Traditional baseline\ntrain_000 event-level", color="#386cb0")
    ax.bar(x + width / 2, v002, width, label="v0.0.2 history\nval/log metrics", color="#f0027f")
