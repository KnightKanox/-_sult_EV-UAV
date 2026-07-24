#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "log" / "versions" / "v1.1.0-traditional-baseline-compare" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CURRENT = {
    "model": "Traditional\nBaseline",
    "iou": 0.598629,
    "seg_acc": 0.835343,
    "precision": 0.678716,
    "pred_positive": 2960,
    "gt_positive": 2405,
    "total_events": 44751,
}

V002 = {
    "model": "v0.0.2\nLeakyReLU",
    "iou": 0.35111120343208313,
    "seg_acc": 0.7100265622138977,
    "precision": None,
    "pd": 0.7963040739185216,
    "fa": 0.0012747462506434986,
}

BASELINE = {
    "model": "v0.0.1\nBaseline",
    "iou": 0.32399630546569824,
    "seg_acc": 0.7172472476959229,
    "pd": 0.8135237295254095,
    "fa": 0.0027119136579062384,
}


def bar_chart(metrics, filename, title):
    labels = [m["model"] for m in metrics]
    keys = [k for k in metrics[0].keys() if k != "model"]
    x = np.arange(len(keys))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, metric in enumerate(metrics):
        values = [metric.get(k, np.nan) for k in keys]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, values, width=width, label=labels[idx])

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, max(v for metric in metrics for v in metric.values() if isinstance(v, (int, float))) * 1.15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=200)
    plt.close(fig)


def compare_current_v002():
    metrics = [
        {"model": CURRENT["model"], "iou": CURRENT["iou"], "seg_acc": CURRENT["seg_acc"]},
        {"model": V002["model"], "iou": V002["iou"], "seg_acc": V002["seg_acc"]},
    ]
    bar_chart(metrics, "iou_segacc_compare.png", "IoU / Seg Acc 对比")


def compare_full_metrics():
    metrics = [
        {"model": BASELINE["model"], "iou": BASELINE["iou"], "seg_acc": BASELINE["seg_acc"], "pd": BASELINE["pd"], "fa": BASELINE["fa"]},
        {"model": V002["model"], "iou": V002["iou"], "seg_acc": V002["seg_acc"], "pd": V002["pd"], "fa": V002["fa"]},
    ]
    bar_chart(metrics, "full_metrics_compare.png", "v0.0.1 Baseline vs v0.0.2 LeakyReLU")


def print_summary():
    summary = {
        "current": CURRENT,
        "v0.0.2": V002,
        "delta_vs_v0.0.2": {
            "iou": CURRENT["iou"] - V002["iou"],
            "seg_acc": CURRENT["seg_acc"] - V002["seg_acc"],
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    compare_current_v002()
    compare_full_metrics()
    print_summary()
    print(f"charts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
