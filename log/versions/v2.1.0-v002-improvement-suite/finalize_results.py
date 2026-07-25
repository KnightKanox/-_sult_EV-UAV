#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(os.environ.get("EVUAV_PROJECT_ROOT", "/root/EV-UAV"))
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
CHARTS_DIR = EXPERIMENT_ROOT / "charts"
SUBMISSION_DIR = EXPERIMENT_ROOT / "submission"
SCRIPT_PATH = EXPERIMENT_ROOT / "v002_improvement_suite.py"
BASELINE_V002 = {
    "variant": "v0.0.2_leaky_relu",
    "iou": 0.351111,
    "seg_acc": 0.710027,
    "pd": 0.796304,
    "fa": 0.001275,
}


def read_csv(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_plan():
    path = RESULTS_DIR / "planned_experiments.csv"
    if not path.exists():
        return []
    return read_csv(path)


def last_metric_row(variant):
    path = RESULTS_DIR / f"{variant}_epoch_summary.csv"
    if not path.exists():
        return None
    rows = read_csv(path)
    scored = [row for row in rows if row.get("val_iou") not in ("", None)]
    if not scored:
        return None
    return max(scored, key=lambda row: float(row["val_iou"]))


def collect_training_results(require_model=True):
    rows = []
    for item in load_plan():
        variant = item["variant"]
        best = last_metric_row(variant)
        summary_path = RESULTS_DIR / f"{variant}_summary.json"
        model_path = EXPERIMENT_ROOT / "models" / variant / "best_iou_seed37.pt"
        if best is None:
            continue
        if require_model and not model_path.exists():
            continue
        best_iou = best["val_iou"]
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            best_iou = summary.get("best_iou", best_iou)
        row = {
            "variant": variant,
            "negative_slope": item.get("negative_slope", ""),
            "pos_weight": item.get("pos_weight", ""),
            "dice_weight": item.get("dice_weight", ""),
            "k": item.get("k", ""),
            "t": item.get("t", ""),
            "best_epoch": best["epoch"],
            "iou": best_iou,
            "seg_acc": best["val_seg_acc"],
            "pd": best["val_pd"],
            "fa": best["val_fa"],
            "model_path": str(model_path),
            "summary_path": str(summary_path) if summary_path.exists() else "",
        }
        rows.append(row)
    return rows


def plot_training_curves(results):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for result in results:
        path = RESULTS_DIR / f"{result['variant']}_epoch_summary.csv"
        rows = read_csv(path)
        xs = [int(row["epoch"]) for row in rows]
        ys = [float(row["mean_loss"]) for row in rows]
        plt.plot(xs, ys, label=result["variant"], linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Mean loss")
    plt.title("v2.1.0 Training Loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "training_loss_curves.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for result in results:
        path = RESULTS_DIR / f"{result['variant']}_epoch_summary.csv"
        rows = [row for row in read_csv(path) if row.get("val_iou") not in ("", None)]
        if not rows:
            continue
        xs = [int(row["epoch"]) for row in rows]
        ys = [float(row["val_iou"]) for row in rows]
        plt.plot(xs, ys, marker="o", markersize=3, label=result["variant"])
    plt.axhline(BASELINE_V002["iou"], color="#111827", linestyle="--", linewidth=1.2, label="v0.0.2 IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Validation IoU")
    plt.title("Validation IoU vs v0.0.2")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "val_iou_vs_v002.png", dpi=180)
    plt.close()


def plot_metric_comparison(results):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["iou", "seg_acc", "pd", "fa"]
    labels = [BASELINE_V002["variant"]] + [row["variant"] for row in results]
    x = list(range(len(labels)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.reshape(-1)
    for ax, metric in zip(axes, metrics):
        values = [BASELINE_V002[metric]] + [float(row[metric]) for row in results]
        colors = ["#6b7280"] + ["#2563eb"] * len(results)
        ax.bar(x, values, color=colors)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("v2.1.0 Validation Metrics vs v0.0.2")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "metrics_vs_v002.png", dpi=180)
    plt.close(fig)

    delta_rows = []
    for row in results:
        delta = {"variant": row["variant"]}
        for metric in metrics:
            delta[f"delta_{metric}"] = float(row[metric]) - BASELINE_V002[metric]
        delta_rows.append(delta)
    write_csv(RESULTS_DIR / "metrics_delta_vs_v002.csv", delta_rows)


def choose_best(results):
    if not results:
        raise RuntimeError("No completed training results found")
    return max(results, key=lambda row: float(row["iou"]))


def run_submission(best, threshold):
    import subprocess

    name = f"{best['variant']}_best_submission"
    zip_path = SUBMISSION_DIR / f"{name}.zip"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "submission",
        "--weight",
        best["model_path"],
        "--threshold",
        str(threshold),
        "--negative-slope",
        str(best.get("negative_slope") or 0.01),
        "--name",
        name,
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0 and not zip_path.exists():
        completed.check_returncode()
    return zip_path


def validate_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    txt_names = [name for name in names if name.endswith(".txt")]
    return {
        "zip_path": str(zip_path),
        "file_count": len(txt_names),
        "has_nested_dir": any("/" in name.strip("/") for name in names),
        "valid_file_count": len(txt_names) == 24,
    }


def main():
    parser = argparse.ArgumentParser(description="Finalize v2.1.0 training outputs")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--skip-submission", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    results = collect_training_results()
    write_csv(RESULTS_DIR / "training_results_summary.csv", results)
    plot_training_curves(results)
    plot_metric_comparison(results)
    best = choose_best(results)
    final = {"best_variant": best, "baseline_v002": BASELINE_V002}
    if not args.skip_submission:
        zip_path = run_submission(best, args.threshold)
        final["submission"] = validate_zip(zip_path)
    with (RESULTS_DIR / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=True)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
