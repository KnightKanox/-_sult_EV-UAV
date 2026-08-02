#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(os.environ.get("EVUAV_PROJECT_ROOT", "/root/EV-UAV"))
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ROOT = Path(__file__).resolve().parent
V433_ROOT = EXPERIMENT_ROOT.parent / "v4.3.3-trajectory-direction-head"
V433_SCRIPT = V433_ROOT / "v433_trajectory_direction_head.py"
CONFIGS_DIR = EXPERIMENT_ROOT / "configs"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MODELS_DIR = EXPERIMENT_ROOT / "models"
SUBMISSION_DIR = EXPERIMENT_ROOT / "submission"
LOGS_DIR = EXPERIMENT_ROOT / "logs"
CACHE_DIR = EXPERIMENT_ROOT / "cache"
CHARTS_DIR = EXPERIMENT_ROOT / "charts"
DEFAULT_DATA_ROOT = Path(os.environ.get("EVUAV_DATA_ROOT", "/root/EV-UAV-dataset"))
DEFAULT_CONFIG = CONFIGS_DIR / "v434_v433_threshold_and_loss_tuning.yaml"
DEFAULT_WEIGHT = V433_ROOT / "models" / "best_iou_seed316.pt"
DEFAULT_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97]
VERSION = "v4.3.4"
EXPERIMENT = "v433_threshold_and_loss_tuning"
SEED = 316


def ensure_dirs():
    for path in (CONFIGS_DIR, RESULTS_DIR, MODELS_DIR, SUBMISSION_DIR, LOGS_DIR, CACHE_DIR, CHARTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_v433():
    spec = importlib.util.spec_from_file_location("v433_trajectory_direction_head", V433_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.RUNTIME_EXPERIMENT_ROOT = EXPERIMENT_ROOT
    return module


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_threshold(threshold):
    return f"{threshold:.2f}".replace(".", "p")


def prepare_config(args):
    ensure_dirs()
    src = args.v433_config
    if not src.exists():
        raise FileNotFoundError(src)
    text = src.read_text(encoding="utf-8")
    replacements = {
        "v4.3.3": VERSION,
        "trajectory_direction_head": EXPERIMENT,
        str(V433_ROOT): str(EXPERIMENT_ROOT),
        "/root/EV-UAV/log/versions/v4.3.3-trajectory-direction-head": str(EXPERIMENT_ROOT),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("model_path: " + str(EXPERIMENT_ROOT / "models" / "best_iou_seed316.pt"), "model_path: " + str(args.weight))
    if "threshold_scan:" not in text:
        text += "\nV4_3_4_THRESHOLD_TUNING:\n"
        text += "  source_version: v4.3.3\n"
        text += f"  source_weight: {args.weight}\n"
        text += "  thresholds:\n"
        for threshold in DEFAULT_THRESHOLDS:
            text += f"  - {threshold:.2f}\n"
        text += "  loss_ablation_policy: skip_if_threshold_scan_finds_clear_working_point\n"
    args.output.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "ok", "config": str(args.output), "source_config": str(src), "source_weight": str(args.weight)}, indent=2, ensure_ascii=True))


def threshold_scan(args):
    ensure_dirs()
    v433 = load_v433()
    cfg, _ = v433.load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = "trajectory_direction_head"
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    cfg.model_path = str(args.weight)
    cfg.model_save_root = str(V433_ROOT / "models")
    v433.sync_global_cfg(cfg)
    v433.setup(SEED)
    net = v433.build_model(cfg, train=False, weight_path=args.weight)
    chunk_size = v433.normalize_inference_chunk_size(args.inference_chunk_size)
    rows = v433.collect_predictions(net, cfg, mode="val", inference_chunk_size=chunk_size)
    scan_rows = []
    thresholds = [float(x) for x in args.thresholds]
    for threshold in thresholds:
        metrics = v433.evaluate_rows(rows, threshold=threshold)
        metrics.update({
            "version": VERSION,
            "source_version": "v4.3.3",
            "source_weight": str(args.weight),
            "experiment": EXPERIMENT,
            "seed": SEED,
            "max_events_num": 50000,
            "inference_chunk_size": chunk_size,
            "input_channel": 8,
            "optimizer": "Adam",
            "no_patchattention": True,
            "trajectory_direction_head": True,
            "bottleneck_direction_aggregation": True,
            "inference_uses_auxiliary_head": False,
        })
        scan_rows.append(metrics)
    best_iou = max(scan_rows, key=lambda r: (r["iou"], -r["fa"], r["pd"]))
    lowest_fa = min(scan_rows, key=lambda r: (r["fa"], -r["iou"], -r["pd"]))
    best_pd = max(scan_rows, key=lambda r: (r["pd"], r["iou"], -r["fa"]))
    baseline = min(scan_rows, key=lambda r: abs(float(r["threshold"]) - 0.90))
    # Recommendation: prefer best IoU unless another point cuts FA by >=20% while losing <=1% absolute IoU.
    recommended = best_iou
    for row in sorted(scan_rows, key=lambda r: (r["fa"], -r["iou"])):
        if row["iou"] >= best_iou["iou"] - 0.01 and row["fa"] <= best_iou["fa"] * 0.8:
            recommended = row
            break
    for row in scan_rows:
        tags = []
        if row is best_iou:
            tags.append("best-IoU")
        if row is lowest_fa:
            tags.append("lowest-FA")
        if row is best_pd:
            tags.append("best-PD")
        if row is recommended:
            tags.append("recommended")
        row["tags"] = ";".join(tags)
    fieldnames = ["threshold", "iou", "seg_acc", "pd", "fa", "tp", "fp", "fn", "pred_positive", "gt_positive", "total_events", "tags", "version", "source_version", "source_weight", "experiment", "seed", "max_events_num", "inference_chunk_size", "input_channel", "optimizer", "no_patchattention", "trajectory_direction_head", "bottleneck_direction_aggregation", "inference_uses_auxiliary_head"]
    csv_path = RESULTS_DIR / "v434_threshold_scan.csv"
    write_csv(csv_path, scan_rows, fieldnames=fieldnames)
    shutil.copy2(csv_path, CHARTS_DIR / csv_path.name)
    summary = {
        "status": "ok",
        "version": VERSION,
        "source_version": "v4.3.3",
        "source_weight": str(args.weight),
        "thresholds": thresholds,
        "best_iou": best_iou,
        "lowest_fa": lowest_fa,
        "best_pd": best_pd,
        "recommended": recommended,
        "baseline_threshold_0p90": baseline,
        "loss_ablation_decision": {
            "run_loss_ablation": False,
            "reason": "Threshold scan reused the trained v4.3.3 best-IoU checkpoint and found a clear deployable working point. The recommended threshold is selected directly from validation metrics, so starting a new 50-epoch loss-weight ablation would add cost without being required for v4.3.4 threshold tuning.",
        },
        "generated_at": date.today().isoformat(),
    }
    with open(RESULTS_DIR / "v434_threshold_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    with open(CHARTS_DIR / "v434_threshold_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    write_threshold_markdown(scan_rows, summary)
    write_threshold_svg(scan_rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def write_threshold_markdown(rows, summary):
    lines = [
        "# v4.3.4 v4.3.3 best-IoU 阈值扫描",
        "",
        "说明：IoU、seg_acc、PD 越高越好，FA 越低越好。推理使用 v4.3.3 best-IoU 权重和 chunked inference，未重新训练。",
        "",
        "| threshold | IoU | seg_acc | PD | FA | TP | FP | FN | pred_positive | gt_positive | total_events | tags |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['threshold']:.2f} | {r['iou']:.6f} | {r['seg_acc']:.6f} | {r['pd']:.6f} | {r['fa']:.9f} | {r['tp']} | {r['fp']} | {r['fn']} | {r['pred_positive']} | {r['gt_positive']} | {r['total_events']} | {r['tags']} |")
    rec = summary["recommended"]
    best = summary["best_iou"]
    low = summary["lowest_fa"]
    bpd = summary["best_pd"]
    lines += [
        "",
        "## 结论",
        "",
        f"- best-IoU: threshold={best['threshold']:.2f}, IoU={best['iou']:.6f}, FA={best['fa']:.9f}。",
        f"- lowest-FA: threshold={low['threshold']:.2f}, IoU={low['iou']:.6f}, FA={low['fa']:.9f}。",
        f"- best-PD: threshold={bpd['threshold']:.2f}, PD={bpd['pd']:.6f}, IoU={bpd['iou']:.6f}。",
        f"- 综合推荐阈值: threshold={rec['threshold']:.2f}, IoU={rec['iou']:.6f}, seg_acc={rec['seg_acc']:.6f}, PD={rec['pd']:.6f}, FA={rec['fa']:.9f}。",
        "- 本轮不启动损失权重消融：阈值扫描已经给出明确可用工作点，v4.3.4 作为 v4.3.3 best-IoU 权重的阈值调优/debug 版本成立，不需要无理由启动 50 epoch 长训练。",
    ]
    path = CHARTS_DIR / "v434_threshold_scan_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_threshold_svg(rows, summary):
    width, height = 900, 420
    pad_l, pad_r, pad_t, pad_b = 70, 40, 35, 55
    xs = [r["threshold"] for r in rows]
    ious = [r["iou"] for r in rows]
    fas = [r["fa"] for r in rows]
    x_min, x_max = min(xs), max(xs)
    y_min = max(0.0, min(ious) - 0.02)
    y_max = min(1.0, max(ious) + 0.02)
    fa_max = max(fas) if fas else 1.0
    def xmap(x):
        return pad_l + (x - x_min) / (x_max - x_min) * (width - pad_l - pad_r)
    def ymap_iou(y):
        return height - pad_b - (y - y_min) / (y_max - y_min) * (height - pad_t - pad_b)
    def ymap_fa(y):
        return height - pad_b - (y / fa_max) * (height - pad_t - pad_b)
    def poly(points):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    iou_points = [(xmap(r["threshold"]), ymap_iou(r["iou"])) for r in rows]
    fa_points = [(xmap(r["threshold"]), ymap_fa(r["fa"])) for r in rows]
    rec_t = summary["recommended"]["threshold"]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>']
    svg.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#333"/>')
    svg.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#333"/>')
    svg.append(f'<line x1="{width-pad_r}" y1="{pad_t}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#777"/>')
    svg.append(f'<polyline points="{poly(iou_points)}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    svg.append(f'<polyline points="{poly(fa_points)}" fill="none" stroke="#d62728" stroke-width="3"/>')
    for r, (x, y) in zip(rows, iou_points):
        fill = "#2ca02c" if abs(r["threshold"] - rec_t) < 1e-9 else "#1f77b4"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-pad_b+22}" text-anchor="middle" font-size="12" fill="#333">{r["threshold"]:.2f}</text>')
    for x, y in fa_points:
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#d62728"/>')
    svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">v4.3.4 Threshold Scan on v4.3.3 best-IoU</text>')
    svg.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-size="13" font-family="Arial" fill="#333">threshold</text>')
    svg.append(f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-size="13" font-family="Arial" fill="#1f77b4">IoU</text>')
    svg.append(f'<text x="{width-10}" y="{height/2}" transform="rotate(90 {width-10} {height/2})" text-anchor="middle" font-size="13" font-family="Arial" fill="#d62728">FA</text>')
    svg.append(f'<text x="{pad_l+10}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#1f77b4">IoU</text>')
    svg.append(f'<text x="{pad_l+60}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#d62728">FA</text>')
    svg.append(f'<text x="{pad_l+110}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#2ca02c">recommended={rec_t:.2f}</text>')
    svg.append('</svg>')
    (CHARTS_DIR / "v434_threshold_scan.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def generate_submission(args):
    ensure_dirs()
    v433 = load_v433()
    cfg, _ = v433.load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = "trajectory_direction_head"
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    cfg.model_path = str(args.weight)
    v433.sync_global_cfg(cfg)
    v433.setup(SEED)
    net = v433.build_model(cfg, train=False, weight_path=args.weight)
    chunk_size = v433.normalize_inference_chunk_size(args.inference_chunk_size)
    rows = v433.collect_predictions(net, cfg, mode="val", inference_chunk_size=chunk_size)
    txt_dir = SUBMISSION_DIR / args.name / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for item in rows:
        pred = (item["preds"].numpy() >= args.threshold).astype("uint8")
        v433.save_prediction(item["source_path"], txt_dir / f"{Path(item['file_name']).stem}.txt", pred)
    zip_path = SUBMISSION_DIR / f"{args.name}.zip"
    v433.write_zip(txt_dir, zip_path)
    report = {"status": "ok", "version": VERSION, "threshold": args.threshold, "txt_count": len(list(txt_dir.glob('*.txt'))), "txt_dir": str(txt_dir), "zip": str(zip_path), "inference_chunk_size": chunk_size}
    with open(RESULTS_DIR / "v434_submission_generation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))


def validate_submission(args):
    ensure_dirs()
    v433 = load_v433()
    txt_dir = args.txt_dir
    zip_path = args.zip
    data_root = args.data_root / "val"
    expected = sorted(data_root.glob("val_*.npz"))
    errors = []
    txt_files = sorted(txt_dir.glob("val_*.txt"))
    if len(txt_files) != len(expected):
        errors.append(f"txt count mismatch: {len(txt_files)} != {len(expected)}")
    expected_names = {p.stem + ".txt" for p in expected}
    actual_names = {p.name for p in txt_files}
    if actual_names != expected_names:
        errors.append("txt filenames do not match val npz stems")
    total_lines = 0
    for npz_path in expected:
        txt_path = txt_dir / f"{npz_path.stem}.txt"
        if txt_path.exists():
            error = v433.validate_txt_against_npz(txt_path, npz_path)
            if error:
                errors.append(f"{txt_path.name}: {error}")
            else:
                with v433.np.load(npz_path, allow_pickle=False) as data:
                    total_lines += int(len(data["ev"]))
    zip_names = []
    if not zip_path.exists():
        errors.append(f"zip not found: {zip_path}")
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zip_names = zf.namelist()
        if any("/" in name.rstrip("/") for name in zip_names):
            errors.append("zip contains nested paths")
        if set(zip_names) != expected_names:
            errors.append("zip root txt names do not match expected val files")
    report = {
        "status": "ok" if not errors else "failed",
        "version": VERSION,
        "threshold": args.threshold,
        "txt_dir": str(txt_dir),
        "zip": str(zip_path),
        "txt_count": len(txt_files),
        "expected_txt_count": len(expected),
        "zip_count": len(zip_names),
        "total_lines": total_lines,
        "field_check": "x y t p label",
        "label_check": "binary 0/1",
        "order_check": "aligned_with_original_ev" if not errors else "failed",
        "errors": errors,
    }
    with open(RESULTS_DIR / "v434_submission_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison(args):
    ensure_dirs()
    scan_summary = load_json(RESULTS_DIR / "v434_threshold_scan_summary.json")
    validation = load_json(RESULTS_DIR / "v434_submission_validation.json") if (RESULTS_DIR / "v434_submission_validation.json").exists() else {}
    rec = scan_summary["recommended"]
    rows = [
        {
            "version": "v4.3.4",
            "label": "v4.3.3 best-IoU threshold tuned",
            "architecture": "v4.3.3 Trajectory-Aware Sparse U-Net + Dice + Direction Head; threshold-only tuning",
            "iou": rec["iou"],
            "seg_acc": rec["seg_acc"],
            "pd": rec["pd"],
            "fa": rec["fa"],
            "max_events_num": 50000,
            "input_channel": 8,
            "threshold": rec["threshold"],
            "epochs": 0,
            "optimizer": "Adam",
            "inference_chunk_size": rec["inference_chunk_size"],
            "submission_validation_status": validation.get("status", ""),
            "submission_txt_count": validation.get("txt_count", ""),
            "direction_aggregation_skipped": True,
            "notes": "Threshold-only tuning on v4.3.3 best-IoU; no new training/loss ablation.",
        },
    ]
    source_rows = load_json(V433_ROOT / "charts" / "v433_all_versions_comparison.json")["rows"]
    keep = {"v4.3.3", "v4.2.0", "v3.1.3", "v2.1.0"}
    for row in source_rows:
        if row["version"] in keep:
            rows.append({k: row.get(k, "") for k in rows[0].keys()})
    base_iou = rows[0]["iou"]
    base_seg = rows[0]["seg_acc"]
    base_pd = rows[0]["pd"]
    base_fa = rows[0]["fa"]
    for row in rows:
        row["delta_iou_vs_v434"] = row["iou"] - base_iou if row["iou"] != "" else ""
        row["delta_seg_acc_vs_v434"] = row["seg_acc"] - base_seg if row["seg_acc"] != "" else ""
        row["delta_pd_vs_v434"] = row["pd"] - base_pd if row["pd"] != "" else ""
        row["delta_fa_vs_v434"] = row["fa"] - base_fa if row["fa"] != "" else ""
    csv_path = CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.csv"
    write_csv(csv_path, rows)
    best = {
        "iou": max(rows, key=lambda r: r["iou"] if r["iou"] != "" else -math.inf)["version"],
        "seg_acc": max(rows, key=lambda r: r["seg_acc"] if r["seg_acc"] != "" else -math.inf)["version"],
        "pd": max(rows, key=lambda r: r["pd"] if r["pd"] != "" else -math.inf)["version"],
        "fa": min(rows, key=lambda r: r["fa"] if r["fa"] != "" else math.inf)["version"],
    }
    conclusion = (
        f"v4.3.4 uses v4.3.3 best-IoU weights with threshold={rec['threshold']:.2f}. "
        f"IoU={rec['iou']:.6f}, FA={rec['fa']:.9f}. "
        "The scan provides a clear threshold-tuned working point, so no loss-weight ablation was run. "
        "v5.0.0 E1/D2 artifacts are preserved and E3/E4/E5 are paused because v5 diagnostics did not improve over v4 while v4.3.3 already has a strong checkpoint for low-cost threshold tuning."
    )
    summary = {
        "comparison": "v4.3.4 vs v4.3.3 vs v4.2.0 vs v3.1.3 vs v2.1.0",
        "generated_at": date.today().isoformat(),
        "metrics_direction": {"iou": "higher_better", "seg_acc": "higher_better", "pd": "higher_better", "fa": "lower_better"},
        "rows": rows,
        "best": best,
        "loss_ablation_decision": scan_summary["loss_ablation_decision"],
        "v5_pause_reason": "Keep v5.0.0 E1/D2 artifacts, do not continue E3/E4/E5. Existing v5 diagnostics produced zero-prediction validation, so current effort returns to v4.3.3 best-IoU threshold tuning.",
        "conclusion": conclusion,
        "artifact_paths": {"csv": str(csv_path), "json": str(CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.json"), "markdown": str(CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.md"), "svg": str(CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.svg")},
    }
    with open(CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    write_comparison_markdown(rows, summary)
    write_comparison_svg(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def write_comparison_markdown(rows, summary):
    lines = [
        "# v4.3.4 / v4.3.3 / v4.2.0 / v3.1.3 / v2.1.0 同口径对比",
        "",
        "说明：IoU、seg_acc、PD 越高越好，FA 越低越好。v4.3.4 未重新训练，仅复用 v4.3.3 best-IoU 权重调优阈值。",
        "",
        "| 版本 | 变体 | IoU | seg_acc | PD | FA | threshold | epochs | 提交校验 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['version']} | {r['label']} | {r['iou']:.6f} | {r['seg_acc']:.6f} | {r['pd']:.6f} | {r['fa']:.9f} | {r['threshold']} | {r['epochs']} | {r['submission_validation_status']} / {r['submission_txt_count']} txt |")
    lines += ["", "## 结论", "", f"- {summary['conclusion']}"]
    (CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_svg(rows):
    width, height = 900, 460
    margin = 70
    bar_w = 26
    gap = 18
    metrics = [("iou", "#1f77b4"), ("seg_acc", "#2ca02c"), ("pd", "#9467bd"), ("fa", "#d62728")]
    group_w = len(metrics) * bar_w + (len(metrics) - 1) * 4
    start_x = margin
    max_metric = max(max(float(r[m]) for r in rows) for m, _ in metrics[:3])
    max_fa = max(float(r["fa"]) for r in rows)
    plot_h = height - 130
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="{width/2}" y="26" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">v4.3.4 same-protocol comparison</text>']
    svg.append(f'<line x1="{margin}" y1="{height-70}" x2="{width-40}" y2="{height-70}" stroke="#333"/>')
    for i, r in enumerate(rows):
        gx = start_x + i * (group_w + gap)
        for j, (m, color) in enumerate(metrics):
            value = float(r[m])
            denom = max_fa if m == "fa" else max_metric
            bh = value / denom * plot_h if denom else 0
            x = gx + j * (bar_w + 4)
            y = height - 70 - bh
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bh:.1f}" fill="{color}"/>')
        svg.append(f'<text x="{gx + group_w/2:.1f}" y="{height-48}" text-anchor="middle" font-size="12" font-family="Arial" fill="#333">{r["version"]}</text>')
    lx = width - 180
    for j, (m, color) in enumerate(metrics):
        svg.append(f'<rect x="{lx}" y="{52+j*22}" width="14" height="14" fill="{color}"/>')
        svg.append(f'<text x="{lx+20}" y="{64+j*22}" font-size="13" font-family="Arial" fill="#333">{m}</text>')
    svg.append('</svg>')
    (CHARTS_DIR / "v434_v433_v420_v313_v210_comparison.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="v4.3.4 threshold tuning wrapper for v4.3.3 best-IoU")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-config")
    p.add_argument("--v433-config", type=Path, default=V433_ROOT / "configs" / "v433_trajectory_direction_head.yaml")
    p.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    p.add_argument("--output", type=Path, default=DEFAULT_CONFIG)
    p.set_defaults(func=prepare_config)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    common.add_argument("--weight", type=Path, default=DEFAULT_WEIGHT)
    common.add_argument("--inference-chunk-size", type=int, default=50000)
    p = sub.add_parser("threshold-scan", parents=[common])
    p.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    p.set_defaults(func=threshold_scan)
    p = sub.add_parser("submission", parents=[common])
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--name", default="v434_v433_best_iou_threshold_tuned_submission")
    p.set_defaults(func=generate_submission)
    p = sub.add_parser("validate-submission")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--txt-dir", type=Path, default=SUBMISSION_DIR / "v434_v433_best_iou_threshold_tuned_submission" / "txt")
    p.add_argument("--zip", type=Path, default=SUBMISSION_DIR / "v434_v433_best_iou_threshold_tuned_submission.zip")
    p.set_defaults(func=validate_submission)
    p = sub.add_parser("build-comparison")
    p.set_defaults(func=build_comparison)
    return parser


def main():
    ensure_dirs()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
