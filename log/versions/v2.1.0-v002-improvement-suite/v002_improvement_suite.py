#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    torch = None
    nn = None
    optim = None
try:
    import tqdm
except ImportError:
    tqdm = None
try:
    import yaml
except ImportError:
    yaml = None

ModuleBase = nn.Module if nn is not None else object

PROJECT_ROOT = Path(os.environ.get("EVUAV_PROJECT_ROOT", "/root/EV-UAV"))
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ROOT = Path(__file__).resolve().parent
RUNTIME_EXPERIMENT_ROOT = Path(os.environ.get("EVUAV_RUNTIME_EXPERIMENT_ROOT", str(EXPERIMENT_ROOT)))
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MODELS_DIR = EXPERIMENT_ROOT / "models"
SUBMISSION_DIR = EXPERIMENT_ROOT / "submission"
CONFIGS_DIR = EXPERIMENT_ROOT / "configs"
RUNTIME_MODELS_DIR = RUNTIME_EXPERIMENT_ROOT / "models"
RUNTIME_SCRIPT = RUNTIME_EXPERIMENT_ROOT / "v002_improvement_suite.py"
SEED = 37

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
DEFAULT_DATA_ROOT = Path(os.environ.get("EVUAV_DATA_ROOT", "/root/EV-UAV-dataset"))
DEFAULT_V002_WEIGHT = PROJECT_ROOT / "log" / "versions" / "v0.0.2-leaky-relu-training" / "models" / "best_iou_seed37.pt"
if not DEFAULT_V002_WEIGHT.exists():
    DEFAULT_V002_WEIGHT = PROJECT_ROOT / "log" / "model_leaky_relu" / "best_iou_seed37.pt"
DEFAULT_D_POOL_ROOT = PROJECT_ROOT / "log" / "versions" / "v0.1.0-sample-pool-ratio-experiments" / "D"
if not DEFAULT_D_POOL_ROOT.exists():
    DEFAULT_D_POOL_ROOT = PROJECT_ROOT / "log" / "versions" / "0.1.0-sample-pool-ratio-experiments" / "D"


def parse_thresholds(value):
    if ":" in value:
        start, stop, step = [float(part) for part in value.split(":")]
        count = int(round((stop - start) / step)) + 1
        return [round(start + i * step, 6) for i in range(count)]
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return [float(part) for part in parts]


def default_cfg_namespace():
    return SimpleNamespace(
        model_name="evissgnet",
        width=12,
        block_residual=True,
        block_reps=2,
        use_coords=True,
        input_channel=4,
        root=str(DEFAULT_DATA_ROOT),
        whole_t=8000,
        res=[346, 260],
        epochs=50,
        batch_size=1,
        train_workers=8,
        optim="Adam",
        lr=0.001,
        k=3,
        t=5,
        max_events_num=20000,
        model_save_root=str(PROJECT_ROOT / "log" / "model"),
        vis=False,
        eval=True,
        roc=True,
        pd_detT=50,
        correct_thresh=0.0001,
        save=True,
        model_path=str(PROJECT_ROOT / "log" / "model" / "best_iou_seed37.pt"),
    )


def load_yaml_config(path):
    if yaml is None:
        return default_cfg_namespace(), None
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat), raw


def write_yaml_config(path, cfg):
    data = {
        "STRUCTURE": {
            "model_name": "evissgnet",
            "width": int(cfg.width),
            "block_residual": True,
            "block_reps": 2,
            "use_coords": True,
        },
        "DATA": {
            "input_channel": int(cfg.input_channel),
            "root": str(cfg.root),
            "whole_t": int(cfg.whole_t),
            "res": list(cfg.res),
        },
        "TRAIN": {
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "train_workers": int(getattr(cfg, "train_workers", 8)),
            "optim": getattr(cfg, "optim", "Adam"),
            "lr": float(cfg.lr),
            "k": int(cfg.k),
            "t": int(cfg.t),
            "max_events_num": int(cfg.max_events_num),
            "model_save_root": str(cfg.model_save_root),
        },
        "TEST": {
            "vis": False,
            "eval": True,
            "roc": True,
            "pd_detT": int(cfg.pd_detT),
            "correct_thresh": float(cfg.correct_thresh),
            "save": True,
            "model_path": str(cfg.model_path),
        },
    }
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
        return
    lines = []
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if isinstance(value, list):
                rendered = "[" + ", ".join(str(item) for item in value) + "]"
            elif isinstance(value, bool):
                rendered = str(value).lower()
            else:
                rendered = str(value)
            lines.append(f"  {key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def require_runtime():
    missing = []
    if np is None:
        missing.append("numpy")
    if torch is None:
        missing.append("torch")
    if tqdm is None:
        missing.append("tqdm")
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing) + ". Run this command inside the EV-UAV Docker/conda environment.")


def setup(seed=SEED):
    require_runtime()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = str(seed)


def sync_global_cfg(cfg):
    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0], "--config", str(DEFAULT_CONFIG)]
    try:
        import configs.configs as config_module
    finally:
        sys.argv = argv_backup
    config_module.cfg.__dict__.update(vars(cfg))
    return config_module.cfg


def replace_relu(module, negative_slope):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        else:
            replace_relu(child, negative_slope)


def build_model(cfg, negative_slope=0.01, weight_path=None, train=False):
    require_runtime()
    from model.evspsegnet import evspsegnet

    net = evspsegnet(cfg)
    replace_relu(net, negative_slope)
    net = net.train() if train else net.eval()
    net.cuda()
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path, map_location="cuda:0"))
    return net


def make_dataloader(cfg, mode, shuffle=False):
    from dataset.ev_uav import EvUAV

    dataset = EvUAV(cfg, mode=mode)
    dataset.file_list = sorted(dataset.file_list)
    sampler = None
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
    )
    return dataset, dataloader


def collect_predictions(net, cfg, mode="val"):
    dataset, dataloader = make_dataloader(cfg, mode=mode, shuffle=False)
    rows = []
    with torch.no_grad():
        for batch_index, ev in enumerate(tqdm.tqdm(dataloader, desc=f"predict-{mode}", unit="sample")):
            p2v_map = ev["p2v_map"].long().cuda()
            ev_locs = ev["locs"]
            labels = ev["seg_label"].float().cpu()
            idx = ev["idx_label"]
            preds, _ = net(ev["voxel_ev"])
            preds = preds[p2v_map].reshape(-1).cpu()
            first_sample = batch_index * cfg.batch_size
            for local_index in ev_locs[:, 0].long().unique(sorted=True).tolist():
                mask = ev_locs[:, 0].long() == local_index
                sample_index = first_sample + local_index
                file_name = dataset.file_list[sample_index]
                rows.append(
                    {
                        "sample": sample_index,
                        "file_name": file_name,
                        "source_path": str(Path(dataset.root) / file_name),
                        "preds": preds[mask].clone(),
                        "labels": labels[mask].clone(),
                        "idx": idx[mask.numpy()] if isinstance(idx, np.ndarray) else idx[mask],
                        "locs": ev_locs[mask].clone(),
                    }
                )
    return rows


def metric_counts(pred, label):
    pred_bool = pred.astype(bool)
    label_bool = label.astype(bool)
    tp = int(np.logical_and(pred_bool, label_bool).sum())
    fp = int(np.logical_and(pred_bool, ~label_bool).sum())
    fn = int(np.logical_and(~pred_bool, label_bool).sum())
    gt_positive = int(label_bool.sum())
    pred_positive = int(pred_bool.sum())
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "iou": tp / union if union else 0.0,
        "seg_acc": tp / gt_positive if gt_positive else 0.0,
        "precision": tp / pred_positive if pred_positive else 0.0,
        "gt_positive": gt_positive,
        "pred_positive": pred_positive,
        "total_events": int(len(label_bool)),
    }


def evaluate_rows(rows, thresholds):
    from utils.eval import evalute

    out = []
    for threshold in thresholds:
        evaluator = evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001))
        totals = {"tp": 0, "fp": 0, "fn": 0, "gt_positive": 0, "pred_positive": 0, "total_events": 0}
        for item in rows:
            pred_binary = (item["preds"].numpy() >= threshold).astype(np.uint8)
            label = item["labels"].numpy().astype(np.uint8)
            counts = metric_counts(pred_binary, label)
            for key in totals:
                totals[key] += counts[key]
            sample_key = str(item["sample"])
            evaluator.matches[sample_key] = {"seg_pred": item["preds"].clone(), "seg_gt": item["labels"].clone()}
            evaluator.roc_update(
                item["locs"][:, 3],
                item["preds"].clone(),
                item["idx"],
                item["labels"].clone(),
                item["locs"].float(),
                thresh=threshold,
            )
        union = totals["tp"] + totals["fp"] + totals["fn"]
        pd_value, fa_value = evaluator.cal_roc()
        out.append(
            {
                "threshold": threshold,
                "iou": totals["tp"] / union if union else 0.0,
                "seg_acc": totals["tp"] / totals["gt_positive"] if totals["gt_positive"] else 0.0,
                "precision": totals["tp"] / totals["pred_positive"] if totals["pred_positive"] else 0.0,
                "pd": float(pd_value),
                "fa": float(fa_value),
                **totals,
            }
        )
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(source_path, output_path, prediction):
    with np.load(source_path) as data:
        source_ev = data["ev"]
        if len(source_ev) != len(prediction):
            raise ValueError(f"{source_path.name}: event count mismatch")
        predicted_ev = np.empty(
            len(source_ev),
            dtype=[
                ("x", source_ev.dtype["x"]),
                ("y", source_ev.dtype["y"]),
                ("t", source_ev.dtype["t"]),
                ("p", source_ev.dtype["p"]),
                ("label", np.int64),
            ],
        )
        for field in ("x", "y", "t", "p"):
            predicted_ev[field] = source_ev[field]
        predicted_ev["label"] = prediction.astype(np.int64)
    np.savetxt(output_path, predicted_ev, fmt=["%d", "%d", "%.9f", "%d", "%d"], delimiter=" ")


def write_zip(txt_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            zf.write(txt_path, arcname=txt_path.name)


def command_threshold_scan(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    sync_global_cfg(cfg)
    thresholds = parse_thresholds(args.thresholds)
    net = build_model(cfg, args.negative_slope, args.weight)
    rows = collect_predictions(net, cfg, mode="val")
    metrics = evaluate_rows(rows, thresholds)
    out_path = RESULTS_DIR / "v002_threshold_fine_scan.csv"
    write_csv(out_path, metrics)
    best_iou = max(metrics, key=lambda row: row["iou"])
    best_fa002 = min(metrics, key=lambda row: abs(row["fa"] - 0.002))
    summary = {
        "weight": str(args.weight),
        "negative_slope": args.negative_slope,
        "best_iou": best_iou,
        "closest_fa_0.002": best_fa002,
        "rows": len(metrics),
    }
    with open(RESULTS_DIR / "v002_threshold_fine_scan_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2))


def command_submission(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    sync_global_cfg(cfg)
    net = build_model(cfg, args.negative_slope, args.weight)
    rows = collect_predictions(net, cfg, mode="val")
    name = args.name or f"v002_threshold_{args.threshold:.3f}".replace(".", "p")
    txt_dir = SUBMISSION_DIR / name / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for item in rows:
        prediction = (item["preds"].numpy() >= args.threshold).astype(np.uint8)
        save_prediction(Path(item["source_path"]), txt_dir / f"{Path(item['file_name']).stem}.txt", prediction)
    zip_path = SUBMISSION_DIR / f"{name}.zip"
    write_zip(txt_dir, zip_path)
    print(json.dumps({"txt_count": len(list(txt_dir.glob("*.txt"))), "zip": str(zip_path)}, indent=2))


class WeightedSTCLoss(ModuleBase):
    def __init__(self, k, t, cfg, pos_weight=1.0, dice_weight=0.0, eps=1e-5):
        super().__init__()
        from utils.stcloss import STCLoss

        self.base = STCLoss(k=k, t=t, cfg=cfg)
        self.pos_weight = float(pos_weight)
        self.dice_weight = float(dice_weight)
        self.eps = eps

    def forward(self, voxel, p2v_map, preds, label):
        stc_voxel = self.base.stc_conv(voxel)
        mean_stc = torch.mean(stc_voxel.features)
        stc_weights = torch.sigmoid(stc_voxel.features - mean_stc)
        stc_weights = stc_weights[p2v_map].squeeze().detach()
        event_preds = torch.clamp(preds[p2v_map].squeeze(), self.eps, 1 - self.eps)
        pos_loss = -torch.log(event_preds)
        neg_loss = -torch.log(1 - event_preds)
        stc_loss = (label * stc_weights * pos_loss * self.pos_weight) + ((1 - label) * (1 - stc_weights) * neg_loss)
        loss = stc_loss.mean()
        if self.dice_weight > 0:
            intersection = (event_preds * label).sum()
            dice = 1 - ((2 * intersection + self.eps) / (event_preds.sum() + label.sum() + self.eps))
            loss = loss + self.dice_weight * dice
        return loss


def evaluate_model(net, cfg, threshold=0.9):
    eval_cfg = SimpleNamespace(**vars(cfg))
    eval_cfg.root = str(DEFAULT_DATA_ROOT)
    sync_global_cfg(eval_cfg)
    rows = collect_predictions(net, eval_cfg, mode="val")
    sync_global_cfg(cfg)
    return evaluate_rows(rows, [threshold])[0]


def train_one(args, variant, cfg, model_dir, config_path):
    from utils.stcloss import STCLoss

    model_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_save_root = str(model_dir)
    cfg.model_path = str(model_dir / "best_iou_seed37.pt")
    write_yaml_config(config_path, cfg)
    sync_global_cfg(cfg)
    setup(SEED)

    net = build_model(cfg, args.negative_slope, train=True)
    _, train_loader = make_dataloader(cfg, mode="train", shuffle=True)
    if args.pos_weight != 1.0 or args.dice_weight != 0.0:
        criterion = WeightedSTCLoss(k=cfg.k, t=cfg.t, cfg=cfg, pos_weight=args.pos_weight, dice_weight=args.dice_weight).cuda()
    else:
        criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float("inf")
    best_iou = -1.0
    loss_rows = []
    epoch_rows = []
    for epoch in range(int(cfg.epochs)):
        epoch_losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"{variant}-epoch-{epoch}", unit="batch")
        for batch_index, ev in enumerate(pbar):
            label = ev["seg_label"].float().cuda()
            p2v_map = ev["p2v_map"].long().cuda()
            preds, voxel = net(ev["voxel_ev"])
            loss = criterion(voxel, p2v_map, preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            value = float(loss.item())
            epoch_losses.append(value)
            loss_rows.append({"variant": variant, "epoch": epoch, "batch": batch_index, "loss": value, "lr": optimizer.param_groups[0]["lr"]})
            if value < best_loss:
                best_loss = value
                torch.save(net.state_dict(), model_dir / "best_loss_seed37.pt")
            pbar.set_postfix(loss=value)
            torch.cuda.empty_cache()
        scheduler.step()
        row = {
            "variant": variant,
            "epoch": epoch,
            "mean_loss": float(np.mean(epoch_losses)),
            "min_loss": float(np.min(epoch_losses)),
            "max_loss": float(np.max(epoch_losses)),
            "best_loss": best_loss,
            "val_iou": "",
            "val_seg_acc": "",
            "val_pd": "",
            "val_fa": "",
        }
        if epoch >= max(0, int(cfg.epochs) - int(args.eval_last_epochs)):
            metrics = evaluate_model(net, cfg, threshold=args.threshold)
            for key in ("iou", "seg_acc", "pd", "fa"):
                row[f"val_{key}"] = metrics[key]
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save(net.state_dict(), model_dir / "best_iou_seed37.pt")
            net.train()
        epoch_rows.append(row)
        write_csv(RESULTS_DIR / f"{variant}_loss_records.csv", loss_rows)
        write_csv(RESULTS_DIR / f"{variant}_epoch_summary.csv", epoch_rows)
    write_csv(RESULTS_DIR / f"{variant}_loss_records.csv", loss_rows)
    write_csv(RESULTS_DIR / f"{variant}_epoch_summary.csv", epoch_rows)
    final = {"variant": variant, "best_loss": best_loss, "best_iou": best_iou, "model_dir": str(model_dir), "config": str(config_path)}
    with open(RESULTS_DIR / f"{variant}_summary.json", "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=True)
    print(json.dumps(final, indent=2))


def command_train(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.train_root)
    cfg.epochs = args.epochs
    cfg.k = args.k
    cfg.t = args.t
    cfg.lr = args.lr
    cfg.max_events_num = args.max_events_num
    cfg.batch_size = args.batch_size
    cfg.model_path = str(args.weight or DEFAULT_V002_WEIGHT)
    variant = args.variant
    model_dir = MODELS_DIR / variant
    config_path = CONFIGS_DIR / f"{variant}.yaml"
    train_one(args, variant, cfg, model_dir, config_path)


def traditional_prediction(source_path, res, threshold, min_area):
    import traditional_baseline as tb

    pred, _ = tb.predict_sample(source_path, res, threshold, min_area, np)
    return pred.astype(np.uint8)


def command_fusion_scan(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    sync_global_cfg(cfg)
    model_thresholds = parse_thresholds(args.model_thresholds)
    low_thresholds = parse_thresholds(args.low_thresholds)
    net = build_model(cfg, args.negative_slope, args.weight)
    rows = collect_predictions(net, cfg, mode="val")
    results = []
    trad_cache = {}
    for item in rows:
        trad_cache[item["file_name"]] = traditional_prediction(Path(item["source_path"]), cfg.res, args.traditional_threshold, args.min_area)
    for model_threshold in model_thresholds:
        for low_threshold in low_thresholds:
            totals = {"tp": 0, "fp": 0, "fn": 0, "gt_positive": 0, "pred_positive": 0, "total_events": 0}
            for item in rows:
                model_high = item["preds"].numpy() >= model_threshold
                model_low = item["preds"].numpy() >= low_threshold
                trad = trad_cache[item["file_name"]].astype(bool)
                pred = np.logical_or(model_high, np.logical_and(model_low, trad)).astype(np.uint8)
                counts = metric_counts(pred, item["labels"].numpy().astype(np.uint8))
                for key in totals:
                    totals[key] += counts[key]
            union = totals["tp"] + totals["fp"] + totals["fn"]
            results.append(
                {
                    "model_threshold": model_threshold,
                    "low_threshold": low_threshold,
                    "traditional_threshold": args.traditional_threshold,
                    "min_area": args.min_area,
                    "iou": totals["tp"] / union if union else 0.0,
                    "seg_acc": totals["tp"] / totals["gt_positive"] if totals["gt_positive"] else 0.0,
                    "precision": totals["tp"] / totals["pred_positive"] if totals["pred_positive"] else 0.0,
                    **totals,
                }
            )
    write_csv(RESULTS_DIR / "traditional_fusion_scan.csv", results)
    print(json.dumps(max(results, key=lambda row: row["iou"]), indent=2))


def command_make_configs(args):
    base_cfg, _ = load_yaml_config(args.config)
    base_cfg.root = str(args.data_root)
    variants = [
        {"variant": "v002_d_pool_retrain", "train_root": str(args.d_pool_root), "negative_slope": 0.01, "pos_weight": 1.0, "dice_weight": 0.0, "k": 3, "t": 5},
        {"variant": "v002_recall_pos2_dice02", "train_root": str(args.data_root), "negative_slope": 0.01, "pos_weight": 2.0, "dice_weight": 0.2, "k": 3, "t": 5},
        {"variant": "v002_stc_t3", "train_root": str(args.data_root), "negative_slope": 0.01, "pos_weight": 1.0, "dice_weight": 0.0, "k": 3, "t": 3},
        {"variant": "v002_stc_t7", "train_root": str(args.data_root), "negative_slope": 0.01, "pos_weight": 1.0, "dice_weight": 0.0, "k": 3, "t": 7},
        {"variant": "v002_slope_0p005", "train_root": str(args.data_root), "negative_slope": 0.005, "pos_weight": 1.0, "dice_weight": 0.0, "k": 3, "t": 5},
        {"variant": "v002_slope_0p02", "train_root": str(args.data_root), "negative_slope": 0.02, "pos_weight": 1.0, "dice_weight": 0.0, "k": 3, "t": 5},
    ]
    commands = []
    for spec in variants:
        cfg = SimpleNamespace(**vars(base_cfg))
        cfg.root = spec["train_root"]
        cfg.epochs = args.epochs
        cfg.model_save_root = str(RUNTIME_MODELS_DIR / spec["variant"])
        cfg.model_path = str(RUNTIME_MODELS_DIR / spec["variant"] / "best_iou_seed37.pt")
        cfg.k = spec["k"]
        cfg.t = spec["t"]
        write_yaml_config(CONFIGS_DIR / f"{spec['variant']}.yaml", cfg)
        command = (
            f"python {RUNTIME_SCRIPT} train "
            f"--variant {spec['variant']} --train-root {spec['train_root']} --epochs {args.epochs} "
            f"--negative-slope {spec['negative_slope']} --pos-weight {spec['pos_weight']} "
            f"--dice-weight {spec['dice_weight']} --k {spec['k']} --t {spec['t']}"
        )
        commands.append({**spec, "command": command})
    write_csv(RESULTS_DIR / "planned_experiments.csv", commands)
    with open(EXPERIMENT_ROOT / "run_plan.txt", "w", encoding="utf-8") as f:
        for row in commands:
            f.write(row["command"] + "\n")
    print(json.dumps({"planned": len(commands), "run_plan": str(EXPERIMENT_ROOT / "run_plan.txt")}, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="v0.0.2 improvement experiment suite")
    sub = parser.add_subparsers(dest="command", required=True)

    common_config = argparse.ArgumentParser(add_help=False)
    common_config.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    common_config.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    common_config.add_argument("--weight", type=Path, default=DEFAULT_V002_WEIGHT)
    common_config.add_argument("--negative-slope", type=float, default=0.01)

    p = sub.add_parser("threshold-scan", parents=[common_config])
    p.add_argument("--thresholds", default="0.70:0.95:0.01")
    p.set_defaults(func=command_threshold_scan)

    p = sub.add_parser("submission", parents=[common_config])
    p.add_argument("--threshold", type=float, default=0.86)
    p.add_argument("--name", default=None)
    p.set_defaults(func=command_submission)

    p = sub.add_parser("train")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--variant", required=True)
    p.add_argument("--train-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--weight", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-events-num", type=int, default=20000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--t", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--negative-slope", type=float, default=0.01)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--dice-weight", type=float, default=0.0)
    p.add_argument("--eval-last-epochs", type=int, default=10)
    p.set_defaults(func=command_train)

    p = sub.add_parser("fusion-scan", parents=[common_config])
    p.add_argument("--model-thresholds", default="0.86,0.88,0.90,0.92")
    p.add_argument("--low-thresholds", default="0.50,0.60,0.70,0.80")
    p.add_argument("--traditional-threshold", type=float, default=2.0)
    p.add_argument("--min-area", type=int, default=3)
    p.set_defaults(func=command_fusion_scan)

    p = sub.add_parser("make-configs")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--d-pool-root", type=Path, default=DEFAULT_D_POOL_ROOT)
    p.add_argument("--epochs", type=int, default=50)
    p.set_defaults(func=command_make_configs)
    return parser


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
