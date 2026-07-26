#!/usr/bin/env python3
import argparse
import csv
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None
try:
    import yaml
except ImportError:
    yaml = None
try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None
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

PROJECT_ROOT = Path(os.environ.get("EVUAV_PROJECT_ROOT", "/root/EV-UAV"))
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENT_ROOT = Path(__file__).resolve().parent
RUNTIME_EXPERIMENT_ROOT = Path(os.environ.get("EVUAV_RUNTIME_EXPERIMENT_ROOT", str(EXPERIMENT_ROOT)))
CONFIGS_DIR = EXPERIMENT_ROOT / "configs"
RESULTS_DIR = EXPERIMENT_ROOT / "results"
MODELS_DIR = EXPERIMENT_ROOT / "models"
SUBMISSION_DIR = EXPERIMENT_ROOT / "submission"
LOGS_DIR = EXPERIMENT_ROOT / "logs"
CACHE_DIR = EXPERIMENT_ROOT / "cache"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
DEFAULT_DATA_ROOT = Path(os.environ.get("EVUAV_DATA_ROOT", "/root/EV-UAV-dataset"))
DEFAULT_V3_CONFIG = CONFIGS_DIR / "v300_handcrafted_feature.yaml"
SEED = 37


def require_numpy_scipy():
    missing = []
    if np is None:
        missing.append("numpy")
    if cKDTree is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError("Missing feature dependencies: " + ", ".join(missing) + ". Install them or run inside the EV-UAV Docker/conda environment.")


def require_runtime():
    missing = []
    if np is None:
        missing.append("numpy")
    if cKDTree is None:
        missing.append("scipy")
    if torch is None:
        missing.append("torch")
    if tqdm is None:
        missing.append("tqdm")
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing) + ". Run inside Docker container evuav with conda env evuav.")


def ensure_dirs():
    for path in (CONFIGS_DIR, RESULTS_DIR, MODELS_DIR, SUBMISSION_DIR, LOGS_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_cfg_namespace():
    return SimpleNamespace(
        model_name="evissgnet",
        width=12,
        block_residual=True,
        block_reps=2,
        use_coords=True,
        input_channel=8,
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
        model_save_root=str(MODELS_DIR),
        vis=False,
        eval=True,
        roc=True,
        pd_detT=50,
        correct_thresh=0.0001,
        save=True,
        model_path=str(MODELS_DIR / "best_iou_seed37.pt"),
        feature_k=32,
        feature_radius=0.035,
        density_sigma=0.018,
        linearity_d0=12.0,
        temporal_tau=0.02,
        negative_slope=0.01,
    )


def load_yaml_config(path):
    if yaml is None or not Path(path).exists():
        return default_cfg_namespace(), None
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    flat = {}
    for section in raw.values():
        if isinstance(section, dict):
            flat.update(section)
    cfg = default_cfg_namespace()
    cfg.__dict__.update(flat)
    return cfg, raw


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
            "input_channel": 8,
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
        "V3_FEATURES": {
            "feature_k": int(cfg.feature_k),
            "feature_radius": float(cfg.feature_radius),
            "density_sigma": float(cfg.density_sigma),
            "linearity_d0": float(cfg.linearity_d0),
            "temporal_tau": float(cfg.temporal_tau),
            "negative_slope": float(cfg.negative_slope),
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


def command_prepare_config(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.base_config)
    cfg.input_channel = 8
    cfg.root = str(args.data_root)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.max_events_num = args.max_events_num
    cfg.model_save_root = str(MODELS_DIR)
    cfg.model_path = str(MODELS_DIR / "best_iou_seed37.pt")
    cfg.feature_k = args.feature_k
    cfg.feature_radius = args.feature_radius
    cfg.density_sigma = args.density_sigma
    cfg.linearity_d0 = args.linearity_d0
    cfg.temporal_tau = args.temporal_tau
    cfg.negative_slope = args.negative_slope
    write_yaml_config(args.output, cfg)
    print(json.dumps({"config": str(args.output), "input_channel": 8, "model_save_root": str(MODELS_DIR)}, indent=2))


def minmax01(values):
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = float(values.min()) if values.size else 0.0
    vmax = float(values.max()) if values.size else 0.0
    if vmax <= vmin:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def event_field(source_ev, field, fallback):
    if getattr(source_ev, "dtype", None) is not None and source_ev.dtype.names and field in source_ev.dtype.names:
        return source_ev[field]
    return fallback


def normalized_xyz(data):
    ev_loc = data["ev_loc"].astype(np.float32)
    evs_norm = data["evs_norm"].astype(np.float32)
    source_ev = data["ev"] if "ev" in data.files else None
    if source_ev is not None:
        x = event_field(source_ev, "x", ev_loc[:, 0]).astype(np.float32)
        y = event_field(source_ev, "y", ev_loc[:, 1]).astype(np.float32)
        t = event_field(source_ev, "t", ev_loc[:, 2]).astype(np.float32)
        p = event_field(source_ev, "p", evs_norm[:, 3]).astype(np.float32)
    else:
        x, y, t = ev_loc[:, 0], ev_loc[:, 1], ev_loc[:, 2]
        p = evs_norm[:, 3]
    coords = np.column_stack([x, y, t]).astype(np.float32)
    coords_norm = np.empty_like(coords, dtype=np.float32)
    for col in range(3):
        coords_norm[:, col] = minmax01(coords[:, col])
    return coords_norm, minmax01(t), p


def compute_handcrafted_features(npz_path, cfg):
    require_numpy_scipy()
    with np.load(npz_path, allow_pickle=False) as data:
        evs_norm = data["evs_norm"].astype(np.float32)
        coords_norm, t_norm, polarity = normalized_xyz(data)
    n = evs_norm.shape[0]
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    k = min(max(2, int(cfg.feature_k)), n)
    radius = float(cfg.feature_radius)
    sigma = max(float(cfg.density_sigma), 1e-6)
    tau = max(float(cfg.temporal_tau), 1e-6)
    tree = cKDTree(coords_norm)
    distances, indices = tree.query(coords_norm, k=k, distance_upper_bound=radius, workers=-1)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    finite = np.isfinite(distances) & (indices < n)
    density_raw = np.zeros(n, dtype=np.float32)
    linearity = np.zeros(n, dtype=np.float32)
    polarity_consistency = np.zeros(n, dtype=np.float32)
    temporal_continuity = np.zeros(n, dtype=np.float32)
    raw_counts = finite.sum(axis=1).astype(np.float32)
    for i in range(n):
        valid_idx = indices[i, finite[i]]
        valid_dist = distances[i, finite[i]]
        if valid_idx.size == 0:
            continue
        weights = np.exp(-((valid_dist / sigma) ** 2))
        density_raw[i] = float(weights.sum())
        neigh = coords_norm[valid_idx]
        if valid_idx.size >= 3:
            centered = neigh - neigh.mean(axis=0, keepdims=True)
            cov = np.dot(centered.T, centered) / max(valid_idx.size - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)[::-1]
            denom = float(eigvals[0] + 1e-8)
            base_l = float((eigvals[0] - eigvals[1]) / denom) if denom > 0 else 0.0
            linearity[i] = np.clip(base_l, 0.0, 1.0) * min(float(valid_idx.size) / float(cfg.linearity_d0), 1.0)
        pol = polarity[valid_idx]
        pos = float((pol > 0).sum())
        neg = float(valid_idx.size - pos)
        polarity_consistency[i] = abs(pos - neg) / (pos + neg + 1e-8)
        temporal_continuity[i] = float(np.mean(np.exp(-np.abs(t_norm[i] - t_norm[valid_idx]) / tau)))
    density = minmax01(np.log1p(density_raw))
    features = np.column_stack([
        density,
        np.clip(linearity, 0.0, 1.0),
        np.clip(polarity_consistency, 0.0, 1.0),
        np.clip(temporal_continuity, 0.0, 1.0),
    ]).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(features, 0.0, 1.0).astype(np.float32)


def cache_path_for(npz_path):
    split = Path(npz_path).parent.name
    return CACHE_DIR / split / (Path(npz_path).stem + "_dLCT.npy")


def load_or_compute_features(npz_path, cfg, use_cache=True):
    cache_path = cache_path_for(npz_path)
    if use_cache and cache_path.exists():
        return np.load(cache_path)
    features = compute_handcrafted_features(npz_path, cfg)
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, features)
    return features


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


def replace_relu(module, negative_slope):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        else:
            replace_relu(child, negative_slope)


def sync_global_cfg(cfg):
    argv_backup = sys.argv[:]
    sys.argv = [sys.argv[0], "--config", str(DEFAULT_CONFIG)]
    try:
        import configs.configs as config_module
    finally:
        sys.argv = argv_backup
    config_module.cfg.__dict__.update(vars(cfg))
    return config_module.cfg


def build_model(cfg, train=False, weight_path=None):
    require_runtime()
    from model.evspsegnet import evspsegnet
    net = evspsegnet(cfg)
    replace_relu(net, float(cfg.negative_slope))
    net = net.train() if train else net.eval()
    net.cuda()
    if weight_path is not None:
        net.load_state_dict(torch.load(weight_path, map_location="cuda:0"))
    return net


class V3EvUAV(torch.utils.data.Dataset if torch is not None else object):
    def __init__(self, configs, mode="train", use_cache=True):
        self.configs = configs
        self.mode = mode
        self.root = Path(configs.root) / mode
        self.file_list = sorted(os.listdir(self.root))
        self.use_cache = use_cache

    def __getitem__(self, idx):
        path = self.root / self.file_list[idx]
        with np.load(path, allow_pickle=False) as events:
            evs_norm = events["evs_norm"].astype(np.float32)
            ev_loc = events["ev_loc"].astype(np.int64)
            seg_label = evs_norm[:, 4].astype(np.float32)
            idx_label = evs_norm[:, 5]
        features = load_or_compute_features(path, self.configs, self.use_cache)
        if features.shape[0] != evs_norm.shape[0]:
            raise ValueError(f"{path.name}: feature rows {features.shape[0]} != event rows {evs_norm.shape[0]}")
        evs_feature = np.concatenate([evs_norm[:, 0:4], features], axis=1).astype(np.float32)
        if self.mode == "train":
            num_events = ev_loc.shape[0]
            if num_events >= int(self.configs.max_events_num):
                sample_idx = np.random.choice(num_events, int(self.configs.max_events_num), replace=False)
                ev_loc = ev_loc[sample_idx]
                evs_feature = evs_feature[sample_idx]
                seg_label = seg_label[sample_idx]
                idx_label = idx_label[sample_idx]
        return {"ev_loc": ev_loc, "evs_feature": evs_feature, "seg_label": seg_label, "idx": idx_label, "file_name": self.file_list[idx], "source_path": str(path)}

    def __len__(self):
        return len(self.file_list)

    @staticmethod
    def custom_collate(batch):
        from dataset.basedataset import voxelization, voxelization_idx
        import spconv.pytorch as spconv
        batch_size = len(batch)
        loc_batches = []
        feature_batches = []
        seg_label_batches = []
        idx_label_batches = []
        meta = []
        for i, ev in enumerate(batch):
            ev_loc = ev["ev_loc"]
            loc_batches.append(np.hstack((i * np.ones((ev_loc.shape[0], 1)), ev_loc)))
            feature_batches.append(ev["evs_feature"])
            seg_label_batches.append(ev["seg_label"])
            idx_label_batches.append(ev["idx"])
            meta.append({"file_name": ev["file_name"], "source_path": ev["source_path"], "local_index": i})
        locs_batches = torch.from_numpy(np.concatenate(loc_batches, axis=0)).to(torch.int64).contiguous()
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs_batches, batch_size, 4)
        feature_tensor = torch.from_numpy(np.concatenate(feature_batches, axis=0)).float().contiguous()
        voxel_feats = voxelization(feature_tensor.cuda(), v2p_map.cuda(), 4).cuda()
        spatial_shape = np.array([11 * 32, 9 * 32, 256 * 32])
        voxel_ev = spconv.SparseConvTensor(voxel_feats, voxel_locs.int().cuda(), spatial_shape, batch_size)
        return {
            "voxel_ev": voxel_ev,
            "seg_label": torch.from_numpy(np.concatenate(seg_label_batches, axis=0)),
            "p2v_map": p2v_map,
            "locs": locs_batches,
            "idx_label": np.concatenate(idx_label_batches, axis=0),
            "meta": meta,
        }


def make_dataloader(cfg, mode, shuffle=False):
    dataset = V3EvUAV(cfg, mode=mode, use_cache=True)
    sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset)))) if shuffle else None
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=sampler, shuffle=False if sampler is not None else shuffle)
    return dataset, loader


def metric_counts(pred, label):
    pred_bool = pred.astype(bool)
    label_bool = label.astype(bool)
    tp = int(np.logical_and(pred_bool, label_bool).sum())
    fp = int(np.logical_and(pred_bool, ~label_bool).sum())
    fn = int(np.logical_and(~pred_bool, label_bool).sum())
    gt_positive = int(label_bool.sum())
    pred_positive = int(pred_bool.sum())
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn, "gt_positive": gt_positive, "pred_positive": pred_positive, "total_events": int(len(label_bool)), "iou": tp / union if union else 0.0, "seg_acc": tp / gt_positive if gt_positive else 0.0}


def collect_predictions(net, cfg, mode="val"):
    dataset, loader = make_dataloader(cfg, mode=mode, shuffle=False)
    rows = []
    with torch.no_grad():
        for ev in tqdm.tqdm(loader, desc=f"predict-{mode}", unit="sample"):
            p2v_map = ev["p2v_map"].long().cuda()
            ev_locs = ev["locs"]
            labels = ev["seg_label"].float().cpu()
            idx = ev["idx_label"]
            preds, _ = net(ev["voxel_ev"])
            preds = preds[p2v_map].reshape(-1).cpu()
            for meta in ev["meta"]:
                local_index = meta["local_index"]
                mask = ev_locs[:, 0].long() == local_index
                rows.append({
                    "sample": len(rows),
                    "file_name": meta["file_name"],
                    "source_path": meta["source_path"],
                    "preds": preds[mask].clone(),
                    "labels": labels[mask].clone(),
                    "idx": idx[mask.numpy()] if isinstance(idx, np.ndarray) else idx[mask],
                    "locs": ev_locs[mask].clone(),
                })
    return rows


def evaluate_rows(rows, threshold=0.9):
    from utils.eval import evalute
    evaluator = evalute(SimpleNamespace(roc=True, pd_detT=50, correct_thresh=0.0001))
    totals = {"tp": 0, "fp": 0, "fn": 0, "gt_positive": 0, "pred_positive": 0, "total_events": 0}
    for item in rows:
        pred_binary = (item["preds"].numpy() >= threshold).astype(np.uint8)
        label = item["labels"].numpy().astype(np.uint8)
        counts = metric_counts(pred_binary, label)
        for key in totals:
            totals[key] += counts[key]
        evaluator.matches[str(item["sample"])] = {"seg_pred": item["preds"].clone(), "seg_gt": item["labels"].clone()}
        evaluator.roc_update(item["locs"][:, 3], item["preds"].clone(), item["idx"], item["labels"].clone(), item["locs"].float(), thresh=threshold)
    union = totals["tp"] + totals["fp"] + totals["fn"]
    pd_value, fa_value = evaluator.cal_roc()
    return {"threshold": threshold, "iou": totals["tp"] / union if union else 0.0, "seg_acc": totals["tp"] / totals["gt_positive"] if totals["gt_positive"] else 0.0, "pd": float(pd_value), "fa": float(fa_value), **totals}


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(net, cfg, threshold):
    was_training = net.training
    net.eval()
    rows = collect_predictions(net, cfg, mode="val")
    metrics = evaluate_rows(rows, threshold=threshold)
    if was_training:
        net.train()
    return metrics


def train_one(args):
    from utils.stcloss import STCLoss
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.epochs = args.epochs
    cfg.max_events_num = args.max_events_num
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.model_save_root = str(MODELS_DIR)
    cfg.model_path = str(MODELS_DIR / "best_iou_seed37.pt")
    cfg.input_channel = 8
    sync_global_cfg(cfg)
    write_yaml_config(DEFAULT_V3_CONFIG, cfg)
    setup(SEED)
    net = build_model(cfg, train=True)
    _, train_loader = make_dataloader(cfg, mode="train", shuffle=True)
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    best_loss = float("inf")
    best_iou = -1.0
    loss_rows = []
    epoch_rows = []
    for epoch in range(int(cfg.epochs)):
        epoch_losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"v300-epoch-{epoch}", unit="batch")
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
            loss_rows.append({"epoch": epoch, "batch": batch_index, "loss": value, "lr": optimizer.param_groups[0]["lr"]})
            if value < best_loss:
                best_loss = value
                torch.save(net.state_dict(), MODELS_DIR / "best_loss_seed37.pt")
            pbar.set_postfix(loss=value)
            torch.cuda.empty_cache()
        scheduler.step()
        row = {"epoch": epoch, "mean_loss": float(np.mean(epoch_losses)), "min_loss": float(np.min(epoch_losses)), "max_loss": float(np.max(epoch_losses)), "best_loss": best_loss, "val_iou": "", "val_seg_acc": "", "val_pd": "", "val_fa": ""}
        if epoch >= max(0, int(cfg.epochs) - int(args.eval_last_epochs)):
            metrics = evaluate_model(net, cfg, args.threshold)
            row.update({"val_iou": metrics["iou"], "val_seg_acc": metrics["seg_acc"], "val_pd": metrics["pd"], "val_fa": metrics["fa"]})
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save(net.state_dict(), MODELS_DIR / "best_iou_seed37.pt")
        epoch_rows.append(row)
        write_csv(RESULTS_DIR / "v300_loss_records.csv", loss_rows)
        write_csv(RESULTS_DIR / "v300_epoch_summary.csv", epoch_rows)
    summary = {"status": "ok", "epochs": int(cfg.epochs), "best_loss": best_loss, "best_iou": best_iou, "model_path": str(MODELS_DIR / "best_iou_seed37.pt"), "config": str(DEFAULT_V3_CONFIG)}
    with open(RESULTS_DIR / "v300_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2))


def command_train(args):
    ensure_dirs()
    train_one(args)


def command_evaluate(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.input_channel = 8
    sync_global_cfg(cfg)
    setup(SEED)
    net = build_model(cfg, train=False, weight_path=args.weight)
    rows = collect_predictions(net, cfg, mode="val")
    metrics = evaluate_rows(rows, threshold=args.threshold)
    with open(RESULTS_DIR / "v300_validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=True)
    print(json.dumps(metrics, indent=2))


def save_prediction(source_path, output_path, prediction):
    with np.load(source_path, allow_pickle=False) as data:
        source_ev = data["ev"]
        if len(source_ev) != len(prediction):
            raise ValueError(f"{Path(source_path).name}: event count mismatch")
        predicted_ev = np.empty(len(source_ev), dtype=[("x", source_ev.dtype["x"]), ("y", source_ev.dtype["y"]), ("t", source_ev.dtype["t"]), ("p", source_ev.dtype["p"]), ("label", np.int64)])
        for field in ("x", "y", "t", "p"):
            predicted_ev[field] = source_ev[field]
        predicted_ev["label"] = prediction.astype(np.int64)
    np.savetxt(output_path, predicted_ev, fmt=["%d", "%d", "%.9f", "%d", "%d"], delimiter=" ")


def write_zip(txt_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            zf.write(txt_path, arcname=txt_path.name)


def command_submission(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.input_channel = 8
    sync_global_cfg(cfg)
    setup(SEED)
    net = build_model(cfg, train=False, weight_path=args.weight)
    rows = collect_predictions(net, cfg, mode="val")
    name = args.name
    txt_dir = SUBMISSION_DIR / name / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for item in rows:
        pred = (item["preds"].numpy() >= args.threshold).astype(np.uint8)
        save_prediction(item["source_path"], txt_dir / f"{Path(item['file_name']).stem}.txt", pred)
    zip_path = SUBMISSION_DIR / f"{name}.zip"
    write_zip(txt_dir, zip_path)
    print(json.dumps({"txt_count": len(list(txt_dir.glob('*.txt'))), "zip": str(zip_path)}, indent=2))


def validate_txt_against_npz(txt_path, npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        source_ev = data["ev"]
    arr = np.loadtxt(txt_path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != len(source_ev):
        return f"line count mismatch: {arr.shape[0]} != {len(source_ev)}"
    if arr.shape[1] != 5:
        return f"field count mismatch: {arr.shape[1]}"
    labels = arr[:, 4]
    if not np.all((labels == 0) | (labels == 1)):
        return "labels contain non-binary values"
    checks = [("x", arr[:, 0]), ("y", arr[:, 1]), ("t", arr[:, 2]), ("p", arr[:, 3])]
    for field, values in checks:
        source = source_ev[field]
        if field == "t":
            ok = np.allclose(values.astype(float), source.astype(float), atol=1e-9, rtol=0)
        else:
            ok = np.array_equal(values.astype(source.dtype), source)
        if not ok:
            return f"{field} column is not aligned with original ev"
    return None


def command_validate_submission(args):
    require_numpy_scipy()
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
    for npz_path in expected:
        txt_path = txt_dir / f"{npz_path.stem}.txt"
        if txt_path.exists():
            error = validate_txt_against_npz(txt_path, npz_path)
            if error:
                errors.append(f"{txt_path.name}: {error}")
    if not zip_path.exists():
        errors.append(f"zip not found: {zip_path}")
    else:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        if any("/" in name.rstrip("/") for name in names):
            errors.append("zip contains nested paths")
        if set(names) != expected_names:
            errors.append("zip root txt names do not match expected val files")
    report = {"status": "ok" if not errors else "failed", "txt_dir": str(txt_dir), "zip": str(zip_path), "txt_count": len(txt_files), "errors": errors}
    with open(RESULTS_DIR / "v300_submission_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def command_quick_check(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    sample = sorted((args.data_root / args.split).glob("*.npz"))[0]
    start = time.time()
    features = load_or_compute_features(sample, cfg, use_cache=args.cache)
    with np.load(sample, allow_pickle=False) as data:
        rows = int(data["evs_norm"].shape[0])
    report = {"sample": str(sample), "rows": rows, "feature_shape": list(features.shape), "finite": bool(np.isfinite(features).all()), "min": features.min(axis=0).round(6).tolist(), "max": features.max(axis=0).round(6).tolist(), "seconds": round(time.time() - start, 3)}
    with open(RESULTS_DIR / "v300_quick_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="v3.0.0 handcrafted-feature EV-SpSegNet")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_V3_CONFIG)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p = sub.add_parser("prepare-config")
    p.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_V3_CONFIG)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-events-num", type=int, default=20000)
    p.add_argument("--feature-k", type=int, default=32)
    p.add_argument("--feature-radius", type=float, default=0.035)
    p.add_argument("--density-sigma", type=float, default=0.018)
    p.add_argument("--linearity-d0", type=float, default=12.0)
    p.add_argument("--temporal-tau", type=float, default=0.02)
    p.add_argument("--negative-slope", type=float, default=0.01)
    p.set_defaults(func=command_prepare_config)
    p = sub.add_parser("quick-check", parents=[common])
    p.add_argument("--split", default="train")
    p.add_argument("--cache", action="store_true")
    p.set_defaults(func=command_quick_check)
    p = sub.add_parser("train", parents=[common])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-events-num", type=int, default=20000)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--eval-last-epochs", type=int, default=10)
    p.set_defaults(func=command_train)
    p = sub.add_parser("evaluate", parents=[common])
    p.add_argument("--weight", type=Path, default=MODELS_DIR / "best_iou_seed37.pt")
    p.add_argument("--threshold", type=float, default=0.9)
    p.set_defaults(func=command_evaluate)
    p = sub.add_parser("submission", parents=[common])
    p.add_argument("--weight", type=Path, default=MODELS_DIR / "best_iou_seed37.pt")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--name", default="v300_handcrafted_feature_best_submission")
    p.set_defaults(func=command_submission)
    p = sub.add_parser("validate-submission")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--txt-dir", type=Path, default=SUBMISSION_DIR / "v300_handcrafted_feature_best_submission" / "txt")
    p.add_argument("--zip", type=Path, default=SUBMISSION_DIR / "v300_handcrafted_feature_best_submission.zip")
    p.set_defaults(func=command_validate_submission)
    return parser


def main():
    ensure_dirs()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
