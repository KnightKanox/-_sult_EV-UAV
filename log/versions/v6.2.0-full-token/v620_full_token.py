#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import sys
import time
import zipfile
from datetime import date
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
try:
    project_root_exists = PROJECT_ROOT.exists()
except OSError:
    project_root_exists = False
if not project_root_exists:
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
CHARTS_DIR = EXPERIMENT_ROOT / "charts"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "evisseg_evuav.yaml"
DEFAULT_DATA_ROOT = Path(os.environ.get("EVUAV_DATA_ROOT", "/root/EV-UAV-dataset"))
DEFAULT_V4_CONFIG = CONFIGS_DIR / "v620_full_token.yaml"
VERSION = "v6.2.0"
MAX_DIRECTION_VOXELS = 1024
EXPERIMENT_NAME = "full_token"
SEED = 316
TRAJECTORY_PROJECTION_DICE_WEIGHT = 0.3
TRAJECTORY_PROJECTION_DICE_MODE = "xy_mean_projection_dice_train_only"
TRAJECTORY_DIRECTION_WEIGHT = 0.2
DIRECTION_NEIGHBOR_K = 16
DIRECTION_HEAD_HIDDEN = 128
TRANSFORMER_TOKEN_LIMIT = 8192
TRANSFORMER_NUM_HEADS = 2
TRANSFORMER_FF_MULTIPLIER = 1
TRANSFORMER_DROPOUT = 0.0
INPUT_STATS_FILENAME = "v620_input_stats.json"
QUICK_CHECK_FILENAME = "v620_quick_check.json"
TRAINING_SUMMARY_FILENAME = "v620_training_summary.json"
VALIDATION_METRICS_FILENAME = "v620_validation_metrics.json"
THRESHOLD_SCAN_CSV = "v620_threshold_scan.csv"
THRESHOLD_SCAN_JSON = "v620_threshold_scan.json"
THRESHOLD_SCAN_MD = "v620_threshold_scan_summary.md"
SUBMISSION_VALIDATION_FILENAME = "v620_submission_validation.json"


def require_numpy_scipy():
    missing = []
    if np is None:
        missing.append("numpy")
    if cKDTree is None:
        missing.append("scipy")
    if missing:
        raise RuntimeError("Missing feature dependencies: " + ", ".join(missing) + ". Install them or run inside the EV-UAV Docker/conda environment.")


def require_runtime(require_sparse=True):
    missing = []
    if np is None:
        missing.append("numpy")
    if cKDTree is None:
        missing.append("scipy")
    if torch is None:
        missing.append("torch")
    if tqdm is None:
        missing.append("tqdm")
    if require_sparse:
        try:
            import spconv.pytorch  # noqa: F401
            from lib.hais_ops import HAIS_OP  # noqa: F401
        except Exception as exc:
            missing.append(f"spconv/HAIS_OP ({exc})")
    if missing:
        raise RuntimeError("Missing runtime dependencies: " + ", ".join(missing) + ". Run inside Docker container evuav with conda env evuav.")


def ensure_dirs():
    for path in (CONFIGS_DIR, RESULTS_DIR, MODELS_DIR, SUBMISSION_DIR, LOGS_DIR, CACHE_DIR, CHARTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def default_cfg_namespace():
    return SimpleNamespace(
        model_name="full_token",
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
        max_events_num=50000,
        seed=SEED,
        model_save_root=str(MODELS_DIR),
        vis=False,
        eval=True,
        roc=True,
        pd_detT=50,
        correct_thresh=0.0001,
        save=True,
        model_path=str(MODELS_DIR / "best_iou_seed316.pt"),
        feature_k=32,
        feature_radius=0.035,
        density_sigma=0.018,
        linearity_d0=12.0,
        temporal_tau=0.02,
        negative_slope=0.01,
        stem_channels=24,
        stage2_channels=40,
        stage3_channels=64,
        bottleneck_channels=96,
        no_patchattention=True,
        downsample_strides=[[2, 2, 2], [2, 2, 2], [2, 2, 4]],
        input_norm_epsilon=1e-6,
        input_norm_clip=False,
        input_norm_stats_path=str(CACHE_DIR / INPUT_STATS_FILENAME),
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
            "model_name": "full_token",
            "input_channel": 8,
            "stem_channels": 24,
            "stage2_channels": 40,
            "stage3_channels": 64,
            "bottleneck_channels": 96,
            "downsample_strides": [[2, 2, 2], [2, 2, 2], [2, 2, 4]],
            "no_patchattention": True,
            "activation": "LeakyReLU",
            "bottleneck_direction_aggregation": True,
            "trajectory_direction_head": True,
            "direction_head_hidden": DIRECTION_HEAD_HIDDEN,
            "direction_neighbor_k": DIRECTION_NEIGHBOR_K,
            "transformer_enabled": True,
            "transformer_insert_position": "after_bottleneck_mstb",
            "transformer_token_limit": TRANSFORMER_TOKEN_LIMIT,
            "transformer_num_heads": TRANSFORMER_NUM_HEADS,
            "transformer_ff_multiplier": TRANSFORMER_FF_MULTIPLIER,
            "transformer_dropout": TRANSFORMER_DROPOUT,
        },
        "DATA": {
            "root": str(cfg.root),
            "whole_t": int(cfg.whole_t),
            "res": list(cfg.res),
        },
        "TRAIN": {
            "epochs": int(cfg.epochs),
            "batch_size": int(cfg.batch_size),
            "train_workers": int(getattr(cfg, "train_workers", 8)),
            "optim": "Adam",
            "lr": float(cfg.lr),
            "k": int(cfg.k),
            "t": int(cfg.t),
            "max_events_num": 50000,
            "seed": SEED,
            "model_save_root": str(MODELS_DIR),
            "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT,
            "trajectory_projection_dice_mode": TRAJECTORY_PROJECTION_DICE_MODE,
            "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT,
            "auxiliary_supervision": TRAJECTORY_PROJECTION_DICE_MODE + "+direction_head",
        },
        "TEST": {
            "vis": False,
            "eval": True,
            "roc": True,
            "pd_detT": int(cfg.pd_detT),
            "correct_thresh": float(cfg.correct_thresh),
            "save": True,
            "model_path": str(MODELS_DIR / "best_iou_seed316.pt"),
        },
        "EXPERIMENT": {
            "version": VERSION,
            "experiment": EXPERIMENT_NAME,
            "experiment_root": str(EXPERIMENT_ROOT),
            "runtime_experiment_root": str(RUNTIME_EXPERIMENT_ROOT),
            "results_dir": str(RESULTS_DIR),
            "models_dir": str(MODELS_DIR),
            "submission_dir": str(SUBMISSION_DIR),
            "cache_dir": str(CACHE_DIR),
            "charts_dir": str(CHARTS_DIR),
            "input_stats_filename": INPUT_STATS_FILENAME,
        },
        "INPUT_NORMALIZATION": {
            "enabled": True,
            "stats_source_split": "train",
            "input_norm_epsilon": float(getattr(cfg, "input_norm_epsilon", 1e-6)),
            "input_norm_clip": bool(getattr(cfg, "input_norm_clip", False)),
            "input_norm_stats_path": str(input_stats_path(cfg)),
            "channels": stats_channel_names(),
        },
        "V3_FEATURES_REUSED": {
            "feature_k": int(cfg.feature_k),
            "feature_radius": float(cfg.feature_radius),
            "density_sigma": float(cfg.density_sigma),
            "linearity_d0": float(cfg.linearity_d0),
            "temporal_tau": float(cfg.temporal_tau),
            "negative_slope": float(cfg.negative_slope),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)
        return
    lines = []
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            if isinstance(value, list):
                rendered = json.dumps(value)
            elif isinstance(value, bool):
                rendered = str(value).lower()
            else:
                rendered = str(value)
            lines.append(f"  {key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_prepare_config(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.base_config)
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    cfg.root = str(args.data_root)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.max_events_num = 50000
    cfg.optim = "Adam"
    cfg.model_save_root = str(MODELS_DIR)
    cfg.seed = SEED
    cfg.model_path = str(MODELS_DIR / "best_iou_seed316.pt")
    cfg.feature_k = args.feature_k
    cfg.feature_radius = args.feature_radius
    cfg.density_sigma = args.density_sigma
    cfg.linearity_d0 = args.linearity_d0
    cfg.temporal_tau = args.temporal_tau
    cfg.negative_slope = args.negative_slope
    write_yaml_config(args.output, cfg)
    print(json.dumps({"config": str(args.output), "version": VERSION, "experiment": EXPERIMENT_NAME, "input_channel": 8, "max_events_num": 50000, "optimizer": "Adam", "seed": SEED, "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "bottleneck_direction_aggregation": True, "trajectory_direction_head": True, "transformer_enabled": True, "transformer_insert_position": "after_bottleneck_mstb", "transformer_token_limit": TRANSFORMER_TOKEN_LIMIT, "transformer_num_heads": TRANSFORMER_NUM_HEADS, "input_norm_epsilon": getattr(cfg, "input_norm_epsilon", 1e-6), "input_norm_clip": getattr(cfg, "input_norm_clip", False), "input_norm_stats_path": str(input_stats_path(cfg))}, indent=2))


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
    features = np.column_stack([
        minmax01(np.log1p(density_raw)),
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


def input_stats_path(cfg):
    stats_path = getattr(cfg, "input_norm_stats_path", None)
    return Path(stats_path) if stats_path else CACHE_DIR / INPUT_STATS_FILENAME


def stats_channel_names():
    return ["x", "y", "t", "p", "d", "L_star", "C", "T"]


def feature_cfg_from(cfg):
    return SimpleNamespace(
        feature_k=int(getattr(cfg, "feature_k", 32)),
        feature_radius=float(getattr(cfg, "feature_radius", 0.035)),
        density_sigma=float(getattr(cfg, "density_sigma", 0.018)),
        linearity_d0=float(getattr(cfg, "linearity_d0", 12.0)),
        temporal_tau=float(getattr(cfg, "temporal_tau", 0.02)),
    )


def compute_input_stats(data_root, cfg, split="train"):
    require_numpy_scipy()
    data_root = Path(data_root)
    split_dir = data_root / split
    samples = sorted(split_dir.glob("*.npz"))
    if not samples:
        raise FileNotFoundError(f"no npz files found under {split_dir}")
    per_channel = {name: [] for name in stats_channel_names()}
    total_count = 0
    feature_cfg = feature_cfg_from(cfg)
    for sample in samples:
        with np.load(sample, allow_pickle=False) as data:
            evs_norm = data["evs_norm"].astype(np.float32)
        features = load_or_compute_features(sample, feature_cfg, use_cache=True)
        if features.shape[0] != evs_norm.shape[0]:
            raise ValueError(f"{sample.name}: feature rows {features.shape[0]} != event rows {evs_norm.shape[0]}")
        merged = np.concatenate([evs_norm[:, 0:4], features], axis=1).astype(np.float32)
        total_count += int(merged.shape[0])
        for col, name in enumerate(stats_channel_names()):
            per_channel[name].append(merged[:, col])
    stats = {
        "version": VERSION,
        "experiment": EXPERIMENT_NAME,
        "split": split,
        "data_root": str(data_root),
        "sample_count": len(samples),
        "event_count": total_count,
        "channels": [],
        "input_norm": {
            "epsilon": float(getattr(cfg, "input_norm_epsilon", 1e-6)),
            "clip": bool(getattr(cfg, "input_norm_clip", False)),
        },
    }
    epsilon = float(getattr(cfg, "input_norm_epsilon", 1e-6))
    for name in stats_channel_names():
        values = np.concatenate(per_channel[name], axis=0).astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        stats["channels"].append({
            "name": name,
            "mean": float(values.mean()),
            "std": max(float(values.std(ddof=0)), epsilon),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(np.quantile(values, 0.01)),
            "p99": float(np.quantile(values, 0.99)),
            "count": int(values.shape[0]),
        })
    stats_path = input_stats_path(cfg)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=True)
    return stats


def command_compute_stats(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    stats = compute_input_stats(args.data_root, cfg, split=args.split)
    print(json.dumps({"status": "ok", "stats_path": str(input_stats_path(cfg)), "sample_count": stats["sample_count"], "event_count": stats["event_count"], "channels": stats["channels"]}, indent=2, ensure_ascii=True))


def load_input_stats(cfg):
    stats_path = input_stats_path(cfg)
    if not stats_path.exists():
        compute_input_stats(cfg.root, cfg, split="train")
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    if len(stats.get("channels", [])) != 8:
        raise ValueError(f"invalid input stats at {stats_path}")
    return stats


def normalize_input_features(features, stats, cfg):
    arr = np.asarray(features, dtype=np.float32).copy()
    epsilon = float(getattr(cfg, "input_norm_epsilon", 1e-6))
    clip_enabled = bool(getattr(cfg, "input_norm_clip", False))
    for idx, channel in enumerate(stats["channels"]):
        if clip_enabled:
            arr[:, idx] = np.clip(arr[:, idx], float(channel["p01"]), float(channel["p99"]))
        std = max(float(channel["std"]), epsilon)
        arr[:, idx] = (arr[:, idx] - float(channel["mean"])) / std
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


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
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        return cfg
    finally:
        sys.argv = argv_backup
    config_module.cfg.__dict__.update(vars(cfg))
    return config_module.cfg


def sparse_replace(x, features):
    return x.replace_feature(features)


def build_model_classes():
    require_runtime()
    import functools
    import spconv.pytorch as spconv
    algo = spconv.ops.ConvAlgo.Native

    class GatedInputFusion(nn.Module):
        def __init__(self, out_channels=24, negative_slope=0.01):
            super().__init__()
            self.raw_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.LeakyReLU(negative_slope, inplace=False), nn.Linear(out_channels, out_channels))
            self.geo_mlp = nn.Sequential(nn.Linear(4, out_channels), nn.LeakyReLU(negative_slope, inplace=False), nn.Linear(out_channels, out_channels))
            self.gate = nn.Sequential(nn.Linear(8, out_channels), nn.Sigmoid())
            self.act = nn.LeakyReLU(negative_slope, inplace=False)

        def forward(self, features):
            raw = self.raw_mlp(features[:, :4])
            geo = self.geo_mlp(features[:, 4:8])
            gate = self.gate(features[:, :8])
            return self.act(gate * raw + (1.0 - gate) * geo)

    class GatedSparseStem(spconv.SparseModule):
        def __init__(self, out_channels, norm_fn, negative_slope):
            super().__init__()
            self.fusion = GatedInputFusion(out_channels, negative_slope)
            self.conv = spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, indice_key="v400_stem", algo=algo)
            self.norm = norm_fn(out_channels)
            self.act = nn.LeakyReLU(negative_slope, inplace=False)

        def forward(self, x):
            x = sparse_replace(x, self.fusion(x.features))
            x = self.conv(x)
            x = sparse_replace(x, self.act(self.norm(x.features)))
            return x

    class SparseActNorm(spconv.SparseModule):
        def __init__(self, channels, norm_fn, negative_slope):
            super().__init__()
            self.norm = norm_fn(channels)
            self.act = nn.LeakyReLU(negative_slope, inplace=False)

        def forward(self, x):
            return sparse_replace(x, self.act(self.norm(x.features)))

    class MSTB(spconv.SparseModule):
        def __init__(self, channels, indice_key, norm_fn, negative_slope=0.01):
            super().__init__()
            self.local = spconv.SparseSequential(
                spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False, indice_key=indice_key + "_local", algo=algo),
                SparseActNorm(channels, norm_fn, negative_slope),
            )
            self.space = spconv.SparseSequential(
                spconv.SubMConv3d(channels, channels, (5, 1, 1), padding=(2, 0, 0), bias=False, indice_key=indice_key + "_space_x", algo=algo),
                SparseActNorm(channels, norm_fn, negative_slope),
                spconv.SubMConv3d(channels, channels, (1, 5, 1), padding=(0, 2, 0), bias=False, indice_key=indice_key + "_space_y", algo=algo),
                SparseActNorm(channels, norm_fn, negative_slope),
            )
            self.time1 = spconv.SubMConv3d(channels, channels, (1, 1, 5), padding=(0, 0, 2), dilation=(1, 1, 1), bias=False, indice_key=indice_key + "_time1", algo=algo)
            self.time2 = spconv.SubMConv3d(channels, channels, (1, 1, 5), padding=(0, 0, 4), dilation=(1, 1, 2), bias=False, indice_key=indice_key + "_time2", algo=algo)
            self.time4 = spconv.SubMConv3d(channels, channels, (1, 1, 5), padding=(0, 0, 8), dilation=(1, 1, 4), bias=False, indice_key=indice_key + "_time4", algo=algo)
            self.time_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.compress = spconv.SubMConv3d(channels * 3, channels, 1, bias=False, indice_key=indice_key + "_compress", algo=algo)
            self.norm = norm_fn(channels)
            self.act = nn.LeakyReLU(negative_slope, inplace=False)

        def temporal_weights(self):
            return torch.softmax(self.time_logits, dim=0)

        def forward(self, x):
            identity = x.features
            local = self.local(x)
            space = self.space(x)
            t1 = self.time1(x)
            t2 = self.time2(x)
            t4 = self.time4(x)
            weights = self.temporal_weights()
            time_features = weights[0] * t1.features + weights[1] * t2.features + weights[2] * t4.features
            merged = local.replace_feature(torch.cat([local.features, space.features, time_features], dim=1))
            out = self.compress(merged)
            out = sparse_replace(out, self.act(self.norm(out.features + identity)))
            return out

    class SparseDown(spconv.SparseModule):
        def __init__(self, in_channels, out_channels, stride, indice_key, norm_fn, negative_slope):
            super().__init__()
            self.seq = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False, indice_key=indice_key, algo=algo),
                SparseActNorm(out_channels, norm_fn, negative_slope),
            )

        def forward(self, x):
            return self.seq(x)

    class SparseUp(spconv.SparseModule):
        def __init__(self, in_channels, skip_channels, out_channels, indice_key, refine_key, norm_fn, negative_slope):
            super().__init__()
            self.inv = spconv.SparseInverseConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False, algo=algo)
            self.merge = spconv.SubMConv3d(out_channels + skip_channels, out_channels, 1, bias=False, indice_key=refine_key + "_merge", algo=algo)
            self.norm = norm_fn(out_channels)
            self.act = nn.LeakyReLU(negative_slope, inplace=False)
            self.refine = MSTB(out_channels, refine_key, norm_fn, negative_slope)

        def forward(self, x, skip):
            up = self.inv(x)
            up = sparse_replace(up, torch.cat([up.features, skip.features], dim=1))
            up = self.merge(up)
            up = sparse_replace(up, self.act(self.norm(up.features)))
            return self.refine(up)

    _MAX_BOTTLENECK_VOXELS_FOR_DIRECTION = 2000

    class BottleneckDirectionAggregation(nn.Module):
        """Direction-aware neighborhood feature aggregation for the Bottleneck layer.

        For each voxel i, finds its K nearest neighbors in 3D sparse loc space
        (batch, x, y, t).  Computes a PCA principal direction v_i from neighbor
        coordinate offsets.  Aggregation weights combine:

            a_ij = distance_weight * exp(kappa * cos(delta_q_ij, v_i))

        where distance_weight = exp(-||delta_q||^2 / (2 * sigma^2)).

        If the number of bottleneck voxels exceeds _MAX_BOTTLENECK_VOXELS_FOR_DIRECTION,
        the module falls back to identity pass-through and sets
        self.skipped = True so the caller can report it.
        """

        def __init__(self, channels, k_neighbors=8, sigma=4.0, kappa=1.0):
            super().__init__()
            self.channels = channels
            self.k = k_neighbors
            self.sigma = sigma
            self.kappa = kappa
            self.skipped = False

        def forward(self, sp_tensor):
            features = sp_tensor.features                 # [N, C]
            indices = sp_tensor.indices                   # [N, 4]  (batch, x, y, t)
            num_voxels = indices.shape[0]

            if num_voxels < 2 or num_voxels > _MAX_BOTTLENECK_VOXELS_FOR_DIRECTION:
                if num_voxels > _MAX_BOTTLENECK_VOXELS_FOR_DIRECTION:
                    self.skipped = True
                return sp_tensor

            # coords: [N, 3] in (x, y, t) scaled roughly to unit range
            coords = indices[:, 1:].float()               # [N, 3]
            coord_range = coords.max(dim=0).values - coords.min(dim=0).values + 1e-6
            coords_norm = coords / coord_range.unsqueeze(0)  # [N, 3]

            k_eff = min(self.k, num_voxels)
            dist_sq = torch.cdist(coords_norm, coords_norm, p=2).pow(2)  # [N, N]
            _, nn_idx = torch.topk(dist_sq, k=k_eff, dim=1, largest=False)  # [N, K]

            # --- compute PCA principal direction v_i for every voxel ---
            principal_dirs = torch.zeros(num_voxels, 3, device=features.device, dtype=features.dtype)
            for i in range(num_voxels):
                nbr = coords_norm[nn_idx[i]]                         # [K, 3]
                if nbr.shape[0] < 3:
                    continue
                centered = nbr - nbr.mean(dim=0, keepdims=True)      # [K, 3]
                try:
                    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
                    principal_dirs[i] = Vh[0]                         # first principal direction
                except Exception:
                    continue

            # --- aggregation weights ---
            delta_q = coords_norm.unsqueeze(1) - coords_norm[nn_idx]  # [N, K, 3]
            delta_q_norm_sq = (delta_q ** 2).sum(dim=2)               # [N, K]
            dist_weight = torch.exp(-delta_q_norm_sq / (2.0 * self.sigma ** 2))  # [N, K]

            v_i_expanded = principal_dirs.unsqueeze(1)                          # [N, 1, 3]
            delta_q_norm_val = torch.norm(delta_q, dim=2).clamp_min(1e-8)       # [N, K]
            cos_dir = (delta_q * v_i_expanded).sum(dim=2) / delta_q_norm_val   # [N, K]

            a_ij_dir = dist_weight * torch.exp(self.kappa * cos_dir)            # [N, K]
            a_sum = a_ij_dir.sum(dim=1, keepdim=True).clamp_min(1e-8)          # [N, 1]
            a_ij_norm = a_ij_dir / a_sum                                        # [N, K]

            neighbor_feats = features[nn_idx]                                   # [N, K, C]
            agg_feats = (a_ij_norm.unsqueeze(2) * neighbor_feats).sum(dim=1)   # [N, C]

            out_feats = features + agg_feats
            return sparse_replace(sp_tensor, out_feats)

    class BottleneckTokenTransformer(nn.Module):
        def __init__(self, channels, token_limit=512, num_heads=4, ff_multiplier=2, dropout=0.0):
            super().__init__()
            self.channels = channels
            self.token_limit = int(token_limit)
            self.coord_proj = nn.Linear(3, channels)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=int(num_heads),
                dim_feedforward=channels * int(ff_multiplier),
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.out_norm = nn.LayerNorm(channels)
            self.gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
            self.last_report = {}

        def select_token_indices(self, num_voxels, device):
            limit = max(1, self.token_limit)
            if num_voxels <= limit:
                return torch.arange(num_voxels, device=device, dtype=torch.long)
            return torch.linspace(0, num_voxels - 1, steps=limit, device=device).round().long().unique()

        def forward(self, sp_tensor):
            features = sp_tensor.features
            orig_dtype = features.dtype
            # TransformerEncoder requires float32; cast for the transformer path
            features_f32 = features.float()
            indices = sp_tensor.indices
            num_voxels = int(features.shape[0])
            self.last_report = {
                "input_voxels": num_voxels,
                "token_limit": int(self.token_limit),
                "status": "skipped_empty" if num_voxels == 0 else "ok",
            }
            if num_voxels == 0:
                return sp_tensor
            token_idx = self.select_token_indices(num_voxels, features.device)
            token_features = features_f32[token_idx]
            coords = indices[token_idx, 1:].float()
            coord_min = coords.min(dim=0, keepdim=True).values
            coord_span = (coords.max(dim=0, keepdim=True).values - coord_min).clamp_min(1.0)
            coords_norm = (coords - coord_min) / coord_span
            tokens = token_features + self.coord_proj(coords_norm)
            # Chunked encoder to avoid OOM with full token set (4488² attention)
            from torch.utils.checkpoint import checkpoint as _ckpt
            chunk_size = 1500
            n_tokens = tokens.size(0)
            encoded_chunks = []
            for c_start in range(0, n_tokens, chunk_size):
                c_end = min(c_start + chunk_size, n_tokens)
                chunk = tokens[c_start:c_end]
                enc = _ckpt(lambda t: self.encoder(t.unsqueeze(0)).squeeze(0), chunk)
                encoded_chunks.append(enc)
            encoded = torch.cat(encoded_chunks, dim=0)
            del encoded_chunks
            encoded = self.out_norm(encoded)
            gate = self.gate(torch.cat([token_features, encoded], dim=1))
            fused_tokens = token_features + gate * encoded
            out_features = features_f32.clone()
            out_features[token_idx] = fused_tokens
            self.last_report = {
                "status": "ok",
                "input_voxels": num_voxels,
                "tokens": int(token_idx.numel()),
                "token_limit": int(self.token_limit),
                "tokenized_full_input": bool(token_idx.numel() == num_voxels),
                "residual_reinjection": True,
                "features_finite_before": bool(torch.isfinite(features).all().detach().cpu().item()),
                "tokens_finite": bool(torch.isfinite(tokens).all().detach().cpu().item()),
                "encoded_finite": bool(torch.isfinite(encoded).all().detach().cpu().item()),
                "gate_finite": bool(torch.isfinite(gate).all().detach().cpu().item()),
                "features_finite_after": bool(torch.isfinite(out_features).all().detach().cpu().item()),
            }
            # Cast back to original dtype if necessary
            if orig_dtype != out_features.dtype:
                out_features = out_features.to(orig_dtype)
            return sparse_replace(sp_tensor, out_features)

    class DirectionPredictionHead(nn.Module):
        """Trajectory direction prediction auxiliary head.

        Takes bottleneck features and predicts a normalized 3D motion direction
        vector for each voxel.  This head is train-only and not used during
        inference/submission.
        """

        def __init__(self, in_channels, hidden=128, negative_slope=0.01):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(negative_slope, inplace=False),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(negative_slope, inplace=False),
                nn.Linear(hidden, 3),
            )

        def forward(self, features):
            v_raw = self.mlp(features)                      # [N, 3]
            v_norm = torch.norm(v_raw, dim=1, keepdim=True).clamp_min(1e-8)
            return v_raw / v_norm                            # [N, 3], L2-normalized

    class TrajectoryAwareSparseUNet(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            c1, c2, c3, c4 = 24, 40, 64, 96
            negative_slope = float(getattr(cfg, "negative_slope", 0.01))
            norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
            self.no_patchattention = True
            self.input_channel = 8
            self.channel_schedule = [c1, c2, c3, c4]
            self.downsample_strides = [[2, 2, 2], [2, 2, 2], [2, 2, 4]]
            self.stem = GatedSparseStem(c1, norm_fn, negative_slope)
            self.stage1 = spconv.SparseSequential(MSTB(c1, "v400_s1a", norm_fn, negative_slope), MSTB(c1, "v400_s1b", norm_fn, negative_slope))
            self.down1 = SparseDown(c1, c2, [2, 2, 2], "v400_down1", norm_fn, negative_slope)
            self.stage2 = spconv.SparseSequential(MSTB(c2, "v400_s2a", norm_fn, negative_slope), MSTB(c2, "v400_s2b", norm_fn, negative_slope))
            self.down2 = SparseDown(c2, c3, [2, 2, 2], "v400_down2", norm_fn, negative_slope)
            self.stage3 = spconv.SparseSequential(MSTB(c3, "v400_s3a", norm_fn, negative_slope), MSTB(c3, "v400_s3b", norm_fn, negative_slope))
            self.down3 = SparseDown(c3, c4, [2, 2, 4], "v400_down3", norm_fn, negative_slope)
            self.bottleneck_mstb = MSTB(c4, "v400_bottleneck", norm_fn, negative_slope)
            self.bottleneck_transformer = BottleneckTokenTransformer(
                c4,
                token_limit=getattr(cfg, "transformer_token_limit", TRANSFORMER_TOKEN_LIMIT),
                num_heads=getattr(cfg, "transformer_num_heads", TRANSFORMER_NUM_HEADS),
                ff_multiplier=getattr(cfg, "transformer_ff_multiplier", TRANSFORMER_FF_MULTIPLIER),
                dropout=getattr(cfg, "transformer_dropout", TRANSFORMER_DROPOUT),
            )
            self.bottleneck_dir_agg = BottleneckDirectionAggregation(c4, k_neighbors=8, sigma=4.0, kappa=1.0)
            self.bottleneck_dir_applied = False
            self.transformer_applied = False
            # Direction prediction auxiliary head (train-only)
            self.direction_head = DirectionPredictionHead(c4, hidden=DIRECTION_HEAD_HIDDEN, negative_slope=negative_slope)
            self.decoder3 = SparseUp(c4, c3, c3, "v400_down3", "v400_dec3", norm_fn, negative_slope)
            self.decoder2 = SparseUp(c3, c2, c2, "v400_down2", "v400_dec2", norm_fn, negative_slope)
            self.decoder1 = SparseUp(c2, c1, c1, "v400_down1", "v400_dec1", norm_fn, negative_slope)
            self.semantic_linear = nn.Sequential(nn.Linear(c1, 1), nn.Sigmoid())

        def temporal_weight_sums(self):
            return {name: float(module.temporal_weights().detach().cpu().sum().item()) for name, module in self.named_modules() if hasattr(module, "temporal_weights")}

        def forward(self, input, return_direction=False):
            x0 = self.stem(input)
            x1 = self.stage1(x0)
            x2 = self.stage2(self.down1(x1))
            x3 = self.stage3(self.down2(x2))
            xb = self.bottleneck_mstb(self.down3(x3))
            xb = self.bottleneck_transformer(xb)
            self.transformer_applied = True
            xb = self.bottleneck_dir_agg(xb)
            if not hasattr(self, "bottleneck_dir_applied") or not self.bottleneck_dir_applied:
                self.bottleneck_dir_applied = True
            # Direction prediction from bottleneck features
            if return_direction:
                direction_pred = self.direction_head(xb.features)
            y3 = self.decoder3(xb, x3)
            y2 = self.decoder2(y3, x2)
            y1 = self.decoder1(y2, x1)
            output = self.semantic_linear(y1.features)
            voxel = y1.replace_feature(output)
            if return_direction:
                return output, voxel, direction_pred, xb.indices
            return output, voxel

    return TrajectoryAwareSparseUNet


def build_model(cfg, train=False, weight_path=None, use_cuda=True, return_model_cls=False):
    model_cls = build_model_classes()
    if return_model_cls:
        return model_cls, model_cls
    net = model_cls(cfg)
    net = net.train() if train else net.eval()
    if use_cuda:
        net.cuda()
    if weight_path is not None:
        map_location = "cuda:0" if use_cuda else "cpu"
        net.load_state_dict(torch.load(weight_path, map_location=map_location))
    return net


class V4EvUAV(torch.utils.data.Dataset if torch is not None else object):
    def __init__(self, configs, mode="train", use_cache=True):
        self.configs = configs
        self.mode = mode
        self.root = Path(configs.root) / mode
        self.file_list = sorted(os.listdir(self.root)) if self.root.exists() else []
        self.use_cache = use_cache

    def __getitem__(self, idx):
        path = self.root / self.file_list[idx]
        with np.load(path, allow_pickle=False) as events:
            evs_norm = events["evs_norm"].astype(np.float32)
            ev_loc = events["ev_loc"].astype(np.int64)
            seg_label = evs_norm[:, 4].astype(np.float32)
            idx_label = evs_norm[:, 5]
            if "ev" in events.files:
                ev = events["ev"]
                raw_event = np.column_stack([ev["x"], ev["y"], ev["t"], ev["p"]])
            else:
                raw_event = evs_norm[:, 0:4]
        features = load_or_compute_features(path, self.configs, self.use_cache)
        if features.shape[0] != evs_norm.shape[0]:
            raise ValueError(f"{path.name}: feature rows {features.shape[0]} != event rows {evs_norm.shape[0]}")
        evs_feature = np.concatenate([evs_norm[:, 0:4], features], axis=1).astype(np.float32)
        stats = load_input_stats(self.configs)
        evs_feature = normalize_input_features(evs_feature, stats, self.configs)
        if self.mode == "train":
            num_events = ev_loc.shape[0]
            max_events = int(getattr(self.configs, "max_events_num", 50000))
            if num_events >= max_events:
                sample_idx = np.random.choice(num_events, max_events, replace=False)
                ev_loc = ev_loc[sample_idx]
                evs_feature = evs_feature[sample_idx]
                seg_label = seg_label[sample_idx]
                idx_label = idx_label[sample_idx]
                raw_event = raw_event[sample_idx]
        return {"ev_loc": ev_loc, "evs_feature": evs_feature, "seg_label": seg_label, "idx": idx_label, "raw_event": raw_event, "file_name": self.file_list[idx], "source_path": str(path)}

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
        raw_event_batches = []
        meta = []
        for i, ev in enumerate(batch):
            ev_loc = ev["ev_loc"]
            loc_batches.append(np.hstack((i * np.ones((ev_loc.shape[0], 1)), ev_loc)))
            feature_batches.append(ev["evs_feature"])
            seg_label_batches.append(ev["seg_label"])
            idx_label_batches.append(ev["idx"])
            raw_event_batches.append(ev["raw_event"])
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
            "raw_event": np.concatenate(raw_event_batches, axis=0),
            "meta": meta,
        }


def make_dataloader(cfg, mode, shuffle=False):
    dataset = V4EvUAV(cfg, mode=mode, use_cache=True)
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


def normalize_inference_chunk_size(chunk_size):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("inference chunk size must be positive")
    return chunk_size


def predict_sample_chunked(net, dataset, sample_index, chunk_size):
    sample = dataset[sample_index]
    total_events = int(sample["ev_loc"].shape[0])
    pred_chunks = []
    loc_chunks = []
    for start in range(0, total_events, chunk_size):
        end = min(start + chunk_size, total_events)
        chunk = {
            "ev_loc": sample["ev_loc"][start:end],
            "evs_feature": sample["evs_feature"][start:end],
            "seg_label": sample["seg_label"][start:end],
            "idx": sample["idx"][start:end],
            "raw_event": sample["raw_event"][start:end],
            "file_name": sample["file_name"],
            "source_path": sample["source_path"],
        }
        ev = dataset.custom_collate([chunk])
        p2v_map = ev["p2v_map"].long().cuda()
        with torch.cuda.amp.autocast():
            preds, _ = net(ev["voxel_ev"])
        pred_chunk = preds[p2v_map].reshape(-1).cpu()
        if pred_chunk.numel() != end - start:
            raise ValueError(f"{sample['file_name']}: chunk prediction count {pred_chunk.numel()} != event count {end - start}")
        pred_chunks.append(pred_chunk)
        loc_chunks.append(ev["locs"].cpu())
        torch.cuda.empty_cache()
    preds = torch.cat(pred_chunks, dim=0) if pred_chunks else torch.empty(0)
    locs = torch.cat(loc_chunks, dim=0) if loc_chunks else torch.empty((0, 4), dtype=torch.int64)
    if preds.numel() != total_events:
        raise ValueError(f"{sample['file_name']}: prediction count {preds.numel()} != original event count {total_events}")
    return {
        "sample": sample_index,
        "file_name": sample["file_name"],
        "source_path": sample["source_path"],
        "preds": preds.clone(),
        "labels": torch.from_numpy(sample["seg_label"]).float().clone(),
        "idx": sample["idx"],
        "locs": locs.clone(),
        "raw_event": sample["raw_event"],
        "inference_chunk_size": chunk_size,
        "original_event_count": total_events,
    }


def collect_predictions(net, cfg, mode="val", inference_chunk_size=50000):
    dataset = V4EvUAV(cfg, mode=mode, use_cache=True)
    chunk_size = normalize_inference_chunk_size(inference_chunk_size)
    rows = []
    with torch.no_grad():
        for sample_index in tqdm.tqdm(range(len(dataset)), desc=f"predict-{mode}", unit="sample"):
            rows.append(predict_sample_chunked(net, dataset, sample_index, chunk_size))
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


def evaluate_model(net, cfg, threshold, inference_chunk_size=50000):
    was_training = net.training
    net.eval()
    # Direction head not used during evaluation
    rows = collect_predictions(net, cfg, mode="val", inference_chunk_size=inference_chunk_size)
    metrics = evaluate_rows(rows, threshold=threshold)
    if was_training:
        net.train()
    return metrics


def trajectory_projection_dice_loss(event_probs, labels, locs, smooth=1.0):
    xy_keys = locs[:, [0, 1, 2]].long()
    unique_xy, inverse = torch.unique(xy_keys, dim=0, return_inverse=True)
    flat_probs = event_probs.reshape(-1)
    flat_labels = labels.reshape(-1).to(dtype=event_probs.dtype)
    if unique_xy.shape[0] == 0:
        return flat_probs.sum() * 0.0
    group_count = unique_xy.shape[0]
    projection_pred = flat_probs.new_zeros(group_count)
    projection_true = flat_probs.new_zeros(group_count)
    counts = flat_probs.new_zeros(group_count)
    projection_pred.index_add_(0, inverse, flat_probs)
    projection_true.index_add_(0, inverse, flat_labels)
    counts.index_add_(0, inverse, torch.ones_like(flat_probs))
    projection_pred = projection_pred / counts.clamp_min(1.0)
    projection_true = projection_true / counts.clamp_min(1.0)
    intersection = (projection_pred * projection_true).sum()
    denominator = projection_pred.sum() + projection_true.sum()
    return 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)


def direction_abs_cosine_loss(v_hat, v_gt, fg_mask):
    """Direction alignment loss: L = 1 - |v_hat^T · v_gt|

    Computed only on foreground voxels indicated by fg_mask.
    The absolute value ensures that opposite directions are treated as
    equivalent (undirected line alignment).

    Args:
        v_hat: [N, 3] predicted normalized direction vectors
        v_gt:  [N, 3] ground-truth PCA principal direction
        fg_mask: [N] bool, True for foreground voxels

    Returns:
        scalar loss; 0.0 if no foreground voxels
    """
    if not fg_mask.any():
        return v_hat.new_zeros(())
    v_hat_fg = v_hat[fg_mask]        # [M, 3]
    v_gt_fg = v_gt[fg_mask]          # [M, 3]
    cos_abs = torch.abs((v_hat_fg * v_gt_fg).sum(dim=1))  # [M]
    return (1.0 - cos_abs).mean()


def compute_direction_gt(bottleneck_indices, event_locs, event_labels, p2v_map, neighbor_k=16, max_direction_voxels=1024):
    """Compute PCA-based ground-truth direction for bottleneck voxels (sampled).

    Maps event-level labels to bottleneck voxels to identify foreground,
    then randomly samples at most ``max_direction_voxels`` voxels (fg-first)
    and computes PCA first principal direction from neighbor voxel coords
    ONLY on the sampled subset.  cdist/KNN/SVD run on CPU.

    Args:
        bottleneck_indices: [N, 4] bottleneck voxel indices (batch, x, y, t)
        event_locs:      [E, 4] event sparse locations (batch, x, y, t)
        event_labels:    [E]    binary labels (0/1)
        p2v_map:         [E]    point-to-voxel index mapping (stem-level, unused)
        neighbor_k:      int    K neighbors for PCA (default 16)
        max_direction_voxels:  int  max voxels to sample for direction GT (default 1024)

    Returns:
        v_gt:    [N, 3] PCA principal direction (zero for non-sampled voxels)
        fg_mask: [N]    bool, True only for sampled foreground voxels
    """
    device = bottleneck_indices.device

    # --- Move inputs to CPU immediately ---
    bn_indices_cpu = bottleneck_indices.cpu()
    event_locs_cpu = event_locs.cpu()
    event_labels_cpu = event_labels.cpu()
    N = bn_indices_cpu.shape[0]

    if N == 0:
        return (torch.zeros(0, 3, device=device),
                torch.zeros(0, dtype=torch.bool, device=device))

    # --- Step 1: Determine full foreground bottleneck voxels (on CPU) ---
    total_stride = torch.tensor([8, 8, 32], dtype=torch.int64)  # CPU
    event_bn_coords = torch.div(event_locs_cpu[:, 1:].long(), total_stride, rounding_mode='trunc')  # [E, 3]
    bn_coords = bn_indices_cpu[:, 1:]  # [N, 3], already at bottleneck resolution

    # Hash map: (x, y, t) → bottleneck voxel index
    bn_key_to_idx = {}
    for i in range(N):
        key = (int(bn_coords[i, 0]), int(bn_coords[i, 1]), int(bn_coords[i, 2]))
        bn_key_to_idx[key] = i

    fg_mask_full = torch.zeros(N, dtype=torch.bool)  # CPU
    fg_events = event_labels_cpu.bool()
    if fg_events.any():
        fg_event_coords = event_bn_coords[fg_events]  # [F, 3]
        for j in range(fg_event_coords.shape[0]):
            key = (int(fg_event_coords[j, 0]), int(fg_event_coords[j, 1]), int(fg_event_coords[j, 2]))
            if key in bn_key_to_idx:
                fg_mask_full[bn_key_to_idx[key]] = True

    # --- Step 2: Sample voxels for direction computation (fg-priority) ---
    fg_voxel_indices = torch.where(fg_mask_full)[0]  # CPU
    bg_voxel_indices = torch.where(~fg_mask_full)[0]  # CPU
    n_fg = fg_voxel_indices.shape[0]
    n_bg = bg_voxel_indices.shape[0]
    max_vox = max(1, max_direction_voxels)

    sampled_idx_list = []
    if n_fg >= max_vox:
        perm = torch.randperm(n_fg)[:max_vox]
        sampled_idx_list = fg_voxel_indices[perm]
    else:
        # Take all fg voxels
        sampled_idx_list = list(fg_voxel_indices.numpy())
        remaining = max_vox - n_fg
        if n_bg > 0 and remaining > 0:
            n_bg_sample = min(remaining, n_bg)
            perm = torch.randperm(n_bg)[:n_bg_sample]
            sampled_idx_list.extend(bg_voxel_indices[perm].numpy().tolist())
        sampled_idx_list = torch.tensor(sampled_idx_list, dtype=torch.long)

    if sampled_idx_list.numel() == 0:
        return (torch.zeros(N, 3, device=device),
                torch.zeros(N, dtype=torch.bool, device=device))

    # --- Step 3: Build fg_mask — True only for sampled foreground voxels ---
    fg_mask = torch.zeros(N, dtype=torch.bool)  # CPU
    fg_set = set(fg_voxel_indices.numpy().tolist())
    sampled_fg = [int(idx) for idx in sampled_idx_list.numpy() if int(idx) in fg_set]
    if sampled_fg:
        fg_mask[torch.tensor(sampled_fg, dtype=torch.long)] = True

    # --- Step 4: PCA direction only on sampled voxel subset (on CPU) ---
    M = sampled_idx_list.shape[0]
    sampled_coords = bn_coords[sampled_idx_list].float()  # [M, 3], CPU
    coord_range = sampled_coords.max(dim=0).values - sampled_coords.min(dim=0).values + 1e-6
    sampled_coords_norm = sampled_coords / coord_range.unsqueeze(0)  # [M, 3], CPU

    v_gt_sampled = torch.zeros(M, 3, dtype=sampled_coords_norm.dtype)  # CPU

    if M >= 2:
        k_eff = min(neighbor_k, M)
        # cdist only on the M×M sampled subset — no chunking needed (M ≤ max_direction_voxels)
        dist_sq = torch.cdist(sampled_coords_norm, sampled_coords_norm, p=2).pow(2)  # [M, M], CPU
        _, nn_idx = torch.topk(dist_sq, k=k_eff, dim=1, largest=False)  # [M, K], CPU

        for i in range(M):
            nbr = sampled_coords_norm[nn_idx[i]]  # [K, 3], CPU
            if nbr.shape[0] < 3:
                continue
            centered = nbr - nbr.mean(dim=0, keepdims=True)  # CPU
            try:
                _, _, Vh = torch.linalg.svd(centered, full_matrices=False)  # CPU
                v_gt_sampled[i] = Vh[0]  # first principal direction
            except Exception:
                continue

    # --- Step 5: Scatter back to full N-sized output ---
    v_gt = torch.zeros(N, 3, dtype=v_gt_sampled.dtype)  # CPU
    v_gt[sampled_idx_list] = v_gt_sampled

    # --- Move results back to GPU ---
    return v_gt.to(device), fg_mask.to(device)


def train_one(args):
    from utils.stcloss import STCLoss
    cfg, _ = load_yaml_config(args.config)
    cfg.model_name = EXPERIMENT_NAME
    cfg.root = str(args.data_root)
    cfg.epochs = args.epochs
    cfg.max_events_num = 50000
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.optim = "Adam"
    cfg.model_save_root = str(MODELS_DIR)
    cfg.seed = SEED
    cfg.model_path = str(MODELS_DIR / "best_iou_seed316.pt")
    cfg.input_channel = 8
    sync_global_cfg(cfg)
    write_yaml_config(DEFAULT_V4_CONFIG, cfg)
    setup(SEED)
    net = build_model(cfg, train=True)
    _, train_loader = make_dataloader(cfg, mode="train", shuffle=True)
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float("inf")
    best_iou = -1.0
    loss_rows = []
    epoch_rows = []
    for epoch in range(int(cfg.epochs)):
        epoch_losses = []
        pbar = tqdm.tqdm(train_loader, desc=f"v620-epoch-{epoch}", unit="batch")
        for batch_index, ev in enumerate(pbar):
            label = ev["seg_label"].float().cuda()
            p2v_map = ev["p2v_map"].long().cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds, voxel, direction_pred, bottleneck_indices = net(ev["voxel_ev"], return_direction=True)
                main_loss = criterion(voxel, p2v_map, preds, label)
                event_probs = preds[p2v_map].reshape(-1)
                traj_loss = trajectory_projection_dice_loss(event_probs, label, ev["locs"].cuda())
                # Direction loss
                v_gt, fg_mask = compute_direction_gt(
                    bottleneck_indices, ev["locs"].cuda(), label,
                    p2v_map, neighbor_k=DIRECTION_NEIGHBOR_K,
                    max_direction_voxels=MAX_DIRECTION_VOXELS,
                )
                dir_loss = direction_abs_cosine_loss(direction_pred, v_gt, fg_mask)
                loss = main_loss + TRAJECTORY_PROJECTION_DICE_WEIGHT * traj_loss + TRAJECTORY_DIRECTION_WEIGHT * dir_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.item())
            main_value = float(main_loss.item())
            traj_value = float(traj_loss.item())
            dir_value = float(dir_loss.item())
            epoch_losses.append(value)
            loss_rows.append({"epoch": epoch, "batch": batch_index, "loss": value, "main_loss": main_value, "trajectory_projection_dice_loss": traj_value, "trajectory_direction_loss": dir_value, "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "lr": optimizer.param_groups[0]["lr"]})
            if value < best_loss:
                best_loss = value
                torch.save(net.state_dict(), MODELS_DIR / "best_loss_seed316.pt")
            pbar.set_postfix(loss=value, main=main_value, traj_dice=traj_value, dir_loss=dir_value)
            torch.cuda.empty_cache()
        scheduler.step()
        row = {"epoch": epoch, "mean_loss": float(np.mean(epoch_losses)), "min_loss": float(np.min(epoch_losses)), "max_loss": float(np.max(epoch_losses)), "best_loss": best_loss, "val_iou": "", "val_seg_acc": "", "val_pd": "", "val_fa": ""}
        if epoch >= max(0, int(cfg.epochs) - int(args.eval_last_epochs)):
            metrics = evaluate_model(net, cfg, args.threshold)
            row.update({"val_iou": metrics["iou"], "val_seg_acc": metrics["seg_acc"], "val_pd": metrics["pd"], "val_fa": metrics["fa"]})
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                torch.save(net.state_dict(), MODELS_DIR / "best_iou_seed316.pt")
        epoch_rows.append(row)
        write_csv(RESULTS_DIR / "v620_loss_records.csv", loss_rows)
        write_csv(RESULTS_DIR / "v620_epoch_summary.csv", epoch_rows)
    dir_agg_skipped = getattr(net, "bottleneck_dir_agg", None) is not None and net.bottleneck_dir_agg.skipped
    summary = {"status": "ok", "version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "epochs": int(cfg.epochs), "best_loss": best_loss, "best_iou": best_iou, "model_path": str(MODELS_DIR / "best_iou_seed316.pt"), "config": str(DEFAULT_V4_CONFIG), "input_norm_stats_path": str(input_stats_path(cfg)), "input_norm_epsilon": float(getattr(cfg, "input_norm_epsilon", 1e-6)), "input_norm_clip": bool(getattr(cfg, "input_norm_clip", False)), "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_projection_dice_mode": TRAJECTORY_PROJECTION_DICE_MODE, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "auxiliary_supervision": TRAJECTORY_PROJECTION_DICE_MODE + "+direction_head", "inference_uses_auxiliary_head": False, "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": dir_agg_skipped, "trajectory_direction_head": True, "transformer_enabled": True, "transformer_insert_position": "after_bottleneck_mstb", "transformer_token_limit": TRANSFORMER_TOKEN_LIMIT}
    with open(RESULTS_DIR / "v620_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2))


def command_train(args):
    ensure_dirs()
    train_one(args)


def command_evaluate(args):
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    sync_global_cfg(cfg)
    setup(SEED)
    net = build_model(cfg, train=False, weight_path=args.weight)
    chunk_size = normalize_inference_chunk_size(args.inference_chunk_size)
    rows = collect_predictions(net, cfg, mode="val", inference_chunk_size=chunk_size)
    metrics = evaluate_rows(rows, threshold=args.threshold)
    dir_agg_skipped = getattr(net, "bottleneck_dir_agg", None) is not None and net.bottleneck_dir_agg.skipped
    metrics.update({"version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "inference_chunk_size": chunk_size, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "no_patchattention": True, "inference_uses_auxiliary_head": False, "trajectory_direction_head": True, "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": dir_agg_skipped, "transformer_enabled": True, "transformer_insert_position": "after_bottleneck_mstb", "transformer_token_limit": TRANSFORMER_TOKEN_LIMIT})
    with open(RESULTS_DIR / "v620_validation_metrics.json", "w", encoding="utf-8") as f:
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
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    sync_global_cfg(cfg)
    setup(SEED)
    net = build_model(cfg, train=False, weight_path=args.weight)
    chunk_size = normalize_inference_chunk_size(args.inference_chunk_size)
    rows = collect_predictions(net, cfg, mode="val", inference_chunk_size=chunk_size)
    name = args.name
    txt_dir = SUBMISSION_DIR / name / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    for item in rows:
        pred = (item["preds"].numpy() >= args.threshold).astype(np.uint8)
        save_prediction(item["source_path"], txt_dir / f"{Path(item['file_name']).stem}.txt", pred)
    zip_path = SUBMISSION_DIR / f"{name}.zip"
    write_zip(txt_dir, zip_path)
    print(json.dumps({"txt_count": len(list(txt_dir.glob('*.txt'))), "zip": str(zip_path), "inference_chunk_size": chunk_size}, indent=2))


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
    report = {"status": "ok" if not errors else "failed", "version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "txt_dir": str(txt_dir), "zip": str(zip_path), "txt_count": len(txt_files), "errors": errors}
    with open(RESULTS_DIR / "v620_submission_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def command_threshold_scan(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    cfg.model_path = str(args.weight)
    sync_global_cfg(cfg)
    setup(SEED)
    net = build_model(cfg, train=False, weight_path=args.weight)
    chunk_size = normalize_inference_chunk_size(args.inference_chunk_size)
    rows = collect_predictions(net, cfg, mode="val", inference_chunk_size=chunk_size)
    scan_rows = []
    thresholds = [float(x) for x in args.thresholds]
    for threshold in thresholds:
        metrics = evaluate_rows(rows, threshold=threshold)
        metrics.update({"version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "inference_chunk_size": chunk_size, "input_channel": 8, "optimizer": "Adam", "no_patchattention": True, "trajectory_direction_head": True, "bottleneck_direction_aggregation": True, "inference_uses_auxiliary_head": False, "input_norm_stats_path": str(input_stats_path(cfg)), "input_norm_epsilon": float(getattr(cfg, "input_norm_epsilon", 1e-6)), "input_norm_clip": bool(getattr(cfg, "input_norm_clip", False)), "transformer_enabled": True, "transformer_insert_position": "after_bottleneck_mstb", "transformer_token_limit": TRANSFORMER_TOKEN_LIMIT})
        scan_rows.append(metrics)
    best_iou = max(scan_rows, key=lambda r: (r["iou"], -r["fa"], r["pd"]))
    lowest_fa = min(scan_rows, key=lambda r: (r["fa"], -r["iou"], -r["pd"]))
    best_pd = max(scan_rows, key=lambda r: (r["pd"], r["iou"], -r["fa"]))
    baseline = min(scan_rows, key=lambda r: abs(float(r["threshold"]) - 0.90))
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
    fieldnames = ["threshold", "iou", "seg_acc", "pd", "fa", "tp", "fp", "fn", "pred_positive", "gt_positive", "total_events", "tags", "version", "experiment", "seed", "max_events_num", "inference_chunk_size", "input_channel", "optimizer", "no_patchattention", "trajectory_direction_head", "bottleneck_direction_aggregation", "inference_uses_auxiliary_head", "input_norm_stats_path", "input_norm_epsilon", "input_norm_clip", "transformer_enabled", "transformer_insert_position", "transformer_token_limit"]
    write_csv(RESULTS_DIR / THRESHOLD_SCAN_CSV, [{k: row.get(k, "") for k in fieldnames} for row in scan_rows])
    write_csv(CHARTS_DIR / THRESHOLD_SCAN_CSV, [{k: row.get(k, "") for k in fieldnames} for row in scan_rows])
    summary = {"status": "ok", "version": VERSION, "thresholds": thresholds, "best_iou": best_iou, "lowest_fa": lowest_fa, "best_pd": best_pd, "recommended": recommended, "baseline_threshold_0p90": baseline, "generated_at": date.today().isoformat()}
    for path in (RESULTS_DIR / THRESHOLD_SCAN_JSON, CHARTS_DIR / THRESHOLD_SCAN_JSON):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
    write_threshold_markdown(scan_rows, summary)
    write_threshold_svg(scan_rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def write_threshold_markdown(rows, summary):
    lines = [
        "# v6.2.0 full-token 阈值扫描",
        "",
        "说明：IoU、seg_acc、PD 越高越好，FA 越低越好。推理使用 best-IoU 权重和 chunked inference。",
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
    lines += ["", "## 结论", "", f"- best-IoU: threshold={best['threshold']:.2f}, IoU={best['iou']:.6f}, FA={best['fa']:.9f}。", f"- lowest-FA: threshold={low['threshold']:.2f}, IoU={low['iou']:.6f}, FA={low['fa']:.9f}。", f"- best-PD: threshold={bpd['threshold']:.2f}, PD={bpd['pd']:.6f}, IoU={bpd['iou']:.6f}。", f"- 综合推荐阈值: threshold={rec['threshold']:.2f}, IoU={rec['iou']:.6f}, seg_acc={rec['seg_acc']:.6f}, PD={rec['pd']:.6f}, FA={rec['fa']:.9f}。"]
    (CHARTS_DIR / THRESHOLD_SCAN_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">v6.2.0 Full Token Scan</text>')
    svg.append(f'<text x="{pad_l+10}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#1f77b4">IoU</text>')
    svg.append(f'<text x="{pad_l+60}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#d62728">FA</text>')
    svg.append(f'<text x="{pad_l+110}" y="{pad_t+18}" font-size="13" font-family="Arial" fill="#2ca02c">recommended={rec_t:.2f}</text>')
    svg.append('</svg>')
    (CHARTS_DIR / "v620_threshold_scan.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def command_build_comparison(args):
    ensure_dirs()
    scan_summary = load_json(RESULTS_DIR / THRESHOLD_SCAN_JSON)
    validation = load_json(RESULTS_DIR / SUBMISSION_VALIDATION_FILENAME) if (RESULTS_DIR / SUBMISSION_VALIDATION_FILENAME).exists() else {}
    rec = scan_summary["recommended"]
    best = scan_summary["best_iou"]
    rows = [
        {"version": "v6.2.0", "work_point": f"recommended@{rec['threshold']:.2f}", "label": "v6.2.0 full-token", "source_version": "v6.2.0", "threshold": rec["threshold"], "iou": rec["iou"], "seg_acc": rec["seg_acc"], "pd": rec["pd"], "fa": rec["fa"], "tp": rec["tp"], "fp": rec["fp"], "fn": rec["fn"], "pred_positive": rec["pred_positive"], "gt_positive": rec["gt_positive"], "total_events": rec["total_events"], "max_events_num": 50000, "input_channel": 8, "optimizer": "Adam", "epochs": 50, "submission_validation_status": validation.get("status", ""), "submission_txt_count": validation.get("txt_count", ""), "notes": "Full token (8192, no subsampling). v6.1.0 + full bottleneck attention."},
        {"version": "v6.2.0", "work_point": f"best-IoU@{best['threshold']:.2f}", "label": "v6.2.0 best-IoU threshold", "source_version": "v6.2.0", "threshold": best["threshold"], "iou": best["iou"], "seg_acc": best["seg_acc"], "pd": best["pd"], "fa": best["fa"], "tp": best["tp"], "fp": best["fp"], "fn": best["fn"], "pred_positive": best["pred_positive"], "gt_positive": best["gt_positive"], "total_events": best["total_events"], "max_events_num": 50000, "input_channel": 8, "optimizer": "Adam", "epochs": 50, "submission_validation_status": "not_submitted", "submission_txt_count": "", "notes": "Highest IoU point from v6.2.0 threshold scan."},
    ]
    v450_source = EXPERIMENT_ROOT.parent / "v4.5.0-input-normalization" / "charts" / "v450_v434_v433_v420_v313_v210_comparison.json"
    if v450_source.exists():
        for row in load_json(v450_source).get("rows", []):
            if row.get("version") in {"v4.5.0", "v4.3.4", "v4.3.3", "v4.2.0", "v3.1.3", "v2.1.0"}:
                rows.append({k: row.get(k, "") for k in rows[0].keys()})
    else:
        source_path = EXPERIMENT_ROOT.parent / "v4.3.4-v433-threshold-and-loss-tuning" / "charts" / "v434_v433_v420_v313_v210_comparison.json"
        if source_path.exists():
            for row in load_json(source_path).get("rows", []):
                if row.get("version") in {"v4.3.4", "v4.3.3", "v4.2.0", "v3.1.3", "v2.1.0"}:
                    rows.append({k: row.get(k, "") for k in rows[0].keys()})
    # Also include v6.1.0 and v6.0.0 from their comparison JSONs
    v610_source = EXPERIMENT_ROOT.parent / "v6.1.0-token-limit-2048" / "charts" / "v610_v434_v433_v420_v313_v210_comparison.json"
    if v610_source.exists():
        for row in load_json(v610_source).get("rows", []):
            if row.get("version") in {"v6.1.0"}:
                rows.append({k: row.get(k, "") for k in rows[0].keys()})
    v600_source = EXPERIMENT_ROOT.parent / "v6.0.0-transformer-on-v450" / "charts" / "v600_v434_v433_v420_v313_v210_comparison.json"
    if v600_source.exists():
        for row in load_json(v600_source).get("rows", []):
            if row.get("version") in {"v6.0.0"}:
                rows.append({k: row.get(k, "") for k in rows[0].keys()})
    base_iou, base_seg, base_pd, base_fa = rec["iou"], rec["seg_acc"], rec["pd"], rec["fa"]
    for row in rows:
        row["delta_iou_vs_v600"] = row["iou"] - base_iou if row["iou"] != "" else ""
        row["delta_seg_acc_vs_v600"] = row["seg_acc"] - base_seg if row["seg_acc"] != "" else ""
        row["delta_pd_vs_v600"] = row["pd"] - base_pd if row["pd"] != "" else ""
        row["delta_fa_vs_v600"] = row["fa"] - base_fa if row["fa"] != "" else ""
    csv_path = CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.csv"
    write_csv(csv_path, rows)
    best_versions = {"iou": max(rows, key=lambda r: r["iou"] if r["iou"] != "" else -math.inf)["version"], "seg_acc": max(rows, key=lambda r: r["seg_acc"] if r["seg_acc"] != "" else -math.inf)["version"], "pd": max(rows, key=lambda r: r["pd"] if r["pd"] != "" else -math.inf)["version"], "fa": min(rows, key=lambda r: r["fa"] if r["fa"] != "" else math.inf)["version"]}
    conclusion = f"v6.2.0 recommended threshold={rec['threshold']:.2f}, IoU={rec['iou']:.6f}, FA={rec['fa']:.9f}; comparison uses the same validation protocol as v4.3.4."
    summary = {"comparison": "v6.2.0 vs v6.1.0 vs v6.0.0 vs v4.5.0 vs v4.3.4 vs v4.3.3 vs v4.2.0 vs v3.1.3 vs v2.1.0", "generated_at": date.today().isoformat(), "metrics_direction": {"iou": "higher_better", "seg_acc": "higher_better", "pd": "higher_better", "fa": "lower_better"}, "rows": rows, "best": best_versions, "conclusion": conclusion, "artifact_paths": {"csv": str(csv_path), "json": str(CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.json"), "markdown": str(CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.md"), "svg": str(CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.svg")}}
    with open(CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    write_comparison_markdown(rows, summary)
    write_comparison_svg(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


def write_comparison_markdown(rows, summary):
    lines = ["# v6.2.0 / v6.1.0 / v6.0.0 / v4.5.0 / v4.3.4 / v4.3.3 / v4.2.0 / v3.1.3 / v2.1.0 同口径对比", "", "说明：IoU、seg_acc、PD 越高越好，FA 越低越好。", "", "| 版本 | 工作点 | IoU | seg_acc | PD | FA | threshold | epochs | 提交校验 |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['version']} | {r['work_point']} | {float(r['iou']):.6f} | {float(r['seg_acc']):.6f} | {float(r['pd']):.6f} | {float(r['fa']):.9f} | {r['threshold']} | {r['epochs']} | {r['submission_validation_status']} / {r['submission_txt_count']} txt |")
    lines += ["", "## 结论", "", f"- {summary['conclusion']}"]
    (CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_svg(rows):
    width, height = 980, 460
    margin = 70
    bar_w = 22
    gap = 18
    metrics = [("iou", "#1f77b4"), ("seg_acc", "#2ca02c"), ("pd", "#9467bd"), ("fa", "#d62728")]
    group_w = len(metrics) * bar_w + (len(metrics) - 1) * 4
    max_metric = max(max(float(r[m]) for r in rows) for m, _ in metrics[:3])
    max_fa = max(float(r["fa"]) for r in rows)
    plot_h = height - 130
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="#ffffff"/>', f'<text x="{width/2}" y="26" text-anchor="middle" font-size="16" font-family="Arial" fill="#111">v6.2.0 full-token comparison</text>', f'<line x1="{margin}" y1="{height-70}" x2="{width-40}" y2="{height-70}" stroke="#333"/>']
    for i, r in enumerate(rows):
        gx = margin + i * (group_w + gap)
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
    (CHARTS_DIR / "v620_v434_v433_v420_v313_v210_comparison.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")


def command_quick_check(args):
    ensure_dirs()
    cfg, _ = load_yaml_config(args.config)
    cfg.root = str(args.data_root)
    cfg.model_name = EXPERIMENT_NAME
    cfg.input_channel = 8
    cfg.max_events_num = 50000
    start = time.time()
    feature_report = {"status": "skipped", "reason": "no npz sample found"}
    samples = sorted((args.data_root / args.split).glob("*.npz"))
    if samples:
        sample = samples[0]
        features = load_or_compute_features(sample, cfg, use_cache=args.cache)
        with np.load(sample, allow_pickle=False) as data:
            rows = int(data["evs_norm"].shape[0])
        feature_report = {"status": "ok", "sample": str(sample), "rows": rows, "feature_shape": list(features.shape), "finite": bool(np.isfinite(features).all()), "min": features.min(axis=0).round(6).tolist(), "max": features.max(axis=0).round(6).tolist()}
    sync_global_cfg(cfg)
    net = build_model(cfg, train=False, use_cuda=False)
    time_weight_sums = net.temporal_weight_sums()
    dir_agg_skipped = getattr(net, "bottleneck_dir_agg", None) is not None and net.bottleneck_dir_agg.skipped
    direction_head_params = sum(p.numel() for p in net.direction_head.parameters())
    normalized_report = {"status": "ok", "sample": str(samples[0]) if samples else "", "rows": int(feature_report.get("rows", 0)), "feature_shape": feature_report.get("feature_shape", []), "finite": feature_report.get("finite", False), "min": feature_report.get("min", []), "max": feature_report.get("max", [])}
    if samples:
        with np.load(samples[0], allow_pickle=False) as data:
            evs_norm = data["evs_norm"].astype(np.float32)
            raw = np.concatenate([evs_norm[:, 0:4], load_or_compute_features(samples[0], cfg, use_cache=args.cache)], axis=1).astype(np.float32)
        stats = load_input_stats(cfg)
        normalized = normalize_input_features(raw, stats, cfg)
        normalized_report = {"status": "ok", "sample": str(samples[0]), "rows": int(raw.shape[0]), "feature_shape": list(normalized.shape), "finite": bool(np.isfinite(normalized).all()), "min": normalized.min(axis=0).round(6).tolist(), "max": normalized.max(axis=0).round(6).tolist(), "raw_mean": raw.mean(axis=0).round(6).tolist(), "raw_std": raw.std(axis=0).round(6).tolist(), "normalized_mean": normalized.mean(axis=0).round(6).tolist(), "normalized_std": normalized.std(axis=0).round(6).tolist()}
    sparse_forward_report = {"status": "skipped", "reason": "no npz sample found"}
    if samples:
        try:
            dataset = V4EvUAV(cfg, mode=args.split, use_cache=args.cache)
            sample_item = dataset[0]
            max_events = min(int(args.forward_events), int(sample_item["ev_loc"].shape[0]))
            sample_item = dict(sample_item)
            for key in ("ev_loc", "evs_feature", "seg_label", "idx", "raw_event"):
                sample_item[key] = sample_item[key][:max_events]
            import spconv.pytorch as spconv
            voxel_locs = torch.from_numpy(np.hstack((np.zeros((sample_item["ev_loc"].shape[0], 1)), sample_item["ev_loc"]))).to(torch.int32).contiguous()
            voxel_feats = torch.from_numpy(sample_item["evs_feature"]).float().contiguous()
            spatial_shape = np.array([11 * 32, 9 * 32, 256 * 32])
            voxel_ev = spconv.SparseConvTensor(voxel_feats, voxel_locs, spatial_shape, 1)
            with torch.no_grad():
                preds, voxel, direction_pred, bottleneck_indices = net(voxel_ev, return_direction=True)
            transformer_report = dict(getattr(net.bottleneck_transformer, "last_report", {}))
            sparse_forward_report = {
                "status": "ok",
                "sample": str(samples[0]),
                "events_used": max_events,
                "input_voxels": int(voxel_feats.shape[0]),
                "output_shape": list(preds.shape),
                "voxel_output_shape": list(voxel.features.shape),
                "direction_shape": list(direction_pred.shape),
                "bottleneck_indices_shape": list(bottleneck_indices.shape),
                "output_finite": bool(torch.isfinite(preds).all().item()),
                "voxel_finite": bool(torch.isfinite(voxel.features).all().item()),
                "direction_finite": bool(torch.isfinite(direction_pred).all().item()),
                "transformer_applied": bool(getattr(net, "transformer_applied", False)),
                "transformer": transformer_report,
            }
        except Exception as exc:
            sparse_forward_report = {"status": "failed", "error": repr(exc)}
            raise
    transformer_params = sum(p.numel() for p in net.bottleneck_transformer.parameters())
    report = {"version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "model_class": type(net).__name__, "channel_schedule": net.channel_schedule, "downsample_strides": net.downsample_strides, "no_patchattention": bool(net.no_patchattention), "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": dir_agg_skipped, "trajectory_direction_head": True, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "direction_head_params": direction_head_params, "transformer_enabled": True, "transformer_insert_position": "after_bottleneck_mstb", "transformer_token_limit": int(getattr(cfg, "transformer_token_limit", TRANSFORMER_TOKEN_LIMIT)), "transformer_num_heads": int(getattr(cfg, "transformer_num_heads", TRANSFORMER_NUM_HEADS)), "transformer_params": transformer_params, "temporal_weight_sums": time_weight_sums, "temporal_weights_sum_to_one": all(abs(value - 1.0) < 1e-6 for value in time_weight_sums.values()), "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "auxiliary_supervision": "main_output_xy_projection_dice_train_only+direction_head", "features": feature_report, "normalized_features": normalized_report, "sparse_forward": sparse_forward_report, "seconds": round(time.time() - start, 3)}
    with open(RESULTS_DIR / "v620_quick_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="v6.2.0 full token (8192, no subsampling) on v6.1.0 Transformer base")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_V4_CONFIG)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p = sub.add_parser("prepare-config")
    p.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_V4_CONFIG)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--feature-k", type=int, default=32)
    p.add_argument("--feature-radius", type=float, default=0.035)
    p.add_argument("--density-sigma", type=float, default=0.018)
    p.add_argument("--linearity-d0", type=float, default=12.0)
    p.add_argument("--temporal-tau", type=float, default=0.02)
    p.add_argument("--negative-slope", type=float, default=0.01)
    p.set_defaults(func=command_prepare_config)
    p = sub.add_parser("compute-stats", parents=[common])
    p.add_argument("--split", default="train")
    p.set_defaults(func=command_compute_stats)
    p = sub.add_parser("quick-check", parents=[common])
    p.add_argument("--split", default="train")
    p.add_argument("--cache", action="store_true")
    p.add_argument("--forward-events", type=int, default=2048)
    p.set_defaults(func=command_quick_check)
    p = sub.add_parser("threshold-scan", parents=[common])
    p.add_argument("--weight", type=Path, default=MODELS_DIR / "best_iou_seed316.pt")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.95, 0.97])
    p.add_argument("--inference-chunk-size", type=int, default=50000)
    p.set_defaults(func=command_threshold_scan)
    p = sub.add_parser("build-comparison", parents=[common])
    p.set_defaults(func=command_build_comparison)
    p = sub.add_parser("train", parents=[common])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--eval-last-epochs", type=int, default=10)
    p.set_defaults(func=command_train)
    p = sub.add_parser("evaluate", parents=[common])
    p.add_argument("--weight", type=Path, default=MODELS_DIR / "best_iou_seed316.pt")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--inference-chunk-size", type=int, default=50000)
    p.set_defaults(func=command_evaluate)
    p = sub.add_parser("submission", parents=[common])
    p.add_argument("--weight", type=Path, default=MODELS_DIR / "best_iou_seed316.pt")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--inference-chunk-size", type=int, default=50000)
    p.add_argument("--name", default="v620_full_token_seed316_best_submission")
    p.set_defaults(func=command_submission)
    p = sub.add_parser("validate-submission")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--txt-dir", type=Path, default=SUBMISSION_DIR / "v620_full_token_seed316_best_submission" / "txt")
    p.add_argument("--zip", type=Path, default=SUBMISSION_DIR / "v620_full_token_seed316_best_submission.zip")
    p.set_defaults(func=command_validate_submission)
    return parser


def main():
    ensure_dirs()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()