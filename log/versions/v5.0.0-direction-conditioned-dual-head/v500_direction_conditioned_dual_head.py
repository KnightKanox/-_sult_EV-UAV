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
try:
    project_root_exists = PROJECT_ROOT.exists()
except PermissionError:
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
DEFAULT_V5_CONFIG = CONFIGS_DIR / "v500_direction_conditioned_dual_head.yaml"
VERSION = "v5.0.0"
MAX_DIRECTION_VOXELS = 1024
EXPERIMENT_NAME = "direction_conditioned_dual_head"
SEED = 316
TRAJECTORY_PROJECTION_DICE_WEIGHT = 0.3
TRAJECTORY_PROJECTION_DICE_MODE = "xy_mean_projection_dice_train_only"
TRAJECTORY_DIRECTION_WEIGHT = 0.1
DIRECTION_NEIGHBOR_K = 16
DIRECTION_HEAD_HIDDEN = 128
LOCAL_DIRECTION_MAX_NEIGHBORS = 44
LOCAL_DIRECTION_DT = 2
LOCAL_DIRECTION_DXY = 1


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
        model_name="direction_conditioned_dual_head",
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
            "model_name": "direction_conditioned_dual_head",
            "input_channel": 8,
            "stem_channels": 24,
            "stage2_channels": 40,
            "stage3_channels": 64,
            "bottleneck_channels": 96,
            "downsample_strides": [[2, 2, 2], [2, 2, 2], [2, 2, 4]],
            "no_patchattention": True,
            "activation": "LeakyReLU",
            "bottleneck_direction_aggregation": True,
            "local_direction_neighborhood": {"dx": LOCAL_DIRECTION_DXY, "dy": LOCAL_DIRECTION_DXY, "dt": LOCAL_DIRECTION_DT, "max_neighbors": LOCAL_DIRECTION_MAX_NEIGHBORS},
            "trajectory_direction_head": True,
            "direction_head_hidden": DIRECTION_HEAD_HIDDEN,
            "direction_neighbor_k": DIRECTION_NEIGHBOR_K,
            "direction_confidence_head": True,
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
            "initial_loss": "L_main + 0.3*L_traj + 0.1*L_dir",
            "direction_loss_components": "abs_cosine + 0.5*confidence_bce + 0.5*confidence_mse",
            "auxiliary_supervision": TRAJECTORY_PROJECTION_DICE_MODE + "+direction_confidence_head",
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
            "chunked_inference": True,
            "inference_chunk_size": 50000,
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
    print(json.dumps({"config": str(args.output), "version": VERSION, "experiment": EXPERIMENT_NAME, "input_channel": 8, "max_events_num": 50000, "optimizer": "Adam", "seed": SEED, "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "bottleneck_direction_aggregation": True, "trajectory_direction_head": True}, indent=2))


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

    class DirectionPredictionHead(nn.Module):
        def __init__(self, in_channels, hidden=128, negative_slope=0.01):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(in_channels, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(negative_slope, inplace=False),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
                nn.LeakyReLU(negative_slope, inplace=False),
            )
            self.direction = nn.Linear(hidden, 3)
            self.confidence = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

        def forward(self, features):
            shared = self.shared(features)
            direction_raw = self.direction(shared)
            direction = direction_raw / torch.norm(direction_raw, dim=1, keepdim=True).clamp_min(1e-8)
            confidence = self.confidence(shared).reshape(-1)
            return direction, confidence

    class BottleneckDirectionAggregation(nn.Module):
        def __init__(self, channels, max_neighbors=44, dx=1, dy=1, dt=2, time_tau=2.0, feature_temperature=0.25):
            super().__init__()
            self.channels = channels
            self.max_neighbors = max_neighbors
            self.dx = dx
            self.dy = dy
            self.dt = dt
            self.time_tau = time_tau
            self.feature_temperature = feature_temperature
            self.skipped = False
            self.last_stats = {"direction_aggregation_skipped": False, "aggregation_apply_ratio": 0.0, "avg_neighbor_count": 0.0, "bottleneck_voxels": 0}

        def forward(self, sp_tensor, direction, confidence):
            features = sp_tensor.features
            indices = sp_tensor.indices
            num_voxels = int(indices.shape[0])
            self.skipped = False
            if num_voxels < 2:
                self.last_stats = {"direction_aggregation_skipped": False, "aggregation_apply_ratio": 0.0, "avg_neighbor_count": 0.0, "bottleneck_voxels": num_voxels}
                return sp_tensor
            key_to_idx = {tuple(int(v) for v in indices[i].detach().cpu().tolist()): i for i in range(num_voxels)}
            out_features = features.clone()
            neighbor_counts = []
            applied = 0
            norm_features = torch.nn.functional.normalize(features, dim=1)
            for i in range(num_voxels):
                b, x, y, t = (int(v) for v in indices[i].detach().cpu().tolist())
                neighbors = []
                for dt in range(-self.dt, self.dt + 1):
                    for dx in range(-self.dx, self.dx + 1):
                        for dy in range(-self.dy, self.dy + 1):
                            if dx == 0 and dy == 0 and dt == 0:
                                continue
                            j = key_to_idx.get((b, x + dx, y + dy, t + dt))
                            if j is not None:
                                neighbors.append((j, dx, dy, dt))
                if not neighbors:
                    neighbor_counts.append(0)
                    continue
                if len(neighbors) > self.max_neighbors:
                    neighbors = neighbors[:self.max_neighbors]
                neighbor_idx = torch.tensor([n[0] for n in neighbors], device=features.device, dtype=torch.long)
                delta = torch.tensor([[n[1], n[2], n[3]] for n in neighbors], device=features.device, dtype=features.dtype)
                delta_norm = delta / torch.tensor([max(self.dx, 1), max(self.dy, 1), max(self.dt, 1)], device=features.device, dtype=features.dtype)
                delta_unit = torch.nn.functional.normalize(delta_norm, dim=1)
                dir_score = torch.abs((delta_unit * direction[i].unsqueeze(0)).sum(dim=1))
                time_score = -torch.abs(delta[:, 2]) / max(float(self.time_tau), 1e-6)
                feat_score = (norm_features[neighbor_idx] * norm_features[i].unsqueeze(0)).sum(dim=1) / max(float(self.feature_temperature), 1e-6)
                weights = torch.softmax(dir_score + time_score + feat_score, dim=0)
                aggregated = (weights.unsqueeze(1) * features[neighbor_idx]).sum(dim=0)
                gate = confidence[i].clamp(0.0, 1.0)
                out_features[i] = features[i] + gate * aggregated
                neighbor_counts.append(len(neighbors))
                applied += 1
            avg_neighbors = float(sum(neighbor_counts) / max(len(neighbor_counts), 1))
            self.last_stats = {"direction_aggregation_skipped": False, "aggregation_apply_ratio": float(applied / max(num_voxels, 1)), "avg_neighbor_count": avg_neighbors, "bottleneck_voxels": num_voxels}
            return sparse_replace(sp_tensor, out_features)

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
            # Bottleneck: MSTB + direction-aware aggregation
            self.bottleneck_mstb = MSTB(c4, "v500_bottleneck", norm_fn, negative_slope)
            self.bottleneck_dir_agg = BottleneckDirectionAggregation(c4, max_neighbors=LOCAL_DIRECTION_MAX_NEIGHBORS, dx=LOCAL_DIRECTION_DXY, dy=LOCAL_DIRECTION_DXY, dt=LOCAL_DIRECTION_DT)
            self.bottleneck_dir_applied = False
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
            direction_pred, direction_confidence = self.direction_head(xb.features)
            xb = self.bottleneck_dir_agg(xb, direction_pred, direction_confidence)
            if not hasattr(self, "bottleneck_dir_applied") or not self.bottleneck_dir_applied:
                self.bottleneck_dir_applied = True
            y3 = self.decoder3(xb, x3)
            y2 = self.decoder2(y3, x2)
            y1 = self.decoder1(y2, x1)
            output = self.semantic_linear(y1.features)
            voxel = y1.replace_feature(output)
            if return_direction:
                return output, voxel, direction_pred, direction_confidence, xb.indices
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


def direction_confidence_loss(direction_pred, confidence_pred, direction_gt, confidence_gt, valid_mask):
    if not valid_mask.any():
        zero = direction_pred.new_zeros(())
        return zero, zero, zero
    cos_abs = torch.abs((direction_pred[valid_mask] * direction_gt[valid_mask]).sum(dim=1))
    cosine_loss = (1.0 - cos_abs).mean()
    conf_target = confidence_gt.to(dtype=confidence_pred.dtype)
    with torch.cuda.amp.autocast(enabled=False):
        bce_loss = torch.nn.functional.binary_cross_entropy(confidence_pred.float().clamp(1e-6, 1.0 - 1e-6), conf_target.float())
    mse_loss = torch.nn.functional.mse_loss(confidence_pred, conf_target)
    return cosine_loss + 0.5 * bce_loss + 0.5 * mse_loss, cosine_loss, bce_loss


def compute_direction_gt(bottleneck_indices, event_locs, event_labels, p2v_map, event_instance_ids=None, neighbor_k=16, max_direction_voxels=1024):
    device = bottleneck_indices.device
    bn_indices_cpu = bottleneck_indices.detach().cpu()
    event_locs_cpu = event_locs.detach().cpu()
    event_labels_cpu = event_labels.detach().cpu().bool()
    instance_cpu = None if event_instance_ids is None else torch.as_tensor(event_instance_ids).detach().cpu()
    N = bn_indices_cpu.shape[0]
    if N == 0:
        return torch.zeros(0, 3, device=device), torch.zeros(0, device=device), torch.zeros(0, dtype=torch.bool, device=device), {"direction_valid_ratio": 0.0}

    total_stride = torch.tensor([8, 8, 32], dtype=torch.int64)
    event_bn_coords = torch.div(event_locs_cpu[:, 1:].long(), total_stride, rounding_mode='trunc')
    bn_coords = bn_indices_cpu[:, 1:].long()
    bn_key_to_idx = {tuple(int(v) for v in bn_coords[i].tolist()): i for i in range(N)}
    voxel_fg = torch.zeros(N, dtype=torch.bool)
    voxel_instances = {}
    for e in torch.where(event_labels_cpu)[0].tolist():
        key = tuple(int(v) for v in event_bn_coords[e].tolist())
        idx = bn_key_to_idx.get(key)
        if idx is None:
            continue
        voxel_fg[idx] = True
        if instance_cpu is not None:
            voxel_instances.setdefault(idx, set()).add(int(instance_cpu[e].item()))
    fg_voxels = torch.where(voxel_fg)[0]
    if fg_voxels.numel() == 0:
        return torch.zeros(N, 3, device=device), torch.zeros(N, device=device), torch.zeros(N, dtype=torch.bool, device=device), {"direction_valid_ratio": 0.0}
    if fg_voxels.numel() > max_direction_voxels:
        fg_voxels = fg_voxels[torch.randperm(fg_voxels.numel())[:max_direction_voxels]]

    sampled_set = set(int(v) for v in fg_voxels.tolist())
    all_fg = [int(v) for v in torch.where(voxel_fg)[0].tolist()]
    direction_gt = torch.zeros(N, 3, dtype=torch.float32)
    confidence_gt = torch.zeros(N, dtype=torch.float32)
    valid_mask = torch.zeros(N, dtype=torch.bool)
    coord_f = bn_coords.float()
    for idx in sampled_set:
        center_instances = voxel_instances.get(idx, set())
        preferred = []
        if center_instances:
            preferred = [j for j in all_fg if j != idx and voxel_instances.get(j, set()).intersection(center_instances)]
        candidates = preferred + [j for j in all_fg if j != idx and j not in preferred]
        if len(candidates) < 2:
            continue
        offsets = coord_f[torch.tensor(candidates, dtype=torch.long)] - coord_f[idx]
        local_mask = (offsets[:, 0].abs() <= LOCAL_DIRECTION_DXY) & (offsets[:, 1].abs() <= LOCAL_DIRECTION_DXY) & (offsets[:, 2].abs() <= LOCAL_DIRECTION_DT)
        local_candidates = [candidates[k] for k in torch.where(local_mask)[0].tolist()]
        if len(local_candidates) < 2:
            local_candidates = candidates[:max(neighbor_k, 2)]
        else:
            local_candidates = local_candidates[:max(neighbor_k, 2)]
        if len(local_candidates) < 2:
            continue
        neigh = coord_f[torch.tensor([idx] + local_candidates, dtype=torch.long)]
        if neigh.shape[0] < 3:
            continue
        centered = neigh - neigh.mean(dim=0, keepdims=True)
        try:
            _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        except Exception:
            continue
        direction_gt[idx] = Vh[0]
        denom = float(S.sum().item()) + 1e-8
        confidence_gt[idx] = float(((S[0] - S[1]) / denom).clamp(0.0, 1.0).item()) if S.numel() > 1 else 1.0
        valid_mask[idx] = True
    stats = {"direction_valid_ratio": float(valid_mask.float().mean().item())}
    return direction_gt.to(device), confidence_gt.to(device), valid_mask.to(device), stats


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
    write_yaml_config(DEFAULT_V5_CONFIG, cfg)
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
        pbar = tqdm.tqdm(train_loader, desc=f"v500-epoch-{epoch}", unit="batch")
        for batch_index, ev in enumerate(pbar):
            label = ev["seg_label"].float().cuda()
            p2v_map = ev["p2v_map"].long().cuda()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds, voxel, direction_pred, direction_confidence, bottleneck_indices = net(ev["voxel_ev"], return_direction=True)
                main_loss = criterion(voxel, p2v_map, preds, label)
                event_probs = preds[p2v_map].reshape(-1)
                traj_loss = trajectory_projection_dice_loss(event_probs, label, ev["locs"].cuda())
                v_gt, conf_gt, dir_valid_mask, dir_stats = compute_direction_gt(
                    bottleneck_indices, ev["locs"].cuda(), label,
                    p2v_map, event_instance_ids=ev.get("idx_label"), neighbor_k=DIRECTION_NEIGHBOR_K,
                    max_direction_voxels=MAX_DIRECTION_VOXELS,
                )
                dir_loss, dir_cosine_loss, dir_confidence_bce = direction_confidence_loss(direction_pred, direction_confidence, v_gt, conf_gt, dir_valid_mask)
                loss = main_loss + TRAJECTORY_PROJECTION_DICE_WEIGHT * traj_loss + TRAJECTORY_DIRECTION_WEIGHT * dir_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            value = float(loss.item())
            main_value = float(main_loss.item())
            traj_value = float(traj_loss.item())
            dir_value = float(dir_loss.item())
            dir_cos_value = float(dir_cosine_loss.item())
            dir_conf_value = float(dir_confidence_bce.item())
            agg_stats = getattr(net.bottleneck_dir_agg, "last_stats", {})
            epoch_losses.append(value)
            loss_rows.append({"epoch": epoch, "batch": batch_index, "loss": value, "main_loss": main_value, "trajectory_projection_dice_loss": traj_value, "trajectory_direction_loss": dir_value, "direction_cosine_loss": dir_cos_value, "direction_confidence_bce": dir_conf_value, "direction_valid_ratio": dir_stats.get("direction_valid_ratio", 0.0), "aggregation_apply_ratio": agg_stats.get("aggregation_apply_ratio", 0.0), "avg_neighbor_count": agg_stats.get("avg_neighbor_count", 0.0), "bottleneck_voxels": agg_stats.get("bottleneck_voxels", 0), "direction_aggregation_skipped": False, "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "lr": optimizer.param_groups[0]["lr"]})
            if value < best_loss:
                best_loss = value
                torch.save(net.state_dict(), MODELS_DIR / "best_loss_seed316.pt")
            pbar.set_postfix(loss=value, main=main_value, traj_dice=traj_value, dir_loss=dir_value, dir_valid=dir_stats.get("direction_valid_ratio", 0.0))
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
        write_csv(RESULTS_DIR / "v500_loss_records.csv", loss_rows)
        write_csv(RESULTS_DIR / "v500_epoch_summary.csv", epoch_rows)
    agg_stats = getattr(net.bottleneck_dir_agg, "last_stats", {})
    direction_valid_ratio = loss_rows[-1].get("direction_valid_ratio", 0.0) if loss_rows else 0.0
    summary = {"status": "ok", "version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "epochs": int(cfg.epochs), "best_loss": best_loss, "best_iou": best_iou, "model_path": str(MODELS_DIR / "best_iou_seed316.pt"), "config": str(DEFAULT_V5_CONFIG), "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_projection_dice_mode": TRAJECTORY_PROJECTION_DICE_MODE, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "direction_confidence_head": True, "direction_gt_note": "foreground-limited PCA; same-instance preferred when idx_label mapping is available; no background PCA fallback", "auxiliary_supervision": TRAJECTORY_PROJECTION_DICE_MODE + "+direction_confidence_head", "inference_uses_auxiliary_head": False, "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": False, "aggregation_apply_ratio": agg_stats.get("aggregation_apply_ratio", 0.0), "avg_neighbor_count": agg_stats.get("avg_neighbor_count", 0.0), "direction_valid_ratio": direction_valid_ratio, "bottleneck_voxels": agg_stats.get("bottleneck_voxels", 0), "trajectory_direction_head": True}
    with open(RESULTS_DIR / "v500_training_summary.json", "w", encoding="utf-8") as f:
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
    agg_stats = getattr(net.bottleneck_dir_agg, "last_stats", {})
    direction_valid_ratio = "train_only"
    summary_path = RESULTS_DIR / "v500_training_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            direction_valid_ratio = json.load(f).get("direction_valid_ratio", direction_valid_ratio)
    metrics.update({"version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "inference_chunk_size": chunk_size, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "no_patchattention": True, "inference_uses_auxiliary_head": False, "trajectory_direction_head": True, "direction_confidence_head": True, "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": False, "aggregation_apply_ratio": agg_stats.get("aggregation_apply_ratio", 0.0), "avg_neighbor_count": agg_stats.get("avg_neighbor_count", 0.0), "direction_valid_ratio": direction_valid_ratio, "bottleneck_voxels": agg_stats.get("bottleneck_voxels", 0)})
    with open(RESULTS_DIR / "v500_validation_metrics.json", "w", encoding="utf-8") as f:
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
    with open(RESULTS_DIR / "v500_submission_validation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


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
    agg_stats = getattr(net.bottleneck_dir_agg, "last_stats", {})
    direction_head_params = sum(p.numel() for p in net.direction_head.parameters())
    report = {"version": VERSION, "experiment": EXPERIMENT_NAME, "seed": SEED, "max_events_num": 50000, "input_channel": int(cfg.input_channel), "optimizer": "Adam", "activation": "LeakyReLU", "model_class": type(net).__name__, "channel_schedule": net.channel_schedule, "downsample_strides": net.downsample_strides, "no_patchattention": bool(net.no_patchattention), "bottleneck_direction_aggregation": True, "direction_aggregation_skipped": False, "aggregation_apply_ratio": agg_stats.get("aggregation_apply_ratio", 0.0), "avg_neighbor_count": agg_stats.get("avg_neighbor_count", 0.0), "bottleneck_voxels": agg_stats.get("bottleneck_voxels", 0), "trajectory_direction_head": True, "direction_confidence_head": True, "direction_head_hidden": DIRECTION_HEAD_HIDDEN, "direction_neighbor_k": DIRECTION_NEIGHBOR_K, "direction_head_params": direction_head_params, "temporal_weight_sums": time_weight_sums, "temporal_weights_sum_to_one": all(abs(value - 1.0) < 1e-6 for value in time_weight_sums.values()), "trajectory_projection_dice_weight": TRAJECTORY_PROJECTION_DICE_WEIGHT, "trajectory_direction_weight": TRAJECTORY_DIRECTION_WEIGHT, "auxiliary_supervision": "main_output_xy_projection_dice_train_only+direction_confidence_head", "features": feature_report, "seconds": round(time.time() - start, 3)}
    with open(RESULTS_DIR / "v500_quick_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="v5.0.0 direction-conditioned dual head with local direction-aware bottleneck aggregation")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_V5_CONFIG)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p = sub.add_parser("prepare-config")
    p.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--output", type=Path, default=DEFAULT_V5_CONFIG)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
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
    p.add_argument("--name", default="v500_direction_conditioned_dual_head_seed316_best_submission")
    p.set_defaults(func=command_submission)
    p = sub.add_parser("validate-submission")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--txt-dir", type=Path, default=SUBMISSION_DIR / "v500_direction_conditioned_dual_head_seed316_best_submission" / "txt")
    p.add_argument("--zip", type=Path, default=SUBMISSION_DIR / "v500_direction_conditioned_dual_head_seed316_best_submission.zip")
    p.set_defaults(func=command_validate_submission)
    return parser


def main():
    ensure_dirs()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()