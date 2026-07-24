import csv
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

sys.path.insert(0, '/root/EV-UAV')

from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet

PROJECT_ROOT = Path('/root/EV-UAV')
DATA_ROOT = Path('/root/EV-UAV-dataset')
OUT_ROOT = PROJECT_ROOT / 'log' / 'versions' / '0.1.0-sample-pool-ratio-experiments'
CLASS_DIR = OUT_ROOT / 'classification'
CONFIG_SOURCE = PROJECT_ROOT / 'configs' / 'evisseg_evuav.yaml'
WEIGHT_PATH = PROJECT_ROOT / 'log' / 'model_leaky_relu' / 'best_iou_seed37.pt'
THRESHOLD = 0.5
POOL_SIZE = 99
SEED = 37
POOL_RATIOS = {
    'A': {'hard_negative': 0.25, 'structured_background': 0.25, 'uniform_background': 0.50},
    'B': {'hard_negative': 0.35, 'structured_background': 0.30, 'uniform_background': 0.35},
    'C': {'hard_negative': 0.45, 'structured_background': 0.30, 'uniform_background': 0.25},
    'D': {'hard_negative': 0.35, 'structured_background': 0.40, 'uniform_background': 0.25},
}
CLASS_ORDER = ['hard_negative', 'structured_background', 'uniform_background']


def load_cfg():
    with CONFIG_SOURCE.open() as f:
        config = yaml.safe_load(f)
    flat = {}
    for section in config.values():
        if isinstance(section, dict):
            flat.update(section)
    cfg = SimpleNamespace(**flat)
    cfg.root = str(DATA_ROOT)
    cfg.batch_size = 1
    return config, cfg


def class_dirs_present(train_root):
    names = {'hard_negative', 'structured_background', 'uniform_background'}
    return sorted([p.name for p in train_root.iterdir() if p.is_dir() and p.name in names])


def concentration(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    span = float(values.max() - values.min())
    if span <= 0:
        return 1.0
    return float(1.0 - min(1.0, (np.percentile(values, 90) - np.percentile(values, 10)) / span))


def spatial_concentration(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    x_span = max(float(x.max() - x.min()), 1.0)
    y_span = max(float(y.max() - y.min()), 1.0)
    area_ratio = ((np.percentile(x, 90) - np.percentile(x, 10)) / x_span) * ((np.percentile(y, 90) - np.percentile(y, 10)) / y_span)
    return float(1.0 - min(1.0, area_ratio))


def collect_metrics(cfg):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = evspsegnet(cfg).eval().to(device)
    net.load_state_dict(torch.load(str(WEIGHT_PATH), map_location=device))
    dataset = EvUAV(cfg, mode='train')
    dataset.file_list = sorted(dataset.file_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.custom_collate)

    rows = []
    with torch.no_grad():
        for sample_index, ev in enumerate(loader):
            preds, _ = net(ev['voxel_ev'])
            preds = preds[ev['p2v_map'].long().to(device)].squeeze().detach().cpu().numpy()
            labels = ev['seg_label'].numpy().astype(np.float32)
            locs = ev['locs'].numpy()
            pred_mask = preds >= THRESHOLD
            bg_mask = labels == 0
            pos_mask = labels == 1
            pred_pos_ratio = float(pred_mask.mean()) if pred_mask.size else 0.0
            target_event_ratio = float(pos_mask.mean()) if pos_mask.size else 0.0
            false_positive_ratio = float((pred_mask & bg_mask).sum() / max(1, bg_mask.sum()))
            true_positive_ratio = float((pred_mask & pos_mask).sum() / max(1, pos_mask.sum()))
            pred_locs = locs[pred_mask]
            if pred_locs.shape[0] > 1:
                spatial_score = spatial_concentration(pred_locs[:, 1], pred_locs[:, 2])
                temporal_score = concentration(pred_locs[:, 3])
            else:
                spatial_score = 0.0
                temporal_score = 0.0
            structure_score = float((spatial_score + temporal_score) / 2.0)
            rows.append({
                'sample_index': sample_index,
                'file': dataset.file_list[sample_index],
                'event_count_used': int(labels.shape[0]),
                'target_event_ratio': target_event_ratio,
                'prediction_positive_ratio': pred_pos_ratio,
                'background_false_positive_ratio': false_positive_ratio,
                'target_true_positive_ratio': true_positive_ratio,
                'spatial_concentration': spatial_score,
                'temporal_concentration': temporal_score,
                'structure_score': structure_score,
            })
    return rows


def assign_classes(rows):
    sorted_by_fp = sorted(rows, key=lambda r: (r['background_false_positive_ratio'], r['prediction_positive_ratio']), reverse=True)
    hard_n = len(rows) // 3
    hard_files = {r['file'] for r in sorted_by_fp[:hard_n]}
    remaining = [r for r in rows if r['file'] not in hard_files]
    structured_n = len(remaining) // 2
    structured_files = {r['file'] for r in sorted(remaining, key=lambda r: (r['structure_score'], r['prediction_positive_ratio']), reverse=True)[:structured_n]}
    for row in rows:
        if row['file'] in hard_files:
            row['class'] = 'hard_negative'
            row['class_reason'] = 'background_false_positive_ratio in top third after LeakyReLU inference'
        elif row['file'] in structured_files:
            row['class'] = 'structured_background'
            row['class_reason'] = 'higher spatial/temporal concentration among remaining samples'
        else:
            row['class'] = 'uniform_background'
            row['class_reason'] = 'lower false-positive and lower structure concentration among remaining samples'
    thresholds = {
        'hard_negative_min_background_false_positive_ratio': min(r['background_false_positive_ratio'] for r in rows if r['class'] == 'hard_negative'),
        'structured_background_min_structure_score': min(r['structure_score'] for r in rows if r['class'] == 'structured_background'),
    }
    return thresholds


def write_classification(rows, thresholds, existing_class_dirs):
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ['sample_index', 'file', 'class', 'class_reason', 'event_count_used', 'target_event_ratio', 'prediction_positive_ratio', 'background_false_positive_ratio', 'target_true_positive_ratio', 'spatial_concentration', 'temporal_concentration', 'structure_score']
    with (CLASS_DIR / 'sample_classification.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = {name: sum(1 for r in rows if r['class'] == name) for name in CLASS_ORDER}
    summary = {
        'source_train_root': str(DATA_ROOT / 'train'),
        'source_train_npz_count': len(rows),
        'existing_class_dirs_found': existing_class_dirs,
        'model_weight': str(WEIGHT_PATH),
        'prediction_threshold': THRESHOLD,
        'classification_basis': [
            'No existing hard_negative/structured_background/uniform_background annotation directories were found.',
            'All train npz files were inferred with the current LeakyReLU model best_iou weight.',
            'Metrics recorded per sample: target_event_ratio, prediction_positive_ratio, background_false_positive_ratio, spatial_concentration, temporal_concentration, structure_score.',
            'hard_negative: top third by background_false_positive_ratio, with prediction_positive_ratio as tie-breaker.',
            'structured_background: among remaining samples, top half by mean spatial/temporal concentration, with prediction_positive_ratio as tie-breaker.',
            'uniform_background: remaining lower false-positive and lower concentration samples.',
        ],
        'thresholds': thresholds,
        'class_counts': counts,
    }
    with (CLASS_DIR / 'classification_summary.json').open('w') as f:
        json.dump(summary, f, indent=2)
    lines = [
        'LeakyReLU train sample classification summary',
        f'source_train_root: {summary["source_train_root"]}',
        f'source_train_npz_count: {len(rows)}',
        f'existing_class_dirs_found: {existing_class_dirs}',
        f'model_weight: {WEIGHT_PATH}',
        f'prediction_threshold: {THRESHOLD}',
        '',
        'classification_basis:',
    ]
    lines.extend([f'- {item}' for item in summary['classification_basis']])
    lines.append('')
    lines.append('class_counts:')
    lines.extend([f'- {name}: {counts[name]}' for name in CLASS_ORDER])
    lines.append('')
    lines.append('thresholds:')
    lines.extend([f'- {k}: {v}' for k, v in thresholds.items()])
    (CLASS_DIR / 'classification_summary.txt').write_text('\n'.join(lines) + '\n')
    return counts


def target_counts(ratios):
    raw = {k: ratios[k] * POOL_SIZE for k in CLASS_ORDER}
    counts = {k: int(math.floor(raw[k])) for k in CLASS_ORDER}
    missing = POOL_SIZE - sum(counts.values())
    for k in sorted(CLASS_ORDER, key=lambda x: raw[x] - counts[x], reverse=True)[:missing]:
        counts[k] += 1
    return counts


def unique_copy_name(source_file, occurrence):
    stem = Path(source_file).stem
    suffix = Path(source_file).suffix
    if occurrence == 0:
        return source_file
    return f'{stem}__dup{occurrence:02d}{suffix}'


def build_pool(pool_name, ratios, rows):
    pool_root = OUT_ROOT / pool_name
    train_dir = pool_root / 'train'
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    by_class = {name: sorted([r['file'] for r in rows if r['class'] == name]) for name in CLASS_ORDER}
    desired = target_counts(ratios)
    copied_rows = []
    rng = random.Random(SEED + ord(pool_name))
    for class_name in CLASS_ORDER:
        sources = by_class[class_name]
        if not sources:
            raise RuntimeError(f'No samples available for class {class_name}')
        selected = []
        if desired[class_name] <= len(sources):
            selected = rng.sample(sources, desired[class_name])
        else:
            selected = sources[:]
            while len(selected) < desired[class_name]:
                selected.append(sources[(len(selected) - len(sources)) % len(sources)])
        occurrence_by_file = {}
        for source_file in selected:
            occurrence = occurrence_by_file.get(source_file, 0)
            copied_file = unique_copy_name(source_file, occurrence)
            occurrence_by_file[source_file] = occurrence + 1
            shutil.copy2(DATA_ROOT / 'train' / source_file, train_dir / copied_file)
            copied_rows.append({
                'source_file': source_file,
                'copied_file': copied_file,
                'class': class_name,
                'is_duplicated': 'true' if occurrence > 0 else 'false',
                'target_ratio': ratios[class_name],
                'actual_ratio': desired[class_name] / POOL_SIZE,
            })
    with (pool_root / 'manifest.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_file', 'copied_file', 'class', 'is_duplicated', 'target_ratio', 'actual_ratio'])
        writer.writeheader()
        writer.writerows(copied_rows)
    return desired


def write_config(pool_name, base_config):
    config = yaml.safe_load(yaml.dump(base_config, sort_keys=False))
    config['DATA']['root'] = str(OUT_ROOT / pool_name)
    config['TRAIN']['model_save_root'] = str(OUT_ROOT / pool_name / 'models')
    config['TEST']['model_path'] = str(OUT_ROOT / pool_name / 'models' / 'best_iou_seed37.pt')
    with (OUT_ROOT / pool_name / 'config.yaml').open('w') as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main():
    base_config, cfg = load_cfg()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    train_root = DATA_ROOT / 'train'
    existing_class_dirs = class_dirs_present(train_root)
    rows = collect_metrics(cfg)
    thresholds = assign_classes(rows)
    counts = write_classification(rows, thresholds, existing_class_dirs)
    pool_counts = {}
    for pool_name, ratios in POOL_RATIOS.items():
        pool_counts[pool_name] = build_pool(pool_name, ratios, rows)
        write_config(pool_name, base_config)
    with (OUT_ROOT / 'pool_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pool', 'class', 'target_ratio', 'actual_count', 'actual_ratio'])
        writer.writeheader()
        for pool_name, ratios in POOL_RATIOS.items():
            for class_name in CLASS_ORDER:
                writer.writerow({
                    'pool': pool_name,
                    'class': class_name,
                    'target_ratio': ratios[class_name],
                    'actual_count': pool_counts[pool_name][class_name],
                    'actual_ratio': pool_counts[pool_name][class_name] / POOL_SIZE,
                })
    print(json.dumps({'class_counts': counts, 'pool_counts': pool_counts}, indent=2))


if __name__ == '__main__':
    main()
