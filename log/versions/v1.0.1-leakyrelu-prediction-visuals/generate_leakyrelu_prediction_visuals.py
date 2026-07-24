import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path('/root/EV-UAV')
DATA_ROOT = Path('/root/EV-UAV-dataset')
OUT_ROOT = Path('/tmp/v1.0.1-leakyrelu-prediction-visuals')
WEIGHT_PATH = Path('/root/EV-UAV/log/model_leaky_relu/best_iou_seed37.pt')
PYTHON_ENV = Path('/root/exit/envs/evuav/bin/python3')
THRESHOLD = 0.9
WIDTH = 346
HEIGHT = 260

sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet


def ensure_dirs():
    for rel in [
        'images/train',
        'images/val',
        'predictions/train',
        'predictions/val',
    ]:
        (OUT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def load_events(npz_path):
    with np.load(npz_path) as data:
        ev = data['ev']
        ev_loc = data['ev_loc']
        evs_norm = data['evs_norm']
        return ev.copy(), ev_loc.copy(), evs_norm.copy()


def make_batch(ev_loc, evs_norm):
    locs = np.hstack((np.zeros((ev_loc.shape[0], 1)), ev_loc))
    locs = torch.from_numpy(locs).to(torch.int64).contiguous()
    voxel_locs, p2v_map, v2p_map = EvUAV.custom_collate.__globals__['voxelization_idx'](locs, 1, 4)

    feats = torch.from_numpy(evs_norm[:, 0:4]).contiguous().float()
    voxelize = EvUAV.custom_collate.__globals__['voxelization']
    voxel_feats = voxelize(feats.cuda(), v2p_map.cuda(), 4).cuda()

    import spconv.pytorch as spconv
    spatial_shape = np.array([11 * 32, 9 * 32, 256 * 32])
    voxel_ev = spconv.SparseConvTensor(voxel_feats, voxel_locs.int().cuda(), spatial_shape, 1)
    return voxel_ev, p2v_map


def predict_file(net, npz_path):
    ev, ev_loc, evs_norm = load_events(npz_path)
    voxel_ev, p2v_map = make_batch(ev_loc, evs_norm)
    with torch.no_grad():
        preds, _ = net(voxel_ev)
        probs = preds[p2v_map.long().cuda()].reshape(-1).detach().cpu().numpy()
    labels = (probs >= THRESHOLD).astype(np.int64)
    if len(ev) != len(labels):
        raise ValueError(f'event count mismatch: ev={len(ev)} prediction={len(labels)}')
    return ev, probs, labels


def normalize_counts(counts):
    out = np.zeros_like(counts, dtype=np.float32)
    positive = counts > 0
    if np.any(positive):
        out[positive] = np.log1p(counts[positive]) / np.log1p(counts.max())
    return out


def render_image(ev, labels, image_path, ppm_path):
    canvas = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    xs = ev['x'].astype(np.int64)
    ys = ev['y'].astype(np.int64)
    ps = ev['p'].astype(np.int64)
    valid = (xs >= 0) & (xs < WIDTH) & (ys >= 0) & (ys < HEIGHT)
    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]
    labels = labels[valid]

    pos_counts = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    neg_counts = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    np.add.at(pos_counts, (ys[ps == 1], xs[ps == 1]), 1)
    np.add.at(neg_counts, (ys[ps == 0], xs[ps == 0]), 1)

    pos_norm = normalize_counts(pos_counts)
    neg_norm = normalize_counts(neg_counts)
    has_event = (pos_counts + neg_counts) > 0
    red = (255 - 150 * pos_norm - 70 * neg_norm).clip(35, 255)
    green = (255 - 115 * np.maximum(pos_norm, neg_norm)).clip(55, 255)
    blue = (255 - 150 * neg_norm - 70 * pos_norm).clip(35, 255)
    canvas[has_event, 0] = red[has_event].astype(np.uint8)
    canvas[has_event, 1] = green[has_event].astype(np.uint8)
    canvas[has_event, 2] = blue[has_event].astype(np.uint8)

    pred_mask = labels == 1
    pred_pixels = np.zeros((HEIGHT, WIDTH), dtype=bool)
    pred_pixels[ys[pred_mask], xs[pred_mask]] = True
    canvas[pred_pixels] = np.array([0, 80, 255], dtype=np.uint8)

    image = Image.fromarray(canvas, mode='RGB')
    image.save(ppm_path)
    image.save(image_path, quality=95)
    return int(pred_mask.sum())


def save_prediction(prediction_path, ev, probs, labels):
    with prediction_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 't', 'p', 'probability', 'label'])
        for i in range(len(ev)):
            writer.writerow([
                int(ev['x'][i]),
                int(ev['y'][i]),
                float(ev['t'][i]),
                int(ev['p'][i]),
                f'{float(probs[i]):.9f}',
                int(labels[i]),
            ])


def process_split(net, split):
    rows = []
    paths = sorted((DATA_ROOT / split).glob('*.npz'))
    for npz_path in paths:
        stem = npz_path.stem
        image_path = OUT_ROOT / 'images' / split / f'{stem}_prediction.jpg'
        ppm_path = OUT_ROOT / 'images' / split / f'{stem}_prediction.ppm'
        prediction_path = OUT_ROOT / 'predictions' / split / f'{stem}_prediction.csv'
        try:
            ev, probs, labels = predict_file(net, npz_path)
            save_prediction(prediction_path, ev, probs, labels)
            pred_positive_count = render_image(ev, labels, image_path, ppm_path)
            rows.append({
                'split': split,
                'source_npz': str(npz_path),
                'image_path': str(image_path),
                'ppm_path': str(ppm_path),
                'prediction_path': str(prediction_path),
                'event_count': len(ev),
                'pred_positive_count': pred_positive_count,
                'model_weight': str(WEIGHT_PATH),
                'status': 'ok',
                'message': f'LeakyReLU inference completed at threshold {THRESHOLD}',
            })
            print(f'{split}/{stem}: ok events={len(ev)} positive={pred_positive_count}', flush=True)
        except Exception as exc:
            rows.append({
                'split': split,
                'source_npz': str(npz_path),
                'image_path': str(image_path),
                'ppm_path': str(ppm_path),
                'prediction_path': str(prediction_path),
                'event_count': '',
                'pred_positive_count': '',
                'model_weight': str(WEIGHT_PATH),
                'status': 'failed',
                'message': f'{type(exc).__name__}: {exc}',
            })
            print(f'{split}/{stem}: failed {type(exc).__name__}: {exc}', flush=True)
    return rows


def write_index(rows):
    fieldnames = [
        'split', 'source_npz', 'image_path', 'ppm_path', 'prediction_path',
        'event_count', 'pred_positive_count', 'model_weight', 'status', 'message'
    ]
    with (OUT_ROOT / 'prediction_index.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ensure_dirs()
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f'model weight not found: {WEIGHT_PATH}')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available in Docker environment')
    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(WEIGHT_PATH, map_location='cuda:0'))
    rows = []
    rows.extend(process_split(net, 'train'))
    rows.extend(process_split(net, 'val'))
    write_index(rows)
    ok = sum(1 for row in rows if row['status'] == 'ok')
    failed = len(rows) - ok
    print(f'done total={len(rows)} ok={ok} failed={failed} out={OUT_ROOT}', flush=True)


if __name__ == '__main__':
    main()
    os._exit(0)
