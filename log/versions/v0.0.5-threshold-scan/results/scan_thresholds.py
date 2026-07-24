import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

sys.path.insert(0, '/root/EV-UAV')

OUT = Path('/root/EV-UAV/log/threshold_scan')
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)]
TARGET_PDS = [0.75, 0.80, 0.85]
TARGET_FAS = [0.001, 0.002, 0.005, 0.010]


def switch_model_files(kind):
    if kind == 'baseline_ReLU':
        shutil.copy('/tmp/evspsegnet.relu_native.py', '/root/EV-UAV/model/evspsegnet.py')
        shutil.copy('/tmp/basemodel.relu_native.py', '/root/EV-UAV/model/basemodel.py')
    else:
        shutil.copy('/tmp/evspsegnet.leaky.py', '/root/EV-UAV/model/evspsegnet.py')
        shutil.copy('/tmp/basemodel.leaky.py', '/root/EV-UAV/model/basemodel.py')
    for p in [
        '/root/EV-UAV/model/__pycache__/evspsegnet.cpython-38.pyc',
        '/root/EV-UAV/model/__pycache__/basemodel.cpython-38.pyc',
    ]:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def collect_predictions(kind, weight_path):
    switch_model_files(kind)
    for mod in list(sys.modules):
        if mod.startswith('model.') or mod in {'model.evspsegnet', 'model.basemodel'}:
            sys.modules.pop(mod, None)

    from configs.configs import cfg
    from dataset.ev_uav import EvUAV
    from model.evspsegnet import evspsegnet

    device = 'cuda:0'
    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(weight_path, map_location=device))
    dataset = EvUAV(cfg, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate)

    samples = []
    with torch.no_grad():
        pbar = tqdm.tqdm(total=len(loader), desc=f'{kind}_pred', unit='video', leave=True)
        for i, ev in enumerate(loader):
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cpu()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().cpu()
            idx = torch.as_tensor(ev['idx_label']).cpu()
            preds, _ = net(x)
            preds = preds[p2v_map].squeeze().detach().cpu()
            samples.append({
                'sample': i,
                'preds': preds,
                'label': label,
                'idx': idx,
                'ev_locs': ev_locs,
                'ts': ev_locs[:, 3],
            })
            pbar.update(1)
        pbar.close()
    return samples


def metric_at_threshold(samples, thresh):
    preds_all = torch.cat([s['preds'] for s in samples], dim=0).clone()
    labels_all = torch.cat([s['label'] for s in samples], dim=0).clone()
    binary = (preds_all >= thresh).float()
    intersection = ((labels_all == 1) & (binary == 1)).sum().float()
    union = ((labels_all == 1) | (binary == 1)).sum().float()
    iou = float(intersection / union) if union > 0 else 0.0
    whole = (labels_all == 1).sum().float()
    correct = ((labels_all == 1) & (binary == 1)).sum().float()
    seg_acc = float(correct / whole) if whole > 0 else 0.0

    correct_thresh = 0.0001
    pd_detT = 50
    frame_num = 0
    obj_num = 0
    correct_num = 0
    false_num = 0

    for s in samples:
        preds = s['preds'].clone()
        label = s['label'].clone()
        idx = s['idx'].clone()
        ev_locs = s['ev_locs'].clone()
        ts = s['ts'].clone()
        frame_num += int((ts.max() - ts.min()) / pd_detT)

        for i in range(int((ts.max() - ts.min()) / pd_detT + 1)):
            t_range = (ts > i * pd_detT) * (ts < (i + 1) * pd_detT)
            idx_frame = idx[t_range]
            preds_frame = (preds[t_range].clone() >= thresh).float()
            label_frame = label[t_range]
            ev_locs_frame = ev_locs[:, 1:4][t_range]
            false_mask = np.zeros((260, 346), dtype=np.uint8)

            for idx_i in set(idx_frame.tolist()):
                if idx_i != 0:
                    obj_num += 1
                    mask = idx_frame == idx_i
                    preds_frame_i = preds_frame[mask]
                    label_frame_i = label_frame[mask]
                    denom = label_frame_i.sum()
                    if denom > 0:
                        num_correct_frame = (preds_frame_i == label_frame_i).sum()
                        if num_correct_frame / denom >= correct_thresh:
                            correct_num += 1

            false_ev = ev_locs_frame[(label_frame == 0) * (preds_frame == 1)]
            for ii in range(false_ev.shape[0]):
                y = int(false_ev[:, 1][ii])
                x = int(false_ev[:, 0][ii])
                if 0 <= y < 260 and 0 <= x < 346:
                    false_mask[y, x] += 1
                    num_labels, _, _, _ = cv2.connectedComponentsWithStats(false_mask, connectivity=8, ltype=cv2.CV_32S)
                    false_num += num_labels - 1

    pd = correct_num / obj_num if obj_num else 0.0
    fa = false_num / (frame_num * 346 * 260) if frame_num else 0.0
    return {
        'threshold': float(thresh),
        'iou': iou,
        'seg_acc': seg_acc,
        'pd': pd,
        'fa': fa,
        'obj_num': obj_num,
        'correct_num': correct_num,
        'false_num': false_num,
        'frame_num': frame_num,
    }


def evaluate_model(kind, weight_path):
    samples = collect_predictions(kind, weight_path)
    rows = []
    for t in tqdm.tqdm(THRESHOLDS, desc=f'{kind}_scan', unit='thr'):
        row = metric_at_threshold(samples, t)
        row['model'] = kind
        rows.append(row)
    with (OUT / f'{kind}_threshold_scan.csv').open('w', newline='') as f:
        fieldnames = ['model', 'threshold', 'iou', 'seg_acc', 'pd', 'fa', 'obj_num', 'correct_num', 'false_num', 'frame_num']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def interp_x_for_y(rows, x_key, y_key, target):
    pts = sorted([(r[x_key], r[y_key]) for r in rows], key=lambda p: p[1])
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    if target < min(ys) or target > max(ys):
        return None
    for i in range(1, len(pts)):
        y0, y1 = ys[i - 1], ys[i]
        x0, x1 = xs[i - 1], xs[i]
        if y0 <= target <= y1 or y1 <= target <= y0:
            if y1 == y0:
                return x0
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return xs[-1]


def write_outputs(all_rows):
    by_model = {}
    for row in all_rows:
        by_model.setdefault(row['model'], []).append(row)
    baseline = by_model['baseline_ReLU']
    leaky = by_model['LeakyReLU']

    fieldnames = ['model', 'threshold', 'iou', 'seg_acc', 'pd', 'fa', 'obj_num', 'correct_num', 'false_num', 'frame_num']
    with (OUT / 'threshold_scan_all.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    same_pd = []
    for target in TARGET_PDS:
        b_fa = interp_x_for_y(baseline, 'fa', 'pd', target)
        l_fa = interp_x_for_y(leaky, 'fa', 'pd', target)
        same_pd.append({
            'target_pd': target,
            'baseline_fa': b_fa,
            'leaky_fa': l_fa,
            'delta_fa': None if b_fa is None or l_fa is None else l_fa - b_fa,
        })
    with (OUT / 'same_pd_compare.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target_pd', 'baseline_fa', 'leaky_fa', 'delta_fa'])
        writer.writeheader()
        writer.writerows(same_pd)

    same_fa = []
    for target in TARGET_FAS:
        b_pd = interp_x_for_y(baseline, 'pd', 'fa', target)
        l_pd = interp_x_for_y(leaky, 'pd', 'fa', target)
        same_fa.append({
            'target_fa': target,
            'baseline_pd': b_pd,
            'leaky_pd': l_pd,
            'delta_pd': None if b_pd is None or l_pd is None else l_pd - b_pd,
        })
    with (OUT / 'same_fa_compare.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['target_fa', 'baseline_pd', 'leaky_pd', 'delta_pd'])
        writer.writeheader()
        writer.writerows(same_fa)

    with (OUT / 'threshold_scan_summary.txt').open('w') as f:
        f.write('Validation threshold scan, thresholds=0.05..0.95 step=0.05\n')
        f.write('Metrics use the EV-UAV eval logic from utils/eval.py.\n\n')
        f.write('Same PD, compare FA (lower is better):\n')
        for row in same_pd:
            f.write('PD={target_pd}: baseline_FA={baseline_fa}, leaky_FA={leaky_fa}, delta_FA={delta_fa}\n'.format(**row))
        f.write('\nSame FA, compare PD (higher is better):\n')
        for row in same_fa:
            f.write('FA={target_fa}: baseline_PD={baseline_pd}, leaky_PD={leaky_pd}, delta_PD={delta_pd}\n'.format(**row))

    plt.figure(figsize=(8, 6))
    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: r['fa'])
        plt.plot([r['fa'] for r in rows], [r['pd'] for r in rows], marker='o', label=model)
        for r in rows:
            if abs(r['threshold'] - 0.90) < 1e-9:
                plt.scatter([r['fa']], [r['pd']], s=90, marker='*')
                plt.text(r['fa'], r['pd'], '  t=0.90', fontsize=8)
    plt.xlabel('FA (lower is better)')
    plt.ylabel('PD (higher is better)')
    plt.title('Validation PD-FA Curve by Threshold')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'pd_fa_curve.png', dpi=180)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, metric, ylabel in zip(axes.ravel(), ['pd', 'fa', 'iou', 'seg_acc'], ['PD', 'FA', 'IoU', 'Seg Acc']):
        for model, rows in by_model.items():
            rows = sorted(rows, key=lambda r: r['threshold'])
            ax.plot([r['threshold'] for r in rows], [r[metric] for r in rows], marker='o', label=model)
        ax.set_xlabel('Threshold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle('Validation Metrics vs Threshold')
    plt.tight_layout()
    plt.savefig(OUT / 'metrics_vs_threshold.png', dpi=180)
    plt.close()

    valid_same_pd = [r for r in same_pd if r['baseline_fa'] is not None and r['leaky_fa'] is not None]
    if valid_same_pd:
        labels = [str(r['target_pd']) for r in valid_same_pd]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, [r['baseline_fa'] for r in valid_same_pd], width, label='baseline_ReLU')
        plt.bar(x + width / 2, [r['leaky_fa'] for r in valid_same_pd], width, label='LeakyReLU')
        plt.xticks(x, labels)
        plt.xlabel('Target PD')
        plt.ylabel('Interpolated FA')
        plt.title('FA at Same PD')
        plt.grid(axis='y', alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'fa_at_same_pd.png', dpi=180)
        plt.close()

    valid_same_fa = [r for r in same_fa if r['baseline_pd'] is not None and r['leaky_pd'] is not None]
    if valid_same_fa:
        labels = [str(r['target_fa']) for r in valid_same_fa]
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, [r['baseline_pd'] for r in valid_same_fa], width, label='baseline_ReLU')
        plt.bar(x + width / 2, [r['leaky_pd'] for r in valid_same_fa], width, label='LeakyReLU')
        plt.xticks(x, labels)
        plt.xlabel('Target FA')
        plt.ylabel('Interpolated PD')
        plt.title('PD at Same FA')
        plt.grid(axis='y', alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / 'pd_at_same_fa.png', dpi=180)
        plt.close()


def main():
    all_rows = []
    all_rows.extend(evaluate_model('baseline_ReLU', '/root/EV-UAV/log/model/best_iou_seed37.pt'))
    all_rows.extend(evaluate_model('LeakyReLU', '/root/EV-UAV/log/model_leaky_relu/best_iou_seed37.pt'))
    write_outputs(all_rows)
    switch_model_files('LeakyReLU')
    print('threshold scan done:', OUT)


if __name__ == '__main__':
    main()
