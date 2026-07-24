import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import tqdm
import yaml

PROJECT_ROOT = Path('/root/EV-UAV')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENT_ROOT = PROJECT_ROOT / 'log/versions/0.1.0-sample-pool-ratio-experiments'
VAL_ROOT = Path('/root/EV-UAV-dataset')
SEED = 37


def load_config(config_path):
    with open(config_path, 'r') as f:
        raw = yaml.load(f, Loader=yaml.CLoader)
    flat = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat)


def setup(seed):
    print('random seed:' + str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_logger(log_path):
    logger = logging.getLogger('group_experiment')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def evaluate_model(net, cfg, dataset_root, EvUAV, evalute, thresh=0.9):
    eval_cfg = SimpleNamespace(**vars(cfg))
    eval_cfg.root = str(dataset_root)
    dataset = EvUAV(eval_cfg, mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate)
    evaluator = evalute(eval_cfg)
    net.eval()
    with torch.no_grad():
        for sample, ev in enumerate(dataloader):
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            ev_locs = ev['locs'].float().requires_grad_()
            idx = ev['idx_label']
            ts = ev_locs[:, 3]
            preds, voxel = net(x)
            preds = preds[p2v_map].squeeze().cpu()
            evaluator.matches[str(sample)] = {'seg_pred': preds.clone(), 'seg_gt': label.cpu()}
            if eval_cfg.roc:
                evaluator.roc_update(ts, preds.clone(), idx, label.cpu(), ev_locs, thresh=thresh)
    iou = float(evaluator.evaluate_semantic_segmantation_miou(thresh=thresh).item())
    seg_acc = float(evaluator.evaluate_semantic_segmantation_accuracy(thresh=thresh).item())
    pd_value, fa_value = evaluator.cal_roc() if eval_cfg.roc else (float('nan'), float('nan'))
    net.train()
    return {
        'iou': iou,
        'seg_acc': seg_acc,
        'pd': float(pd_value),
        'fa': float(fa_value),
        'threshold': thresh,
    }


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_group(group, cfg, global_cfg, EvUAV, evspsegnet, STCLoss, evalute):
    group_dir = EXPERIMENT_ROOT / group
    models_dir = group_dir / 'models'
    logs_dir = group_dir / 'logs'
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_save_root = str(models_dir)
    global_cfg.__dict__.update(vars(cfg))

    logger = configure_logger(logs_dir / 'train.log')
    logger.info('Start group %s training with config %s', group, group_dir / 'config.yaml')
    logger.info('epochs=%s batch_size=%s train_root=%s model_save_root=%s', cfg.epochs, cfg.batch_size, cfg.root, cfg.model_save_root)

    setup(SEED)
    net = evspsegnet(cfg).train().cuda()
    dataset = EvUAV(cfg, mode='train')
    sampler = torch.utils.data.sampler.RandomSampler(list(range(len(dataset))))
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=dataset.custom_collate, sampler=sampler)
    criterion = STCLoss(k=cfg.k, t=cfg.t, cfg=cfg).cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_loss = float('inf')
    best_iou = -1.0
    loss_rows = []
    epoch_rows = []

    for epoch in range(int(cfg.epochs)):
        epoch_losses = []
        pbar = tqdm.tqdm(total=len(train_dataloader), unit='Batch', unit_scale=True, desc='Group {} Epoch {}'.format(group, epoch), position=0, leave=True)
        for batch_index, ev in enumerate(train_dataloader):
            x = ev['voxel_ev']
            label = ev['seg_label'].float().cuda()
            p2v_map = ev['p2v_map'].long().cuda()
            preds, voxel = net(x)
            loss = criterion(voxel, p2v_map, preds, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            epoch_losses.append(loss_value)
            loss_rows.append({'group': group, 'epoch': epoch, 'batch': batch_index, 'loss': loss_value, 'lr': optimizer.param_groups[0]['lr']})
            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(net.state_dict(), models_dir / 'best_loss_seed37.pt')
            pbar.set_postfix(loss=loss_value)
            pbar.update(1)
            torch.cuda.empty_cache()
        pbar.close()
        scheduler.step()

        epoch_row = {
            'group': group,
            'epoch': epoch,
            'mean_loss': float(np.mean(epoch_losses)),
            'min_loss': float(np.min(epoch_losses)),
            'max_loss': float(np.max(epoch_losses)),
            'last_loss': float(epoch_losses[-1]),
            'best_loss_so_far': float(best_loss),
            'val_iou': '',
        }
        if epoch >= int(cfg.epochs) - 10:
            metrics = evaluate_model(net, cfg, VAL_ROOT, EvUAV, evalute)
            epoch_row['val_iou'] = metrics['iou']
            logger.info('group=%s epoch=%d mean_loss=%.6f best_loss=%.6f val_iou=%.6f', group, epoch, epoch_row['mean_loss'], best_loss, metrics['iou'])
            if metrics['iou'] > best_iou:
                best_iou = metrics['iou']
                torch.save(net.state_dict(), models_dir / 'best_iou_seed37.pt')
        else:
            logger.info('group=%s epoch=%d mean_loss=%.6f best_loss=%.6f', group, epoch, epoch_row['mean_loss'], best_loss)
        epoch_rows.append(epoch_row)

    pd.DataFrame(loss_rows).to_csv(group_dir / 'loss_records.csv', index=False)
    pd.DataFrame(epoch_rows).to_csv(group_dir / 'epoch_loss_summary.csv', index=False)
    logger.info('Finished group %s best_loss=%.6f best_iou=%.6f', group, best_loss, best_iou)
    return {'group': group, 'best_loss': best_loss, 'best_iou': best_iou}


def evaluate_checkpoint(group, cfg, global_cfg, EvUAV, evspsegnet, evalute):
    global_cfg.__dict__.update(vars(cfg))
    model_path = EXPERIMENT_ROOT / group / 'models/best_iou_seed37.pt'
    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(model_path))
    metrics = evaluate_model(net, cfg, VAL_ROOT, EvUAV, evalute)
    metrics['group'] = group
    metrics['model_path'] = str(model_path)
    return metrics


def plot_outputs(results_df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    charts_dir = EXPERIMENT_ROOT / 'charts'
    charts_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for group in ['A', 'B', 'C', 'D']:
        path = EXPERIMENT_ROOT / group / 'epoch_loss_summary.csv'
        if path.exists():
            df = pd.read_csv(path)
            plt.plot(df['epoch'], df['mean_loss'], marker='o', markersize=2, label=group)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Loss')
    plt.title('Epoch Mean Loss Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'group_epoch_loss_comparison.png', dpi=160)
    plt.close()

    metric_cols = ['iou', 'seg_acc', 'pd', 'fa']
    x = np.arange(len(results_df['group']))
    width = 0.2
    plt.figure(figsize=(10, 6))
    for i, metric in enumerate(metric_cols):
        plt.bar(x + (i - 1.5) * width, results_df[metric], width, label=metric)
    plt.xticks(x, results_df['group'])
    plt.ylabel('Value')
    plt.title('Group Validation Metrics')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / 'group_val_metrics_bar.png', dpi=160)
    plt.close()

    x = np.arange(len(results_df['group']))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - 0.18, results_df['pd'], 0.36, label='PD', color='#3b82f6')
    ax1.set_ylabel('PD')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['group'])
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, results_df['fa'], 0.36, label='FA', color='#ef4444')
    ax2.set_ylabel('FA')
    ax1.set_title('PD and FA Comparison')
    ax1.grid(axis='y', alpha=0.3)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    fig.tight_layout()
    fig.savefig(charts_dir / 'group_pd_fa_comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', default=['A', 'B', 'C', 'D'])
    parser.add_argument('--skip-train', action='store_true')
    args = parser.parse_args()

    first_config = EXPERIMENT_ROOT / args.groups[0] / 'config.yaml'
    sys.argv = [sys.argv[0], '--config', str(first_config)]
    from configs.configs import cfg as global_cfg
    from dataset.ev_uav import EvUAV
    from model.evspsegnet import evspsegnet
    from utils.stcloss import STCLoss
    from utils.eval import evalute

    train_summaries = []
    for group in args.groups:
        cfg = load_config(EXPERIMENT_ROOT / group / 'config.yaml')
        cfg.epochs = 50
        if not args.skip_train:
            train_summaries.append(train_group(group, cfg, global_cfg, EvUAV, evspsegnet, STCLoss, evalute))

    results = []
    for group in args.groups:
        cfg = load_config(EXPERIMENT_ROOT / group / 'config.yaml')
        cfg.epochs = 50
        results.append(evaluate_checkpoint(group, cfg, global_cfg, EvUAV, evspsegnet, evalute))
    results_df = pd.DataFrame(results)[['group', 'iou', 'seg_acc', 'pd', 'fa', 'threshold', 'model_path']]
    results_df.to_csv(EXPERIMENT_ROOT / 'group_val_results.csv', index=False)

    with open(EXPERIMENT_ROOT / 'group_val_summary.txt', 'w') as f:
        f.write('Group validation results on /root/EV-UAV-dataset/val, threshold=0.9\n')
        f.write(results_df.to_string(index=False))
        f.write('\n')
        if train_summaries:
            f.write('\nTraining summaries:\n')
            f.write(pd.DataFrame(train_summaries).to_string(index=False))
            f.write('\n')
    plot_outputs(results_df)
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
