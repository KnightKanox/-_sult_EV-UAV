import csv
import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

VERSION = "v2.0.0"
THRESHOLD = 0.5
EPOCHS = 50
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
SEED = 37
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ.get("EVUAV_DATASET_DIR", "/root/EV-UAV-dataset"))

SUBDIRS = [
    "images/train", "images/val", "masks/train", "maps/train", "maps/val",
    "predictions/val", "submission", "logs", "model", "archive",
]


def ensure_dirs():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)


def get_ev(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    ev = data["ev"]
    if ev.dtype.names:
        x = ev["x"]
        y = ev["y"]
        t = ev["t"]
        p = ev["p"]
        label = ev["label"] if "label" in ev.dtype.names else np.zeros(len(ev), dtype=np.int64)
    else:
        x, y, t, p = ev[:, 0], ev[:, 1], ev[:, 2], ev[:, 3]
        label = ev[:, 4] if ev.shape[1] > 4 else np.zeros(len(ev), dtype=np.int64)
    return x, y, t, p, label


def normalize_xy(x, y):
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    min_x, min_y = int(xi.min()), int(yi.min())
    xi = xi - min_x
    yi = yi - min_y
    width = int(xi.max()) + 1
    height = int(yi.max()) + 1
    return xi, yi, width, height, min_x, min_y


def project_sample(npz_path, split):
    stem = npz_path.stem
    x, y, t, p, label = get_ev(npz_path)
    xi, yi, width, height, min_x, min_y = normalize_xy(x, y)
    n = len(xi)

    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32)
    positive_mask = np.zeros((height, width), dtype=np.uint8)

    polarity = (p > 0).astype(np.bool_)
    np.add.at(pos, (yi[polarity], xi[polarity]), 1.0)
    np.add.at(neg, (yi[~polarity], xi[~polarity]), 1.0)
    np.add.at(density, (yi, xi), 1.0)
    np.maximum.at(positive_mask, (yi, xi), (label > 0).astype(np.uint8))

    scale = max(float(density.max()), 1.0)
    red = 255.0 - np.clip(pos / scale * 255.0, 0, 255)
    green = 255.0 - np.clip(density / scale * 220.0, 0, 220)
    blue = 255.0 - np.clip(neg / scale * 255.0, 0, 255)
    image = np.stack([red, green, blue], axis=2).astype(np.uint8)

    image_path = BASE_DIR / "images" / split / f"{stem}.png"
    Image.fromarray(image, mode="RGB").save(image_path)

    mask_path = ""
    if split == "train":
        mask_path = BASE_DIR / "masks" / "train" / f"{stem}_mask.png"
        Image.fromarray((positive_mask * 255).astype(np.uint8), mode="L").save(mask_path)

    order = np.lexsort((np.arange(n, dtype=np.int64), xi, yi))
    sorted_pixel = yi[order].astype(np.int64) * width + xi[order].astype(np.int64)
    unique_pixels, start_idx, counts = np.unique(sorted_pixel, return_index=True, return_counts=True)
    event_indices_sorted = order.astype(np.int64)
    event_labels = (positive_mask[yi, xi] > 0).astype(np.uint8)
    map_path = BASE_DIR / "maps" / split / f"{stem}_pixel_event_map.npz"
    np.savez_compressed(
        map_path,
        width=np.array(width, dtype=np.int64),
        height=np.array(height, dtype=np.int64),
        min_x=np.array(min_x, dtype=np.int64),
        min_y=np.array(min_y, dtype=np.int64),
        pixel_ids=unique_pixels.astype(np.int64),
        start=start_idx.astype(np.int64),
        count=counts.astype(np.int64),
        event_indices=event_indices_sorted,
        event_x=x,
        event_y=y,
        event_t=t,
        event_p=p,
        event_label=event_labels,
        source_label=(label > 0).astype(np.uint8),
    )

    return {
        "split": split,
        "stem": stem,
        "source_npz": str(npz_path),
        "image_path": str(image_path),
        "mask_path": str(mask_path) if mask_path else "",
        "map_path": str(map_path),
        "width": width,
        "height": height,
        "event_count": n,
        "positive_event_count": int((label > 0).sum()),
        "positive_pixel_count": int(positive_mask.sum()),
        "status": "ok",
        "message": "",
    }


class EventImageDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(row["mask_path"]).convert("L"), dtype=np.float32) / 255.0
        image_t = torch.from_numpy(image.transpose(2, 0, 1))
        mask_t = torch.from_numpy(mask[None, :, :])
        return image_t, mask_t


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.block(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = self.block(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self.block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self.block(32, 16)
        self.out = nn.Conv2d(16, 1, 1)

    @staticmethod
    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        m = self.mid(self.pool2(e2))
        d2 = self.up2(m)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def train_model(rows):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EventImageDataset(rows)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    model = SmallUNet().to(device)
    positives = sum(int(r["positive_pixel_count"]) for r in rows)
    pixels = sum(int(r["width"]) * int(r["height"]) for r in rows)
    neg = max(pixels - positives, 1)
    pos_weight = torch.tensor([min(neg / max(positives, 1), 50.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    log_rows = []
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_losses = []
        for image, mask in loader:
            image = image.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        log_rows.append({"epoch": epoch, "mean_loss": mean_loss})
        print(f"epoch={epoch} mean_loss={mean_loss:.6f}", flush=True)

    weight_path = BASE_DIR / "model" / "small_unet_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model": "SmallUNet",
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "pos_weight": float(pos_weight.item()),
        "device": str(device),
    }, weight_path)

    with (BASE_DIR / "logs" / "train_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "mean_loss"])
        writer.writeheader()
        writer.writerows(log_rows)

    config = {
        "version": VERSION,
        "model": "SmallUNet (U-Net binary semantic segmentation)",
        "classes": {"0": "background", "1": "target"},
        "threshold": THRESHOLD,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "device": str(device),
        "pos_weight": float(pos_weight.item()),
        "weight_path": str(weight_path),
        "train_seconds": round(time.time() - start, 3),
    }
    (BASE_DIR / "logs" / "train_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return model, weight_path, config


def infer_and_restore(model, val_rows):
    device = next(model.parameters()).device
    model.eval()
    pred_rows = []
    with torch.no_grad():
        for row in val_rows:
            stem = row["stem"]
            image = np.asarray(Image.open(row["image_path"]).convert("RGB"), dtype=np.float32) / 255.0
            image_t = torch.from_numpy(image.transpose(2, 0, 1)[None]).to(device)
            prob = torch.sigmoid(model(image_t))[0, 0].cpu().numpy()
            pred = (prob >= THRESHOLD).astype(np.uint8)
            pred_path = BASE_DIR / "predictions" / "val" / f"{stem}_pred_mask.png"
            prob_path = BASE_DIR / "predictions" / "val" / f"{stem}_prob.npy"
            Image.fromarray((pred * 255).astype(np.uint8), mode="L").save(pred_path)
            np.save(prob_path, prob.astype(np.float32))

            m = np.load(row["map_path"], allow_pickle=True)
            width = int(m["width"])
            labels = np.zeros(len(m["event_indices"]), dtype=np.uint8)
            for pix, start, count in zip(m["pixel_ids"], m["start"], m["count"]):
                y = int(pix) // width
                x = int(pix) % width
                labels[int(start):int(start) + int(count)] = pred[y, x]
            restored = np.zeros_like(labels)
            restored[m["event_indices"].astype(np.int64)] = labels

            submit_path = BASE_DIR / "submission" / f"{stem}.txt"
            event_x, event_y, event_t, event_p = m["event_x"], m["event_y"], m["event_t"], m["event_p"]
            with submit_path.open("w", encoding="utf-8") as f:
                for x, y, t, p, lab in zip(event_x, event_y, event_t, event_p, restored):
                    f.write(f"{fmt_value(x)} {fmt_value(y)} {fmt_value(t)} {fmt_value(p)} {int(lab)}\n")

            pred_rows.append({
                "stem": stem,
                "image_path": row["image_path"],
                "map_path": row["map_path"],
                "prediction_mask_path": str(pred_path),
                "probability_path": str(prob_path),
                "submission_path": str(submit_path),
                "event_count": int(row["event_count"]),
                "pred_positive_pixel_count": int(pred.sum()),
                "pred_positive_event_count": int(restored.sum()),
                "threshold": THRESHOLD,
                "status": "ok",
                "message": "",
            })
    return pred_rows


def fmt_value(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.9f}"
    return str(v)


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_zip(pred_rows):
    zip_path = BASE_DIR / "archive" / "v2.0.0-2d-vision-pipeline_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for row in sorted(pred_rows, key=lambda r: r["stem"]):
            path = Path(row["submission_path"])
            zf.write(path, arcname=path.name)
    return zip_path


def validate_outputs(val_rows, pred_rows, zip_path):
    errors = []
    val_by_stem = {r["stem"]: r for r in val_rows}
    pred_by_stem = {r["stem"]: r for r in pred_rows}
    if set(val_by_stem) != set(pred_by_stem):
        errors.append("submission stems do not match val stems")
    for stem, row in sorted(val_by_stem.items()):
        submit_path = Path(pred_by_stem[stem]["submission_path"])
        if not submit_path.exists():
            errors.append(f"missing {submit_path}")
            continue
        lines = submit_path.read_text(encoding="utf-8").splitlines()
        if len(lines) != int(row["event_count"]):
            errors.append(f"{stem} line_count {len(lines)} != {row['event_count']}")
        for i, line in enumerate(lines[:10]):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{stem} line {i + 1} field_count {len(parts)}")
                break
            if parts[-1] not in {"0", "1"}:
                errors.append(f"{stem} line {i + 1} invalid label {parts[-1]}")
                break
        labels = set()
        for line in lines:
            parts = line.split()
            if len(parts) == 5:
                labels.add(parts[-1])
        if not labels.issubset({"0", "1"}):
            errors.append(f"{stem} contains non-binary labels")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
    if any("/" in name.rstrip("/") for name in names):
        errors.append("zip contains nested paths")
    if sorted(names) != sorted([f"{r['stem']}.txt" for r in val_rows]):
        errors.append("zip file list does not match val txt names")
    required = [BASE_DIR / "compression_index.csv", BASE_DIR / "prediction_index.csv", BASE_DIR / "解释文档.md", BASE_DIR / "model" / "small_unet_final.pt"]
    for path in required:
        if not path.exists():
            errors.append(f"missing required artifact {path}")
    report = {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "val_count": len(val_rows),
        "submission_count": len(pred_rows),
        "zip_path": str(zip_path),
        "zip_nested": any("/" in name.rstrip("/") for name in names),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (BASE_DIR / "logs" / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_doc(train_rows, val_rows, pred_rows, config, zip_path, validation):
    final_loss = ""
    train_log_path = BASE_DIR / "logs" / "train_log.csv"
    if train_log_path.exists():
        with train_log_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            final_loss = rows[-1]["mean_loss"]

    doc = f"""# v2.0.0 二维视觉训练识别管线说明

## 数据来源

本版本读取 Docker 容器内 `/root/EV-UAV-dataset/train/*.npz` 与 `/root/EV-UAV-dataset/val/*.npz`，宿主机对应目录为 `/media/sult/山江冬夏/训练集、验证集/train` 和 `/media/sult/山江冬夏/训练集、验证集/val`。

每个 `.npz` 包含 `ev`、`ev_loc`、`evs_norm`。本流程直接使用 `ev` 的 `x, y, t, p, label, name` 字段完成二维压缩和提交还原；其中验证集提交保留原始 `x y t p`，只替换第五列二分类标签。

## 二维压缩

- 使用原始事件 `x/y` 坐标确定性投影到二维平面，坐标按样本内最小 `x/y` 平移到图像原点。
- 输出白底 RGB 图片：红通道编码正极性事件密度，蓝通道编码负极性事件密度，绿通道编码总密度。
- 同一像素可对应多个原始事件点，映射文件保存 `pixel_ids/start/count/event_indices` 以及原始 `x/y/t/p`，用于从二维像素预测还原到全部事件。
- train mask 使用完整事件级标注聚合：同一像素只要任一事件 `label=1`，该像素 mask 即为 255，否则为 0。

## 模型与训练

- 模型：SmallUNet，U-Net 二分类语义分割结构。
- 类别：0=背景，1=目标。
- 输入：压缩后的 RGB 事件投影图，尺寸保持各样本原始二维范围。
- 损失：`BCEWithLogitsLoss`，使用训练像素正负比例设置 `pos_weight`。
- 训练配置：全量训练 {EPOCHS} epochs，batch_size={BATCH_SIZE}，learning_rate={LEARNING_RATE}，seed={SEED}，device={config['device']}。
- 训练环境：Docker 容器 `evuav`，项目路径 `/root/EV-UAV`，数据路径 `/root/EV-UAV-dataset`，Python `/root/exit/envs/evuav/bin/python3`。
- 最终训练 loss：epoch {EPOCHS} mean_loss={final_loss}。
- 推理阈值：{THRESHOLD}。
- 权重：`model/small_unet_final.pt`，容器内路径 `{config['weight_path']}`。

## 训练与推理结果

- train 样本：{len(train_rows)} 个，全部生成图片、mask 和像素事件映射。
- val 样本：{len(val_rows)} 个，全部生成图片和像素事件映射。
- 验证集预测：{len(pred_rows)} 个二维预测 mask，失败样本 0 个。
- 提交文件：`submission/val_*.txt`，共 {len(pred_rows)} 个。
- 提交压缩包：`archive/{Path(zip_path).name}`，zip 根目录直接包含 txt，无子目录嵌套。
- 验证状态：{validation['status']}。

## 输出目录

```text
v2.0.0-2d-vision-pipeline/
  run_v200_2d_pipeline.py
  compression_index.csv
  prediction_index.csv
  解释文档.md
  images/train/ images/val/
  masks/train/
  maps/train/ maps/val/
  predictions/val/
  submission/
  archive/
  logs/
  model/
```

## 逆向还原规则

对每个验证样本，先读取二维预测 mask，再根据映射文件找到投影到该像素的所有原始事件编号；该像素预测为 1 时，对应全部事件标签写 1，否则写 0。最终每个 `val_*.txt` 按原始事件顺序输出，每行严格为 `x y t p label`，事件数与源 `.npz` 完全一致。
"""
    (BASE_DIR / "解释文档.md").write_text(doc, encoding="utf-8")


def main():
    ensure_dirs()
    train_files = sorted((DATASET_DIR / "train").glob("*.npz"))
    val_files = sorted((DATASET_DIR / "val").glob("*.npz"))
    all_rows = []
    train_rows = []
    val_rows = []
    for path in train_files:
        row = project_sample(path, "train")
        train_rows.append(row)
        all_rows.append(row)
    for path in val_files:
        row = project_sample(path, "val")
        val_rows.append(row)
        all_rows.append(row)

    write_csv(BASE_DIR / "compression_index.csv", all_rows, [
        "split", "stem", "source_npz", "image_path", "mask_path", "map_path",
        "width", "height", "event_count", "positive_event_count", "positive_pixel_count", "status", "message",
    ])
    model, weight_path, config = train_model(train_rows)
    pred_rows = infer_and_restore(model, val_rows)
    write_csv(BASE_DIR / "prediction_index.csv", pred_rows, [
        "stem", "image_path", "map_path", "prediction_mask_path", "probability_path", "submission_path",
        "event_count", "pred_positive_pixel_count", "pred_positive_event_count", "threshold", "status", "message",
    ])
    zip_path = make_zip(pred_rows)
    write_doc(train_rows, val_rows, pred_rows, config, zip_path, {"status": "pending"})
    validation = validate_outputs(val_rows, pred_rows, zip_path)
    write_doc(train_rows, val_rows, pred_rows, config, zip_path, validation)
    print(json.dumps({
        "train": len(train_rows),
        "val": len(val_rows),
        "weight_path": str(weight_path),
        "zip_path": str(zip_path),
        "validation": validation,
    }, ensure_ascii=False, indent=2), flush=True)
    if validation["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
