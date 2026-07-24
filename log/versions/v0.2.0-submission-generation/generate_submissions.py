import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import tqdm

sys.path.insert(0, "/root/EV-UAV")

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet

OUT_ROOT = Path("/tmp/evuav_submission_v0_1_1/outputs")
THRESHOLD = 0.9
WEIGHTS = {
    "v0.1.0-D_best_iou_seed37": Path("/tmp/evuav_submission_v0_1_1/weights/D_best_iou_seed37.pt"),
    "v0.0.2-leaky-relu_best_iou_seed37": Path("/tmp/evuav_submission_v0_1_1/weights/v0.0.2_best_iou_seed37.pt"),
}


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
        predicted_ev["label"] = prediction

    np.savetxt(output_path, predicted_ev, fmt=["%d", "%d", "%.9f", "%d", "%d"], delimiter=" ")


def write_zip(txt_dir, zip_path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_path in sorted(txt_dir.glob("*.txt")):
            zf.write(txt_path, arcname=txt_path.name)


def generate_one(name, weight_path):
    txt_dir = OUT_ROOT / name / "txt"
    txt_dir.mkdir(parents=True, exist_ok=True)

    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(weight_path, map_location="cuda:0"))

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        shuffle=False,
    )

    with torch.no_grad():
        for batch_index, ev in enumerate(tqdm.tqdm(dataloader, desc=name, unit="video")):
            p2v_map = ev["p2v_map"].long().cuda()
            ev_locs = ev["locs"]
            preds, _ = net(ev["voxel_ev"])
            preds = preds[p2v_map].reshape(-1).cpu()

            first_sample = batch_index * cfg.batch_size
            for local_index in ev_locs[:, 0].long().unique(sorted=True).tolist():
                sample_index = first_sample + local_index
                file_name = dataset.file_list[sample_index]
                source_path = Path(dataset.root) / file_name
                output_path = txt_dir / f"{Path(file_name).stem}.txt"
                sample_mask = ev_locs[:, 0].long() == local_index
                prediction = (preds[sample_mask] >= THRESHOLD).to(torch.int64).numpy()
                save_prediction(source_path, output_path, prediction)

    write_zip(txt_dir, OUT_ROOT / f"{name}.zip")
    return len(list(txt_dir.glob("*.txt")))


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for name, weight_path in WEIGHTS.items():
        count = generate_one(name, weight_path)
        print(f"{name}: {count} txt files")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
