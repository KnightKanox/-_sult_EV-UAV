from pathlib import Path

import numpy as np
import torch
import tqdm

from configs.configs import cfg
from dataset.ev_uav import EvUAV
from model.evspsegnet import evspsegnet


MODEL_PATH = Path("/EV-UAV/pretrained/best_iou_seed37.pt")
OUTPUT_DIR = Path("/EV-UAV-dataset/val-pred-txt")
PREDICTION_THRESHOLD = 0.9


def save_prediction(source_path, output_path, prediction):
    with np.load(source_path) as data:
        source_ev = data["ev"]

        if len(source_ev) != len(prediction):
            raise ValueError(
                f"{source_path.name}: event count {len(source_ev)} does not "
                f"match prediction count {len(prediction)}"
            )

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

    np.savetxt(
        output_path,
        predicted_ev,
        fmt=["%d", "%d", "%.9f", "%d", "%d"],
        delimiter=" ",
    )


if __name__ == "__main__":
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model weight not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    net = evspsegnet(cfg).eval().cuda()
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cuda:0"))
    print("dict load: ", MODEL_PATH)

    dataset = EvUAV(cfg, mode="val")
    dataset.file_list = sorted(dataset.file_list)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=dataset.custom_collate,
        shuffle=False,
    )

    pbar = tqdm.tqdm(
        total=len(dataloader),
        desc="video",
        unit="video",
        unit_scale=True,
        position=0,
        leave=True,
    )

    for batch_index, ev in enumerate(dataloader):
        with torch.no_grad():
            p2v_map = ev["p2v_map"].long().cuda()
            ev_locs = ev["locs"]

            preds, _ = net(ev["voxel_ev"])
            preds = preds[p2v_map].reshape(-1).cpu()

            first_sample = batch_index * cfg.batch_size
            for local_index in ev_locs[:, 0].long().unique(sorted=True).tolist():
                sample_index = first_sample + local_index
                file_name = dataset.file_list[sample_index]
                source_path = Path(dataset.root) / file_name
                output_path = OUTPUT_DIR / f"{Path(file_name).stem}.txt"
                sample_mask = ev_locs[:, 0].long() == local_index
                prediction = (
                    preds[sample_mask] >= PREDICTION_THRESHOLD
                ).to(torch.int64).numpy()
                save_prediction(source_path, output_path, prediction)

        pbar.update(1)

    pbar.close()
    print(f"prediction txt files saved to: {OUTPUT_DIR}")
