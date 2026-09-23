import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

REPO_ROOT = next(
    path
    for path in [Path.cwd(), *Path.cwd().parents]
    if (path / "common" / "paths.py").is_file()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.paths import ROOT

from common.models import EmeanRegressor, TranslationNetwork

torch.set_float32_matmul_precision("high")

DATASET_NAME = "kidney"
NOTEBOOK_DIR = ROOT / "experiment" / DATASET_NAME / "04_other"
PAIR_CSV = (
    ROOT
    / "experiment"
    / DATASET_NAME
    / "01_data_processing"
    / "cache"
    / "pairs_seed_42.csv"
)
MODEL_DIR = ROOT / "experiment" / DATASET_NAME / "02_ablation"
GENERATOR_WEIGHT = MODEL_DIR / "all_innovations_generator.pt"
EMEAN_WEIGHT = MODEL_DIR / "all_innovations_emean_head.pt"
OUTPUT_ROOT = NOTEBOOK_DIR / "patient_level_exports"
MANIFEST_PATH = OUTPUT_ROOT / "export_manifest.json"
training = False
BATCH_SIZE = 16
NUM_WORKERS = 4
SEED = 42

pl.seed_everything(SEED, workers=True)


class SplitExportDataset(Dataset):
    def __init__(self, frame, image_size=256):
        self.frame = frame.reset_index(drop=True)
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row["gray_path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            gray = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
        return {
            "gray": gray,
            "true_emean": torch.tensor(float(row["emean"]), dtype=torch.float32),
            "image_name": str(row["image_name"]),
            "patient_id": str(row["patient_id"]),
            "source_gray_path": str(path),
        }


def make_loader(frame):
    return DataLoader(
        SplitExportDataset(frame),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


class SplitExportLightning(pl.LightningModule):
    def __init__(self, target_mean, target_std):
        super().__init__()
        self.translation = TranslationNetwork(use_fem=True)
        self.regressor = EmeanRegressor()
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    def forward(self, gray):
        virtual_swe = self.translation(gray)
        standardized = self.regressor(virtual_swe)
        predicted_emean = torch.expm1(
            standardized * self.target_std + self.target_mean
        ).clamp_min(0.0)
        return virtual_swe, predicted_emean

    def predict_step(self, batch, batch_index, dataloader_index=0):
        virtual_swe, predicted_emean = self(batch["gray"])
        return {
            "virtual_swe": virtual_swe.detach().cpu(),
            "predicted_emean": predicted_emean.detach().cpu(),
            "true_emean": batch["true_emean"].detach().cpu(),
            "image_name": list(batch["image_name"]),
            "patient_id": list(batch["patient_id"]),
            "source_gray_path": list(batch["source_gray_path"]),
        }


def load_weights(model):
    model.translation.load_state_dict(
        torch.load(
            GENERATOR_WEIGHT,
            map_location="cpu",
            weights_only=True,
        )
    )
    model.regressor.load_state_dict(
        torch.load(
            EMEAN_WEIGHT,
            map_location="cpu",
            weights_only=True,
        )
    )


def save_split(outputs, split):
    split_dir = OUTPUT_ROOT / split
    image_dir = split_dir / "virtual_swe"
    image_dir.mkdir(parents=True, exist_ok=True)
    for path in image_dir.glob("*.png"):
        path.unlink()
    rows = []
    sequence = 0
    for output in outputs:
        images = (
            torch.clamp((output["virtual_swe"] + 1.0) / 2.0, 0.0, 1.0)
            .mul(255)
            .byte()
            .permute(0, 2, 3, 1)
            .numpy()
        )
        predicted = output["predicted_emean"].numpy()
        observed = output["true_emean"].numpy()
        for image, prediction, target, image_name, patient_id, source_path in zip(
            images,
            predicted,
            observed,
            output["image_name"],
            output["patient_id"],
            output["source_gray_path"],
        ):
            generated_path = image_dir / f"{sequence:06d}_{Path(image_name).stem}.png"
            Image.fromarray(image).save(generated_path)
            rows.append(
                {
                    "split": split,
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "source_gray_path": source_path,
                    "generated_swe_path": str(generated_path),
                    "true_emean": float(target),
                    "predicted_emean": float(prediction),
                }
            )
            sequence += 1
    image_frame = pd.DataFrame(rows)
    patient_frame = (
        image_frame.groupby("patient_id", as_index=False)[
            ["true_emean", "predicted_emean"]
        ]
        .mean()
        .sort_values("patient_id", kind="stable")
    )
    image_csv = split_dir / "predicted_emean_image_level.csv"
    patient_csv = split_dir / "predicted_emean_patient_level.csv"
    image_frame.to_csv(image_csv, index=False, encoding="utf-8-sig")
    patient_frame.to_csv(patient_csv, index=False, encoding="utf-8-sig")
    if len(image_frame) != len(list(image_dir.glob("*.png"))):
        raise RuntimeError(split)
    return {
        "images": int(len(image_frame)),
        "patients": int(len(patient_frame)),
        "virtual_swe_directory": str(image_dir),
        "image_level_csv": str(image_csv),
        "patient_level_csv": str(patient_csv),
    }


if training:
    raise RuntimeError("This notebook only exports fixed-model predictions")
for path in [PAIR_CSV, GENERATOR_WEIGHT, EMEAN_WEIGHT]:
    if not path.is_file():
        raise FileNotFoundError(path)

frame = pd.read_csv(PAIR_CSV)
train_frame = frame.loc[frame["split"] == "train"].copy()
log_targets = np.log1p(train_frame["emean"].to_numpy(dtype=float))
target_mean = float(log_targets.mean())
target_std = float(log_targets.std(ddof=1))
if not np.isfinite(target_std) or target_std <= 0:
    raise ValueError("Invalid Emean standard deviation")

model = SplitExportLightning(target_mean, target_std)
load_weights(model)
reloaded = SplitExportLightning(target_mean, target_std)
load_weights(reloaded)
model.load_state_dict(reloaded.state_dict())
model.eval()

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    deterministic=True,
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    enable_progress_bar=False,
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
results = {}
for source_split, output_split in [
    ("train", "train"),
    ("val", "validation"),
    ("test", "internal_test"),
]:
    split_frame = frame.loc[frame["split"] == source_split].copy()
    predictions = trainer.predict(model, make_loader(split_frame))
    results[output_split] = save_split(predictions, output_split)

manifest = {
    "dataset": DATASET_NAME,
    "training": training,
    "generator_weight": str(GENERATOR_WEIGHT),
    "emean_weight": str(EMEAN_WEIGHT),
    "target_mean": target_mean,
    "target_std": target_std,
    "splits": results,
}
MANIFEST_PATH.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
manifest
