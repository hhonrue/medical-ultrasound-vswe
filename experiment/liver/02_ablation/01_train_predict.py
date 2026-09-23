import gc
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.models import resnet18
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
from common.models import (
    EmeanRegressor,
    PatchDiscriminator,
    UnetGenerator,
    initialize_map2sat,
)
from common.train import EpochRecorder
ORGAN = "liver"
NOTEBOOK_DIR = ROOT / "experiment" / ORGAN / "02_ablation"
CACHE_CSV = ROOT / "experiment" / ORGAN / "01_data_processing" / "cache" / "pairs_all.csv"
PRETRAINED_PATH = ROOT / "reference" / "baseline" / "code" / "pretrained_models" / "map2sat.pth"
OUTPUT_DIR = NOTEBOOK_DIR / "outputs"
WEIGHT_DIR = NOTEBOOK_DIR / "weights"
training = False
DATA_MODE = "supplementary"
SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 0
PRETRAINED_EPOCHS = 300
RANDOM_EPOCHS = 20
FAIR_EPOCH = 20
EMEAN_EPOCHS = 120
GENERATOR_BASE_CHANNELS = 16
DISCRIMINATOR_BASE_CHANNELS = 32
EMEAN_HEAD_WIDTH = 256
EMEAN_DROPOUT = 0.2
GENERATOR_LEARNING_RATE = 1e-4
EMEAN_LEARNING_RATE = 5e-4
RECONSTRUCTION_WEIGHT = 100.0
GRADIENT_WEIGHT = 10.0
SSIM_WEIGHT = 10.0

pl.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")

class PairedDataset(Dataset):
    def __init__(self, frame, augment=False, preload=False):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.pairs = [self._load_pair(row) for _, row in self.frame.iterrows()] if preload else None

    def __len__(self):
        return len(self.frame)

    def _load_pair(self, row):
        with Image.open(row["gray_path"]) as image:
            gray = image.convert("RGB")
        with Image.open(row["swe_path"]) as image:
            swe = image.convert("RGB")
        mode = str(row["alignment_mode"])
        if mode == "right_crop_gray":
            common_height = min(gray.height, swe.height)
            common_width = min(gray.width, swe.width)
            gray = gray.crop((gray.width - common_width, 0, gray.width, common_height))
            swe = swe.crop((0, 0, common_width, common_height))
        else:
            common_height = min(gray.height, swe.height)
            common_width = min(gray.width, swe.width)
            gray_left = (gray.width - common_width) // 2
            gray_top = (gray.height - common_height) // 2
            swe_left = (swe.width - common_width) // 2
            swe_top = (swe.height - common_height) // 2
            gray = gray.crop((gray_left, gray_top, gray_left + common_width, gray_top + common_height))
            swe = swe.crop((swe_left, swe_top, swe_left + common_width, swe_top + common_height))
        gray = TF.resize(gray, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
        swe = TF.resize(swe, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
        gray = TF.normalize(TF.to_tensor(gray), [0.5] * 3, [0.5] * 3)
        swe = TF.normalize(TF.to_tensor(swe), [0.5] * 3, [0.5] * 3)
        return gray, swe

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        gray, swe = self.pairs[index] if self.pairs is not None else self._load_pair(row)
        if self.augment and torch.rand(()) < 0.5:
            gray = TF.hflip(gray)
            swe = TF.hflip(swe)
        return {"gray": gray, "swe": swe, "emean": torch.tensor(float(row["emean"]), dtype=torch.float32), "image_name": str(row["image_name"]), "patient_id": str(row["patient_id"]), "original_split": str(row["original_split"])}


def make_loader(frame, shuffle, augment, batch_size=BATCH_SIZE):
    return DataLoader(PairedDataset(frame, augment=augment, preload=False), batch_size=batch_size, shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), persistent_workers=NUM_WORKERS > 0)

def gradient_loss(prediction, target):
    prediction_x = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    target_x = target[:, :, :, 1:] - target[:, :, :, :-1]
    prediction_y = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    target_y = target[:, :, 1:, :] - target[:, :, :-1, :]
    return torch.mean(torch.abs(prediction_x - target_x)) + torch.mean(torch.abs(prediction_y - target_y))


class TranslationLightning(pl.LightningModule):
    def __init__(self, learning_rate=GENERATOR_LEARNING_RATE):
        super().__init__()
        self.automatic_optimization = False
        self.generator = UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS)
        self.discriminator = PatchDiscriminator(base_channels=DISCRIMINATOR_BASE_CHANNELS)
        self.learning_rate = learning_rate
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()

    def forward(self, inputs):
        return self.generator(inputs)

    def training_step(self, batch, batch_index):
        generator_optimizer, discriminator_optimizer = self.optimizers()
        gray = batch["gray"]
        target = batch["swe"]
        generated = self(gray)
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(True)
        real_logits = self.discriminator(gray, target)
        fake_logits = self.discriminator(gray, generated.detach())
        discriminator_loss = 0.5 * (self.adversarial_loss(real_logits, torch.ones_like(real_logits)) + self.adversarial_loss(fake_logits, torch.zeros_like(fake_logits)))
        discriminator_optimizer.zero_grad()
        self.manual_backward(discriminator_loss)
        discriminator_optimizer.step()
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(False)
        fake_logits = self.discriminator(gray, generated)
        adversarial = self.adversarial_loss(fake_logits, torch.ones_like(fake_logits))
        reconstruction = self.reconstruction_loss(generated, target)
        edges = gradient_loss(generated, target)
        ssim = structural_similarity_index_measure(torch.clamp((generated + 1) / 2, 0, 1), torch.clamp((target + 1) / 2, 0, 1), data_range=1.0)
        generator_loss = adversarial + RECONSTRUCTION_WEIGHT * reconstruction + GRADIENT_WEIGHT * edges + SSIM_WEIGHT * (1 - ssim)
        generator_optimizer.zero_grad()
        self.manual_backward(generator_loss)
        generator_optimizer.step()
        for parameter in self.discriminator.parameters():
            parameter.requires_grad_(True)
        self.log("train_generator_loss", generator_loss, on_epoch=True, prog_bar=True)
        self.log("train_discriminator_loss", discriminator_loss, on_epoch=True)
        self.log("train_l1", reconstruction, on_epoch=True, prog_bar=True)
        self.log("train_ssim", ssim, on_epoch=True)
        return generator_loss

    def test_step(self, batch, batch_index):
        generated = self(batch["gray"])
        l1 = self.reconstruction_loss(generated, batch["swe"])
        self.log("test_l1", l1, on_epoch=True)
        return l1

    def predict_step(self, batch, batch_index, dataloader_index=0):
        generated = self(batch["gray"])
        gray = torch.clamp((batch["gray"] + 1) / 2, 0, 1).mul(255).byte().cpu()
        prediction = torch.clamp((generated + 1) / 2, 0, 1).mul(255).byte().cpu()
        target = torch.clamp((batch["swe"] + 1) / 2, 0, 1).mul(255).byte().cpu()
        return {"gray": gray, "prediction": prediction, "target": target, "emean": batch["emean"].detach().cpu(), "image_name": list(batch["image_name"]), "patient_id": list(batch["patient_id"]), "original_split": list(batch["original_split"])}

    def configure_optimizers(self):
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        return [generator_optimizer, discriminator_optimizer]

def save_half_state_dict(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, path)


def load_state_dict(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(name, str) and torch.is_tensor(value) for name, value in state.items()):
        raise RuntimeError(path)
    model.load_state_dict(state, strict=True)
    return model


def validate_weight_file(path, expected):
    state = torch.load(path, map_location="cpu", weights_only=True)
    signature = {name: tuple(value.shape) for name, value in state.items()}
    if signature != expected:
        raise RuntimeError(path)
    del state


class WeightSnapshotCallback(pl.Callback):
    """按固定 epoch 保存生成器权重（epoch 曲线由 common.train.EpochRecorder 负责）。"""

    def __init__(self, condition):
        super().__init__()
        self.condition = condition

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        wanted = (self.condition == "pretrained" and epoch in {FAIR_EPOCH, PRETRAINED_EPOCHS}) or (
            self.condition == "random_init" and epoch == RANDOM_EPOCHS
        )
        if wanted:
            save_half_state_dict(
                pl_module.generator,
                WEIGHT_DIR / f"{self.condition}_epoch_{epoch:03d}.pt",
            )


def trainer_for(epochs, callbacks=None):
    return pl.Trainer(max_epochs=epochs, accelerator="auto", devices=1, precision="16-mixed" if torch.cuda.is_available() else "32-true", deterministic=False, logger=False, enable_checkpointing=False, enable_model_summary=False, enable_progress_bar=False, callbacks=callbacks or [], num_sanity_val_steps=0)


def collect(outputs):
    return {
        "gray": torch.cat([item["gray"] for item in outputs]),
        "prediction": torch.cat([item["prediction"] for item in outputs]),
        "target": torch.cat([item["target"] for item in outputs]),
        "emean": torch.cat([item["emean"] for item in outputs]),
        "image_name": [value for item in outputs for value in item["image_name"]],
        "patient_id": [value for item in outputs for value in item["patient_id"]],
        "original_split": [value for item in outputs for value in item["original_split"]],
    }


def yule_q(prediction, target, threshold=0.5):
    predicted = prediction.mean(axis=-1) > threshold
    observed = target.mean(axis=-1) > threshold
    a = np.logical_and(predicted, observed).sum(dtype=np.float64)
    b = np.logical_and(predicted, np.logical_not(observed)).sum(dtype=np.float64)
    c = np.logical_and(np.logical_not(predicted), observed).sum(dtype=np.float64)
    d = np.logical_and(np.logical_not(predicted), np.logical_not(observed)).sum(dtype=np.float64)
    denominator = a * d + b * c
    return 0.0 if denominator == 0 else float((a * d - b * c) / denominator)


def image_metrics(collected):
    rows = []
    for start in range(0, len(collected["prediction"]), 16):
        prediction = collected["prediction"][start:start + 16].float().div(255)
        target = collected["target"][start:start + 16].float().div(255)
        ms_ssim = multiscale_structural_similarity_index_measure(prediction, target, data_range=1.0, reduction="none").numpy()
        prediction_array = prediction.permute(0, 2, 3, 1).numpy()
        target_array = target.permute(0, 2, 3, 1).numpy()
        for offset, (predicted, observed) in enumerate(zip(prediction_array, target_array)):
            index = start + offset
            difference = predicted - observed
            mse = float(np.mean(difference ** 2))
            rows.append({"image_name": collected["image_name"][index], "patient_id": collected["patient_id"][index], "original_split": collected["original_split"][index], "mae": float(np.mean(np.abs(difference))), "rmse": float(np.sqrt(mse)), "mape": float(np.mean(np.abs(difference) / np.maximum(np.abs(observed), 1.0 / 255.0))), "psnr": 99.0 if mse <= 1e-12 else float(10 * np.log10(1 / mse)), "ms_ssim": float(ms_ssim[offset]), "coy": yule_q(predicted, observed)})
    return pd.DataFrame(rows)


def save_fid_images(collected, directory):
    prediction_dir = directory / "prediction"
    target_dir = directory / "target"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    prediction = collected["prediction"].permute(0, 2, 3, 1).numpy()
    target = collected["target"].permute(0, 2, 3, 1).numpy()
    for index, (predicted, observed) in enumerate(zip(prediction, target)):
        Image.fromarray(predicted).save(prediction_dir / f"{index:06d}.png")
        Image.fromarray(observed).save(target_dir / f"{index:06d}.png")
    return prediction_dir, target_dir


def calculate_fid(collected, name):
    from pytorch_fid import fid_score
    directory = OUTPUT_DIR / f"fid_{name}"
    try:
        prediction_dir, target_dir = save_fid_images(collected, directory)
        return float(fid_score.calculate_fid_given_paths([str(prediction_dir), str(target_dir)], batch_size=32, device="cuda:0" if torch.cuda.is_available() else "cpu", dims=2048))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def evaluate_weight(frame, weight_path, name, save_all_images=False, fid_value=None):
    model = TranslationLightning()
    load_state_dict(model.generator, weight_path)
    reloaded = UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS)
    load_state_dict(reloaded, weight_path)
    model.generator.load_state_dict(reloaded.state_dict(), strict=True)
    evaluator = trainer_for(1)
    loader = make_loader(frame, False, False)
    evaluator.test(model, loader)
    collected = collect(evaluator.predict(model, loader))
    metrics = image_metrics(collected)
    metrics.to_csv(OUTPUT_DIR / f"{name}_image_metrics.csv", index=False, encoding="utf-8-sig")
    patient_metrics = metrics.groupby("patient_id", as_index=False)[["mae", "rmse", "mape", "psnr", "ms_ssim", "coy"]].mean()
    patient_metrics.to_csv(OUTPUT_DIR / f"{name}_patient_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {metric: float(metrics[metric].mean()) for metric in ["mae", "rmse", "mape", "psnr", "ms_ssim", "coy"]}
    summary["fid"] = calculate_fid(collected, name) if fid_value is None else float(fid_value)
    if save_all_images:
        image_dir = OUTPUT_DIR / "final_pretrained_virtual_swe"
        image_dir.mkdir(parents=True, exist_ok=True)
        images = collected["prediction"].permute(0, 2, 3, 1).numpy()
        rows = []
        for index, image in enumerate(images):
            path = image_dir / f"{index:06d}_{Path(collected['image_name'][index]).stem}.png"
            Image.fromarray(image).save(path)
            rows.append({"image_name": collected["image_name"][index], "patient_id": collected["patient_id"][index], "original_split": collected["original_split"][index], "generated_swe_path": str(path)})
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "final_pretrained_generated_files.csv", index=False, encoding="utf-8-sig")
    preview = {
        "gray": collected["gray"][:8],
        "prediction": collected["prediction"][:8],
        "target": collected["target"][:8],
    }
    del collected, model, evaluator, loader
    gc.collect()
    torch.cuda.empty_cache()
    return summary, preview


def save_comparison_grid(collections, path, limit=8):
    names = ["gray", "pretrained_020", "random_020", "pretrained_300", "real_swe"]
    arrays = [
        collections["pretrained_epoch_020"]["gray"][:limit].float().div(255),
        collections["pretrained_epoch_020"]["prediction"][:limit].float().div(255),
        collections["random_init_epoch_020"]["prediction"][:limit].float().div(255),
        collections["pretrained_epoch_300"]["prediction"][:limit].float().div(255),
        collections["pretrained_epoch_300"]["target"][:limit].float().div(255),
    ]
    rows = min(limit, len(arrays[0]))
    figure, axes = plt.subplots(rows, len(names), figsize=(15, 3 * rows), squeeze=False)
    for column, name in enumerate(names):
        axes[0, column].set_title(name)
    for row in range(rows):
        for column, array in enumerate(arrays):
            axes[row, column].imshow(array[row].permute(1, 2, 0).numpy())
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)

class EmeanLightning(pl.LightningModule):
    def __init__(self, generator, target_mean, target_std):
        super().__init__()
        self.generator = generator.eval()
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
        self.regressor = EmeanRegressor(head_width=EMEAN_HEAD_WIDTH, dropout=EMEAN_DROPOUT)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))
        self.loss_function = nn.HuberLoss(delta=1.0)

    def on_train_epoch_start(self):
        self.generator.eval()

    def standardized(self, batch):
        with torch.no_grad():
            virtual_swe = self.generator(batch["gray"])
        return self.regressor(virtual_swe)

    def forward(self, batch):
        return torch.expm1(self.standardized(batch) * self.target_std + self.target_mean).clamp_min(0)

    def training_step(self, batch, batch_index):
        target = (torch.log1p(batch["emean"]) - self.target_mean) / self.target_std
        loss = self.loss_function(self.standardized(batch), target)
        self.log("train_emean_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_index):
        prediction = self(batch)
        mae = torch.mean(torch.abs(prediction - batch["emean"]))
        self.log("test_emean_mae", mae, on_epoch=True)
        return mae

    def predict_step(self, batch, batch_index, dataloader_index=0):
        return {"prediction": self(batch).detach().cpu(), "target": batch["emean"].detach().cpu(), "image_name": list(batch["image_name"]), "patient_id": list(batch["patient_id"]), "original_split": list(batch["original_split"])}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.regressor.parameters(), lr=EMEAN_LEARNING_RATE, weight_decay=1e-4)


if not CACHE_CSV.is_file() or not PRETRAINED_PATH.is_file():
    raise FileNotFoundError((CACHE_CSV, PRETRAINED_PATH))
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
frame = pd.read_csv(CACHE_CSV)
conditions = [("pretrained", PRETRAINED_EPOCHS, True), ("random_init", RANDOM_EPOCHS, False)]
needs_training = training and any(not (WEIGHT_DIR / f"{condition}_epoch_{epochs:03d}.pt").is_file() for condition, epochs, _ in conditions)
fit_loader = make_loader(frame, True, True) if needs_training else None
training_manifest = []
expected_generator_signature = {name: tuple(value.shape) for name, value in UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS).state_dict().items()}
for condition, epochs, use_pretrained in conditions:
    pl.seed_everything(SEED, workers=True)
    model = TranslationLightning()
    init_report = (
        initialize_map2sat(model.generator, PRETRAINED_PATH, allow_slice=True)
        if use_pretrained
        else {"loaded": 0, "sliced": False}
    )
    recorder = EpochRecorder(
        OUTPUT_DIR / f"{condition}_epoch_metrics.csv",
        keys=[
            "train_generator_loss",
            "train_discriminator_loss",
            "train_l1",
            "train_ssim",
        ],
        when="train",
    )
    trainer = trainer_for(epochs, [recorder, WeightSnapshotCallback(condition)])
    final_path = WEIGHT_DIR / f"{condition}_epoch_{epochs:03d}.pt"
    if training and not final_path.is_file():
        trainer.fit(model, fit_loader)
        if not final_path.is_file():
            save_half_state_dict(model.generator, final_path)
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    load_state_dict(model.generator, final_path)
    reloaded = UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS)
    load_state_dict(reloaded, final_path)
    required_paths = [final_path] if condition == "random_init" else [WEIGHT_DIR / f"pretrained_epoch_{FAIR_EPOCH:03d}.pt", final_path]
    for saved_path in required_paths:
        validate_weight_file(saved_path, expected_generator_signature)
    training_manifest.append({"condition": condition, "epochs": epochs, "map2sat_init": init_report, "final_weight": str(final_path), "saved_epoch_weights": len(required_paths)})

del trainer, recorder, model, reloaded
if fit_loader is not None:
    del fit_loader
gc.collect()
torch.cuda.empty_cache()
evaluation_loader = make_loader(frame, False, False)

comparison_specs = [
    ("pretrained_epoch_020", WEIGHT_DIR / "pretrained_epoch_020.pt", False),
    ("random_init_epoch_020", WEIGHT_DIR / "random_init_epoch_020.pt", False),
    ("pretrained_epoch_300", WEIGHT_DIR / "pretrained_epoch_300.pt", True),
]
comparison_summary_path = OUTPUT_DIR / "comparison_summary.csv"
comparison_image_path = OUTPUT_DIR / "comparison_examples.png"
existing_summary = pd.read_csv(comparison_summary_path) if comparison_summary_path.is_file() else pd.DataFrame()
if not existing_summary.empty and "mape" in existing_summary.columns and comparison_image_path.is_file():
    summary_frame = pd.read_csv(comparison_summary_path)
else:
    existing_fids = dict(zip(existing_summary["condition"], existing_summary["fid"])) if not existing_summary.empty and "fid" in existing_summary.columns else {}
    summary_rows = []
    collections = {}
    for name, path, save_all_images in comparison_specs:
        summary, collected = evaluate_weight(frame, path, name, save_all_images, existing_fids.get(name))
        collections[name] = collected
        summary_rows.append({"condition": name, **summary})
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(comparison_summary_path, index=False, encoding="utf-8-sig")
    save_comparison_grid(collections, comparison_image_path)

final_generator = UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS)
load_state_dict(final_generator, WEIGHT_DIR / "pretrained_epoch_300.pt")
log_targets = np.log1p(frame["emean"].to_numpy(dtype=float))
target_mean = float(log_targets.mean())
target_std = float(log_targets.std(ddof=1))
emean_model = EmeanLightning(final_generator, target_mean, target_std)
emean_weight = WEIGHT_DIR / "final_pretrained_emean_head.pt"
if training and not emean_weight.is_file():
    emean_recorder = EpochRecorder(
        OUTPUT_DIR / "emean_epoch_metrics.csv",
        keys=["train_emean_loss"],
        when="train",
    )
    emean_trainer = trainer_for(EMEAN_EPOCHS, [emean_recorder])
    emean_trainer.fit(emean_model, make_loader(frame, True, True, batch_size=32))
    save_half_state_dict(emean_model.regressor, emean_weight)
else:
    emean_trainer = trainer_for(1)
if not emean_weight.is_file():
    raise FileNotFoundError(emean_weight)
emean_epoch_weights = sorted(WEIGHT_DIR.glob("emean_epoch_*.pt"))
expected_emean_signature = {name: tuple(value.shape) for name, value in EmeanRegressor(head_width=EMEAN_HEAD_WIDTH, dropout=EMEAN_DROPOUT).state_dict().items()}
validate_weight_file(emean_weight, expected_emean_signature)
load_state_dict(emean_model.regressor, emean_weight)
reloaded_regressor = EmeanRegressor(head_width=EMEAN_HEAD_WIDTH, dropout=EMEAN_DROPOUT)
load_state_dict(reloaded_regressor, emean_weight)
emean_model.regressor.load_state_dict(reloaded_regressor.state_dict(), strict=True)
emean_trainer.test(emean_model, evaluation_loader)
emean_outputs = emean_trainer.predict(emean_model, evaluation_loader)
predictions = torch.cat([item["prediction"] for item in emean_outputs]).numpy()
targets = torch.cat([item["target"] for item in emean_outputs]).numpy()
image_names = [value for item in emean_outputs for value in item["image_name"]]
patient_ids = [value for item in emean_outputs for value in item["patient_id"]]
original_splits = [value for item in emean_outputs for value in item["original_split"]]
emean_frame = pd.DataFrame({"image_name": image_names, "patient_id": patient_ids, "original_split": original_splits, "emean_true": targets, "emean_pred": predictions})
emean_frame.to_csv(OUTPUT_DIR / "final_pretrained_emean_predictions.csv", index=False, encoding="utf-8-sig")
patient_frame = emean_frame.groupby("patient_id", as_index=False)[["emean_true", "emean_pred"]].mean()
patient_frame.to_csv(OUTPUT_DIR / "final_pretrained_emean_patient_predictions.csv", index=False, encoding="utf-8-sig")
def regression_summary(values):
    true = values["emean_true"].to_numpy()
    predicted = values["emean_pred"].to_numpy()
    mse = mean_squared_error(true, predicted)
    return {"mae": float(mean_absolute_error(true, predicted)), "rmse": float(np.sqrt(mse)), "r2": float(r2_score(true, predicted)) if len(true) > 1 and np.var(true) > 0 else None}
emean_summary = {"image_level": regression_summary(emean_frame), "patient_level": regression_summary(patient_frame)}

lower = ["mae", "rmse", "mape", "fid"]
higher = ["psnr", "ms_ssim", "coy"]
pre50 = summary_frame.loc[summary_frame["condition"] == "pretrained_epoch_020"].iloc[0]
random50 = summary_frame.loc[summary_frame["condition"] == "random_init_epoch_020"].iloc[0]
pre300 = summary_frame.loc[summary_frame["condition"] == "pretrained_epoch_300"].iloc[0]
fair_wins = sum(pre50[key] < random50[key] for key in lower) + sum(pre50[key] > random50[key] for key in higher)
long_wins = sum(pre300[key] < random50[key] for key in lower) + sum(pre300[key] > random50[key] for key in higher)
manifest = {"organ": ORGAN, "data_mode": DATA_MODE, "fit_images": int(len(frame)), "seed": SEED, "training_manifest": training_manifest, "fair_comparison": {"epoch": FAIR_EPOCH, "pretrained_metric_wins": int(fair_wins), "metrics_total": 7}, "unequal_budget_comparison": {"pretrained_epochs": PRETRAINED_EPOCHS, "random_epochs": RANDOM_EPOCHS, "pretrained_metric_wins": int(long_wins), "metrics_total": 7}, "emean": {**emean_summary, "saved_epoch_weights": len(emean_epoch_weights)}, "final_generator_weight": str(WEIGHT_DIR / "pretrained_epoch_300.pt"), "final_emean_weight": str(emean_weight)}
(OUTPUT_DIR / "experiment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
