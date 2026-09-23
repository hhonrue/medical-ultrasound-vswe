import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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

from common.metrics import load_weight_only
from common.paths import PROSPECTIVE_GRAY_ROOT, ROOT
from common.models import EmeanRegressor, UnetGenerator
from common.preprocess import detect_roi_frame

NOTEBOOK_DIR = ROOT / "experiment" / "kidney" / "04_other"
DATA_ROOT = PROSPECTIVE_GRAY_ROOT
WEIGHT_ROOT = NOTEBOOK_DIR / "weights"
# 前瞻/肝脏使用窄通道轻量变体，与主实验（base=64）不是同一个模型
GENERATOR_BASE_CHANNELS = 16
EMEAN_HEAD_WIDTH = 256
EMEAN_DROPOUT = 0.2
GENERATOR_WEIGHT = WEIGHT_ROOT / "prospective_generator.pt"
EMEAN_WEIGHT = WEIGHT_ROOT / "prospective_emean_head.pt"
TARGET_MEAN_PATH = WEIGHT_ROOT / "target_mean.npy"
TARGET_STD_PATH = WEIGHT_ROOT / "target_std.npy"
OUTPUT_ROOT = NOTEBOOK_DIR / "prospective_grayscale_prediction_20260906"
MANIFEST_PATH = OUTPUT_ROOT / "prediction_manifest.json"
training = False
IMAGE_SIZE = 256
BATCH_SIZE = 16 if torch.cuda.is_available() else 4
NUM_WORKERS = 0
SEED = 42

pl.seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")
if not torch.cuda.is_available():
    torch.set_num_threads(min(2, os.cpu_count() or 1))


class ProspectiveGrayDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.paths = sorted(
            [
                path
                for path in self.directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ],
            key=lambda path: path.name.casefold(),
        )
        if not self.paths:
            raise ValueError(self.directory)
        self.stats = []
        stems = [path.stem.casefold() for path in self.paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate image stems in {self.directory}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            self._check_cropped_roi(image, path)
            image = TF.resize(
                image,
                [IMAGE_SIZE, IMAGE_SIZE],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image = TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)
        return {
            "gray": image,
            "image_name": path.name,
            "patient_id": path.stem,
            "source_gray_path": str(path),
        }


    # 前瞻队列的图片已经裁成 ROI（数据包目录名明确写着"已裁剪"），
    # 因此这里不能再做白框裁剪；只做校验，防止误把整屏截图喂进模型：
    # 未裁剪的原始超声截图中心会有一个 ROI 白框，裁好的 ROI 图不会。
    def _check_cropped_roi(self, image, path):
        if min(image.size) < 32:
            raise ValueError(f"图像过小，疑似不是 ROI 裁剪图：{path} {image.size}")
        array = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
        box = detect_roi_frame(array)
        if box is not None:
            raise ValueError(
                f"检测到 ROI 白框 {box}，疑似未裁剪的原始截图：{path}"
            )
        if len(self.stats) < 2000:
            self.stats.append(
                {
                    "image_name": path.name,
                    "width": int(image.size[0]),
                    "height": int(image.size[1]),
                    "roi_frame_detected": False,
                }
            )


def make_loader(directory):
    return DataLoader(
        ProspectiveGrayDataset(directory),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


class PredictionLightning(pl.LightningModule):
    def __init__(self, generator, regressor, target_mean, target_std):
        super().__init__()
        self.generator = generator
        self.regressor = regressor
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    def forward(self, inputs):
        virtual_swe = self.generator(inputs)
        standardized = self.regressor(virtual_swe)
        emean = torch.expm1(
            standardized * self.target_std + self.target_mean
        ).clamp_min(0.0)
        return virtual_swe, emean

    def predict_step(self, batch, batch_index, dataloader_index=0):
        virtual_swe, predicted_emean = self(batch["gray"])
        return {
            "virtual_swe": virtual_swe,
            "predicted_emean": predicted_emean,
            "image_name": batch["image_name"],
            "patient_id": batch["patient_id"],
            "source_gray_path": batch["source_gray_path"],
        }


def clear_output():
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def save_cohort(model, cohort, device):
    input_dir = DATA_ROOT / cohort
    output_dir = OUTPUT_ROOT / cohort
    image_dir = output_dir / "predicted_swe"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    loader = make_loader(input_dir)
    total = len(loader.dataset)
    for batch_index, batch in enumerate(loader, start=1):
        gray = batch["gray"].to(device, non_blocking=torch.cuda.is_available())
        batch["gray"] = gray
        with torch.inference_mode():
            prediction = model.predict_step(batch, batch_index)
        virtual_swe = prediction["virtual_swe"]
        predicted_emean = prediction["predicted_emean"]
        images = (
            torch.clamp((virtual_swe + 1.0) / 2.0, 0.0, 1.0)
            .mul(255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )
        values = predicted_emean.cpu().numpy()
        for image, value, image_name, patient_id, source_path in zip(
            images,
            values,
            batch["image_name"],
            batch["patient_id"],
            batch["source_gray_path"],
        ):
            generated_path = image_dir / f"{Path(image_name).stem}.png"
            Image.fromarray(image).save(generated_path)
            rows.append(
                {
                    "cohort": cohort,
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "source_gray_path": source_path,
                    "generated_swe_path": str(generated_path),
                    "predicted_emean": float(value),
                }
            )
        if batch_index == 1 or batch_index % 50 == 0 or batch_index == len(loader):
            print(f"{cohort}: {min(batch_index * BATCH_SIZE, total)}/{total}", flush=True)
        del gray, virtual_swe, predicted_emean, images, values

    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "predicted_emean.csv", index=False, encoding="utf-8-sig")
    generated_count = len(list(image_dir.glob("*.png")))
    if len(result) != total or generated_count != total:
        raise RuntimeError((cohort, total, len(result), generated_count))
    checked = loader.dataset.stats
    widths = [row["width"] for row in checked]
    heights = [row["height"] for row in checked]
    return {
        "input_check": {
            "assume_cropped": True,
            "checked_images": len(checked),
            "resolution_median": [
                int(np.median(widths)) if widths else None,
                int(np.median(heights)) if heights else None,
            ],
            "roi_frames_detected": sum(
                1 for row in checked if row["roi_frame_detected"]
            ),
        },
        "input_images": total,
        "generated_images": generated_count,
        "prediction_rows": len(result),
        "prediction_min": float(result["predicted_emean"].min()),
        "prediction_max": float(result["predicted_emean"].max()),
        "prediction_mean": float(result["predicted_emean"].mean()),
        "prediction_csv": str(output_dir / "predicted_emean.csv"),
        "predicted_swe_directory": str(image_dir),
    }


def main():
    if training:
        raise RuntimeError("Prospective grayscale data have no training labels")
    required_paths = [
        GENERATOR_WEIGHT,
        EMEAN_WEIGHT,
        TARGET_MEAN_PATH,
        TARGET_STD_PATH,
        DATA_ROOT / "internal",
        DATA_ROOT / "external",
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    target_mean = float(np.load(TARGET_MEAN_PATH))
    target_std = float(np.load(TARGET_STD_PATH))
    if not np.isfinite(target_mean) or not np.isfinite(target_std) or target_std <= 0:
        raise ValueError((target_mean, target_std))

    generator = load_weight_only(UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS), GENERATOR_WEIGHT)
    regressor = load_weight_only(EmeanRegressor(head_width=EMEAN_HEAD_WIDTH, dropout=EMEAN_DROPOUT), EMEAN_WEIGHT)
    model = PredictionLightning(generator, regressor, target_mean, target_std)
    reloaded = PredictionLightning(
        load_weight_only(UnetGenerator(base_channels=GENERATOR_BASE_CHANNELS), GENERATOR_WEIGHT),
        load_weight_only(EmeanRegressor(head_width=EMEAN_HEAD_WIDTH, dropout=EMEAN_DROPOUT), EMEAN_WEIGHT),
        target_mean,
        target_std,
    )
    model.load_state_dict(reloaded.state_dict(), strict=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    clear_output()
    print(f"device: {device}", flush=True)
    print(f"batch_size: {BATCH_SIZE}", flush=True)
    cohort_results = {
        cohort: save_cohort(model, cohort, device)
        for cohort in ["internal", "external"]
    }
    manifest = {
        "data_type": "prospective grayscale-only inference",
        "training": training,
        "source_data_root": str(DATA_ROOT),
        "generator_weight": str(GENERATOR_WEIGHT),
        "emean_weight": str(EMEAN_WEIGHT),
        "target_mean_path": str(TARGET_MEAN_PATH),
        "target_std_path": str(TARGET_STD_PATH),
        "target_mean": target_mean,
        "target_std": target_std,
        "image_size": IMAGE_SIZE,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "input_check": {
            cohort: result["input_check"] for cohort, result in cohort_results.items()
        },
        "has_real_swe": False,
        "has_real_emean": False,
        "cohorts": cohort_results,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
