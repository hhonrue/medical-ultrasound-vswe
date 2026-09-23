import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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
RAW_ZIP = ROOT / "data" / "raw" / "肝脏.zip"
CACHE_DIR = ROOT / "experiment" / "liver" / "01_data_processing" / "cache"
EXTRACT_DIR = CACHE_DIR / "extracted"
GRAY_DIR = ROOT / "data" / "input" / "liver_cropped"
SWE_DIR = ROOT / "data" / "output" / "liver_cropped"
OUTPUT_CSV = CACHE_DIR / "pairs_all.csv"
IMAGE_SIZE = 256
BATCH_SIZE = 4
FRAME_PADDING = 12


def decode(content):
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image decode failed")
    return image


def member_by_suffix(archive, suffix):
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError((suffix, len(matches)))
    return matches[0]


def color_box(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 1], 50, 255)
    height, width = mask.shape
    yy, xx = np.indices(mask.shape)
    mask[(xx <= 70) | (xx >= width - 20) | (yy <= 80) | (yy >= height - 40)] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ValueError("color region not found")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, box_width, box_height, area = stats[component]
    if area < 3000:
        raise ValueError(("color region too small", int(area)))
    side = max(box_width, box_height) + FRAME_PADDING * 2
    center_x = x + box_width / 2
    center_y = y + box_height / 2
    left = int(round(center_x - side / 2))
    top = int(round(center_y - side / 2))
    left = max(0, min(left, width - side))
    top = max(0, min(top, height - side))
    return left, top, left + side, top + side


def crop(image, box):
    left, top, right, bottom = box
    return image[top:bottom, left:right]


def write_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(path)


def build_dataset():
    if not RAW_ZIP.is_file():
        raise FileNotFoundError(RAW_ZIP)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GRAY_DIR.mkdir(parents=True, exist_ok=True)
    SWE_DIR.mkdir(parents=True, exist_ok=True)
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True)
    with zipfile.ZipFile(RAW_ZIP) as archive:
        for info in archive.infolist():
            target = (EXTRACT_DIR / info.filename).resolve()
            if EXTRACT_DIR.resolve() not in target.parents and target != EXTRACT_DIR.resolve():
                raise RuntimeError(info.filename)
        archive.extractall(EXTRACT_DIR)
        labels_member = member_by_suffix(archive, "/肝脏117.xlsx")
        labels = pd.read_excel(archive.open(labels_member), sheet_name="Sheet1")
        rows = []
        for index, row in labels.reset_index(drop=True).iterrows():
            patient_id = str(row["ID"]).strip()
            image_name = f"_{index:04d}_{patient_id}.JPG.jpg"
            gray_member = member_by_suffix(archive, f"/肝灰阶/{image_name}")
            swe_member = member_by_suffix(archive, f"/肝弹性/{image_name}")
            gray = decode(archive.read(gray_member))
            swe = decode(archive.read(swe_member))
            swe_box = color_box(swe)
            gray_offset = gray.shape[1] - swe.shape[1]
            gray_box = (
                swe_box[0] + gray_offset,
                swe_box[1],
                swe_box[2] + gray_offset,
                swe_box[3],
            )
            if gray_box[0] < 0 or gray_box[2] > gray.shape[1] or gray_box[3] > gray.shape[0]:
                raise ValueError((image_name, gray.shape, swe.shape, gray_box, swe_box))
            gray_path = GRAY_DIR / f"{index:06d}.png"
            swe_path = SWE_DIR / f"{index:06d}.png"
            write_image(crop(gray, gray_box), gray_path)
            write_image(crop(swe, swe_box), swe_path)
            rows.append(
                {
                    "image_name": image_name,
                    "patient_id": patient_id,
                    "emean": float(row["Emean"]),
                    "emin": float(row["Emin"]),
                    "emax": float(row["Emax"]),
                    "original_split": "provided_all",
                    "gray_path": str(gray_path),
                    "swe_path": str(swe_path),
                    "split": "fit",
                    "alignment_mode": "already_cropped",
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 117 or frame["patient_id"].duplicated().any():
        raise RuntimeError((len(frame), int(frame["patient_id"].duplicated().sum())))
    frame.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    return frame


class PreviewDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row["gray_path"]) as gray_image:
            gray_image = gray_image.convert("RGB")
        with Image.open(row["swe_path"]) as swe_image:
            swe_image = swe_image.convert("RGB")
        gray_image = TF.resize(gray_image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
        swe_image = TF.resize(swe_image, [IMAGE_SIZE, IMAGE_SIZE], interpolation=InterpolationMode.BILINEAR, antialias=True)
        return {
            "gray": TF.normalize(TF.to_tensor(gray_image), [0.5] * 3, [0.5] * 3),
            "swe": TF.normalize(TF.to_tensor(swe_image), [0.5] * 3, [0.5] * 3),
            "emean": torch.tensor(float(row["emean"]), dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
        }


frame = build_dataset()
loader = DataLoader(PreviewDataset(frame), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
batch = next(iter(loader))
print("liver", len(frame), frame["original_split"].value_counts().to_dict())
print(batch["gray"].dtype, tuple(batch["gray"].shape), float(batch["gray"].min()), float(batch["gray"].max()))
print(batch["swe"].dtype, tuple(batch["swe"].shape), float(batch["swe"].min()), float(batch["swe"].max()))
print(batch["emean"].dtype, tuple(batch["emean"].shape), batch["emean"].tolist())
print({"keys": list(batch), "patient_ids": batch["patient_id"]})
