import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class PairedElasticityDataset(Dataset):
    def __init__(self, frame, augment=False, image_size=256):
        self.frame = frame.reset_index(drop=True)
        self.augment = augment
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def _tensor(self, path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            return TF.normalize(TF.to_tensor(image), [0.5] * 3, [0.5] * 3)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        gray = self._tensor(row["gray_path"])
        swe = self._tensor(row["swe_path"])
        if self.augment and torch.rand(()) < 0.5:
            gray = TF.hflip(gray)
            swe = TF.hflip(swe)
        return {
            "gray": gray,
            "swe": swe,
            "emean": torch.tensor(float(row["emean"]), dtype=torch.float32),
            "image_name": str(row["image_name"]),
            "patient_id": str(row["patient_id"]),
            "split": str(row["split"]),
        }


def load_frame(path, split=None):
    frame = pd.read_csv(path)
    if split is not None:
        frame = frame.loc[frame["split"] == split].copy()
    return frame


def make_loader(frame, batch_size, shuffle, augment, num_workers=4):
    if frame is None or frame.empty:
        raise ValueError("Cannot build a DataLoader from an empty cohort")
    dataset = PairedElasticityDataset(frame, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
