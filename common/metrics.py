"""指标实现（唯一来源）。

指标口径（写进论文时必须与此一致）：
- MAE / RMSE / MAPE / PSNR：在还原到 [0,1] 的裁剪图上逐像素计算。
  MAPE 分母有 1/255 的稳定下限；PSNR 在 MSE 极小时封顶 99，避免出现 inf。
- MS-SSIM：torchmetrics 实现，data_range=1。
- CoY：二值化（通道均值 > 0.5）后的 Yule's Q；另提供阈值 IoU 版本 coy_iou。
- inner_* ：先裁掉外圈 8% 再算同一套指标，作为边界区域的鲁棒性参考。
- FID：pytorch-fid 标准定义（dims=2048）。
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

# 内区指标裁掉的外圈比例（相对于图像短边）
INNER_MARGIN_RATIO = 0.08

PIXEL_COLUMNS = ["mae", "rmse", "mape", "psnr", "ms_ssim", "coy", "coy_iou"]
IMAGE_METRIC_COLUMNS = PIXEL_COLUMNS + [f"inner_{column}" for column in PIXEL_COLUMNS]


def denormalize(images):
    return torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)


def inner_crop(images, ratio=INNER_MARGIN_RATIO):
    """裁掉外圈 ratio 比例的边框；比例用短边计算，保证得到正方形内区。"""
    height, width = images.shape[-2:]
    margin = int(round(min(height, width) * ratio))
    if margin * 2 >= min(height, width):
        raise ValueError("inner_crop margin too large")
    return images[..., margin : height - margin, margin : width - margin]


def _yule_q(prediction, target, threshold=0.5):
    predicted = prediction.mean(axis=-1) > threshold
    observed = target.mean(axis=-1) > threshold
    a = np.logical_and(predicted, observed).sum(dtype=np.float64)
    b = np.logical_and(predicted, np.logical_not(observed)).sum(dtype=np.float64)
    c = np.logical_and(np.logical_not(predicted), observed).sum(dtype=np.float64)
    d = np.logical_and(np.logical_not(predicted), np.logical_not(observed)).sum(
        dtype=np.float64
    )
    denominator = a * d + b * c
    if denominator == 0:
        return 0.0
    return float((a * d - b * c) / denominator)


def _threshold_iou(prediction, target, threshold=0.5):
    predicted = prediction > threshold
    observed = target > threshold
    intersection = np.logical_and(predicted, observed).sum(dtype=np.float64)
    union = np.logical_or(predicted, observed).sum(dtype=np.float64)
    return 0.0 if union == 0 else float(intersection / union)


def _pixel_metrics(prediction, target, prefix=""):
    difference = prediction - target
    mse = float(np.mean(difference**2))
    return {
        f"{prefix}mae": float(np.mean(np.abs(difference))),
        f"{prefix}rmse": float(np.sqrt(mse)),
        f"{prefix}mape": float(
            np.mean(np.abs(difference) / np.maximum(np.abs(target), 1.0 / 255.0))
        ),
        f"{prefix}psnr": 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse)),
        f"{prefix}coy": _yule_q(prediction, target),
        f"{prefix}coy_iou": _threshold_iou(prediction, target),
    }


def _ms_ssim(prediction_tensor, target_tensor):
    return (
        multiscale_structural_similarity_index_measure(
            prediction_tensor,
            target_tensor,
            data_range=1.0,
            reduction="none",
        )
        .detach()
        .cpu()
        .numpy()
    )


def image_metric_frame(predictions, targets, image_names, patient_ids):
    prediction_tensor = denormalize(predictions).float().cpu()
    target_tensor = denormalize(targets).float().cpu()
    ms_ssim_values = _ms_ssim(prediction_tensor, target_tensor)
    inner_prediction_tensor = inner_crop(prediction_tensor)
    inner_target_tensor = inner_crop(target_tensor)
    inner_ms_ssim_values = _ms_ssim(inner_prediction_tensor, inner_target_tensor)

    prediction_array = prediction_tensor.permute(0, 2, 3, 1).numpy()
    target_array = target_tensor.permute(0, 2, 3, 1).numpy()
    inner_prediction_array = inner_prediction_tensor.permute(0, 2, 3, 1).numpy()
    inner_target_array = inner_target_tensor.permute(0, 2, 3, 1).numpy()
    rows = []
    for index, (image_name, patient_id) in enumerate(zip(image_names, patient_ids)):
        row = {"image_name": str(image_name), "patient_id": str(patient_id)}
        row.update(_pixel_metrics(prediction_array[index], target_array[index]))
        row["ms_ssim"] = float(ms_ssim_values[index])
        row.update(
            _pixel_metrics(
                inner_prediction_array[index],
                inner_target_array[index],
                prefix="inner_",
            )
        )
        row["inner_ms_ssim"] = float(inner_ms_ssim_values[index])
        rows.append(row)
    return pd.DataFrame(rows)


def _mean_sd(frame, columns):
    result = {}
    for column in columns:
        result[column] = {
            "mean": float(frame[column].mean()),
            "sd": float(frame[column].std(ddof=1)) if len(frame) > 1 else 0.0,
        }
    return result


def summarize_image_metrics(frame):
    patient_frame = frame.groupby("patient_id", as_index=False)[IMAGE_METRIC_COLUMNS].mean()
    return {
        "image_level": _mean_sd(frame, IMAGE_METRIC_COLUMNS),
        "patient_level": _mean_sd(patient_frame, IMAGE_METRIC_COLUMNS),
    }, patient_frame


def regression_frames(predictions, targets, image_names, patient_ids):
    image_frame = pd.DataFrame(
        {
            "image_name": list(map(str, image_names)),
            "patient_id": list(map(str, patient_ids)),
            "emean_true": np.asarray(targets, dtype=float),
            "emean_pred": np.asarray(predictions, dtype=float),
        }
    )
    patient_frame = (
        image_frame.groupby("patient_id", as_index=False)[["emean_true", "emean_pred"]]
        .mean()
        .reset_index(drop=True)
    )
    return image_frame, patient_frame


def _regression_summary(frame):
    true = frame["emean_true"].to_numpy(dtype=float)
    predicted = frame["emean_pred"].to_numpy(dtype=float)
    mse = mean_squared_error(true, predicted)
    return {
        "mse": float(mse),
        "mae": float(mean_absolute_error(true, predicted)),
        "rmse": float(np.sqrt(mse)),
        "r2": None if len(true) < 2 or np.var(true) == 0 else float(r2_score(true, predicted)),
    }


def summarize_regression(image_frame, patient_frame):
    return {
        "image_level": _regression_summary(image_frame),
        "patient_level": _regression_summary(patient_frame),
    }


def save_fid_images(predictions, targets, directory):
    directory = Path(directory)
    prediction_dir = directory / "predicted"
    target_dir = directory / "target"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in list(prediction_dir.glob("*.png")) + list(target_dir.glob("*.png")):
        path.unlink()
    prediction_array = (
        denormalize(predictions).cpu().mul(255).byte().permute(0, 2, 3, 1).numpy()
    )
    target_array = denormalize(targets).cpu().mul(255).byte().permute(0, 2, 3, 1).numpy()
    for index, (prediction, target) in enumerate(zip(prediction_array, target_array)):
        Image.fromarray(prediction).save(prediction_dir / f"{index:06d}.png")
        Image.fromarray(target).save(target_dir / f"{index:06d}.png")
    return prediction_dir, target_dir


FID_SPEC = {"implementation": "pytorch-fid", "dims": 2048, "scale_factor": 1.0}


def calculate_fid(predictions, targets, directory, device):
    """标准 FID（dims=2048，不做任何缩放）。需要可选依赖 pytorch-fid。"""
    from pytorch_fid import fid_score

    try:
        prediction_dir, target_dir = save_fid_images(predictions, targets, directory)
        return float(
            fid_score.calculate_fid_given_paths(
                [str(prediction_dir), str(target_dir)],
                batch_size=32,
                device=str(device),
                dims=FID_SPEC["dims"],
            )
        )
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def save_translation_grid(gray, predictions, targets, path, limit=6):
    gray_array = (
        denormalize(gray[:limit]).float().cpu().permute(0, 2, 3, 1).numpy()
    )
    prediction_array = (
        denormalize(predictions[:limit])
        .float()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    target_array = (
        denormalize(targets[:limit]).float().cpu().permute(0, 2, 3, 1).numpy()
    )
    rows = len(gray_array)
    figure, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows), squeeze=False)
    axes[0, 0].set_title("Grayscale")
    axes[0, 1].set_title("Virtual SWE")
    axes[0, 2].set_title("Real SWE")
    for index in range(rows):
        axes[index, 0].imshow(gray_array[index])
        axes[index, 1].imshow(prediction_array[index])
        axes[index, 2].imshow(target_array[index])
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_weight_only(model, path):
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save(state, path)


def load_weight_only(model, path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"权重文件不存在：{path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def dump_json(data, path):
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _demo():
    """自检：内区指标必须不看外圈；患者级聚合必须等于逐图指标的均值。"""
    prediction = torch.rand(2, 3, 256, 256) * 2 - 1
    target = prediction.clone()
    target[:, :, :12, :] = 1.0  # 只在外圈制造差异
    frame = image_metric_frame(prediction, target, ["a", "b"], ["p1", "p2"])
    assert frame["inner_mae"].max() < 1e-6, frame["inner_mae"].tolist()
    assert frame["mae"].min() > 0, frame["mae"].tolist()

    summary, patient_frame = summarize_image_metrics(frame)
    assert abs(summary["patient_level"]["mae"]["mean"] - float(patient_frame["mae"].mean())) < 1e-9
    assert "inner_ms_ssim" in frame.columns and "coy_iou" in frame.columns
    assert inner_crop(torch.zeros(1, 3, 256, 256)).shape[-2:] == (216, 216)
    print("metrics self-check ok：内区指标忽略外圈差异，患者级聚合正确")


if __name__ == "__main__":
    _demo()
