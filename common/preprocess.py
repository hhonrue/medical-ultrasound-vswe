"""原始 ZIP -> 配对裁剪图 + 患者级划分。

裁剪口径：
1. 配对对齐。灰阶与弹性的画布尺寸通常不同（例如 438x291 vs 462x291）。
   以灰阶为参考坐标系，用画布宽高差把弹性框映射过来：差异在
   `box_tolerance` 内就共用同一个框；超出则取两框交集，并把
   `crop_aligned=False` 记进 pairs CSV / crop_mismatch.csv。
2. ROI 内缩。检测框向内缩进短边的 `inset_ratio`（默认 2%）后再裁剪，
   降低边界干扰。`common/metrics.py` 另提供 `inner_*` 指标
   （裁掉外圈 8% 再算），可作为边界鲁棒性参考。
"""

from io import BytesIO
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile

import cv2
import numpy as np
import pandas as pd

from common.split import assign_patient_split, normalized_stem, patient_id_from_name


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# 裁剪内缩比例：短边的 2%（见模块 docstring 第 2 条）
DEFAULT_INSET_RATIO = 0.02
DEFAULT_BOX_TOLERANCE = 4
# 判定"该行/列属于白框线"的阈值（白框实测 200~226，非纯白）
FRAME_GRAY_THRESHOLD = 200


def _member_by_suffix(archive, suffix):
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one member ending with {suffix}, found {len(matches)}")
    return matches[0]


def _image_members(archive, marker):
    result = {}
    duplicates = set()
    for name in archive.namelist():
        if marker in name and name.lower().endswith(IMAGE_SUFFIXES):
            key = normalized_stem(PurePosixPath(name).name)
            if key in result:
                duplicates.add(key)
            else:
                result[key] = name
    for key in duplicates:
        result.pop(key, None)
    return result, duplicates


def _read_labels(archive, workbook_suffix):
    member = _member_by_suffix(archive, workbook_suffix)
    frame = pd.read_excel(BytesIO(archive.read(member)), sheet_name=0)
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    if "id" not in lookup or "emean" not in lookup:
        raise ValueError(f"Workbook {member} must contain ID and Emean")
    frame = frame[[lookup["id"], lookup["emean"]]].copy()
    frame.columns = ["image_name", "emean"]
    frame = frame.dropna(subset=["image_name", "emean"])
    frame["match_key"] = frame["image_name"].map(normalized_stem)
    duplicates = set(
        frame.loc[frame["match_key"].duplicated(keep=False), "match_key"].tolist()
    )
    return frame, duplicates


def _decode(content):
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image decode failed")
    return image


def _white_box(image, threshold=200):
    """返回图像中央白色矩形（ROI 边框）的外接矩形，不做内缩。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, threshold, 255)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    image_area = float(height * width)
    center = np.asarray([width / 2.0, height / 2.0])
    candidates = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        rectangle_area = float(box_width * box_height)
        ratio = rectangle_area / image_area
        aspect_ratio = box_width / box_height
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
        rotated_area = float(rotated_width * rotated_height)
        contour_fill = cv2.contourArea(contour) / max(rotated_area, 1.0)
        rectangle_center = np.asarray([x + box_width / 2.0, y + box_height / 2.0])
        normalized_center = rectangle_center / np.asarray([width, height])
        if (
            box_width < 8
            or box_height < 8
            or ratio < 0.0001
            or ratio > 0.75
            or aspect_ratio < 0.25
            or aspect_ratio > 4.0
            or len(vertices) < 4
            or len(vertices) > 8
            or contour_fill < 0.5
            or np.any(normalized_center < 0.05)
            or np.any(normalized_center > 0.95)
        ):
            continue
        distance = np.linalg.norm(
            (rectangle_center - center) / np.asarray([width, height])
        )
        score = rectangle_area * max(0.1, 1.0 - distance)
        candidates.append((score, x, y, box_width, box_height))
    if not candidates:
        return None
    _, x, y, box_width, box_height = max(candidates)
    return (x, y, box_width, box_height)


def detect_roi_frame(image_bgr, min_side=20, min_area_ratio=0.03):
    """公开接口：整图里是否存在"ROI 白框"。

    未裁剪的原始超声截图中心有一个白色 ROI 矩形；已经裁好的 ROI 图没有。
    小于 min_side 或占比低于 min_area_ratio 的亮区视为组织高回声，不算 ROI 框。
    返回 (x, y, w, h) 或 None。
    """
    box = _white_box(image_bgr)
    if box is None:
        return None
    _, _, width, height = box
    if min(width, height) < min_side:
        return None
    area_ratio = width * height / float(image_bgr.shape[0] * image_bgr.shape[1])
    if area_ratio < min_area_ratio:
        return None
    return box


def _elasticity_box(image):
    """白色框检测失败时的兜底：用高饱和（彩色弹性图）区域的外接矩形。"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:, :, 1], 50, 255)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    image_area = float(height * width)
    center = np.asarray([width / 2.0, height / 2.0])
    candidates = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        rectangle_area = float(box_width * box_height)
        ratio = rectangle_area / image_area
        aspect_ratio = box_width / box_height
        contour_fill = cv2.contourArea(contour) / max(rectangle_area, 1.0)
        rectangle_center = np.asarray([x + box_width / 2.0, y + box_height / 2.0])
        normalized_center = rectangle_center / np.asarray([width, height])
        if (
            box_width < 8
            or box_height < 8
            or ratio < 0.0001
            or ratio > 0.75
            or aspect_ratio < 0.25
            or aspect_ratio > 4.0
            or contour_fill < 0.25
            or normalized_center[0] < 0.15
            or normalized_center[0] > 0.95
            or normalized_center[1] < 0.1
            or normalized_center[1] > 0.95
        ):
            continue
        distance = np.linalg.norm(
            (rectangle_center - center) / np.asarray([width, height])
        )
        candidates.append(
            (rectangle_area * max(0.1, 1.0 - distance), x, y, box_width, box_height)
        )
    if not candidates:
        return None
    _, x, y, box_width, box_height = max(candidates)
    return (x, y, box_width, box_height)


def _inset_box(box, ratio=DEFAULT_INSET_RATIO):
    """向内收缩 ratio（相对短边），默认 2%。"""
    x, y, width, height = box
    inset = max(1, int(round(min(width, height) * ratio)))
    new_width = max(1, width - 2 * inset)
    new_height = max(1, height - 2 * inset)
    return (max(0, x + inset), max(0, y + inset), new_width, new_height)


def frame_residue(image, box, ratio=DEFAULT_INSET_RATIO, band=2):
    """内缩后边框带里仍属于白框线的像素比例（只用于自检与统计，不参与裁剪）。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, width, height = _inset_box(box, ratio)
    patch = gray[y : y + height, x : x + width]
    border = np.concatenate(
        [
            patch[:band].ravel(),
            patch[-band:].ravel(),
            patch[:, :band].ravel(),
            patch[:, -band:].ravel(),
        ]
    )
    return float((border >= FRAME_GRAY_THRESHOLD).mean())


def _shift_box(box, delta_x, delta_y):
    x, y, width, height = box
    return (x + delta_x, y + delta_y, width, height)


def _boxes_agree(left, right, tolerance=DEFAULT_BOX_TOLERANCE):
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def _intersect_boxes(left, right):
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    x = max(left_x, right_x)
    y = max(left_y, right_y)
    right_edge = min(left_x + left_width, right_x + right_width)
    bottom_edge = min(left_y + left_height, right_y + right_height)
    if right_edge <= x or bottom_edge <= y:
        return None
    return (x, y, right_edge - x, bottom_edge - y)


def _crop(image, box):
    x, y, width, height = box
    return image[y : y + height, x : x + width]


def _write_pair(
    gray,
    swe,
    gray_path,
    swe_path,
    box_tolerance=DEFAULT_BOX_TOLERANCE,
    inset_ratio=DEFAULT_INSET_RATIO,
):
    """把配对图裁成同一块组织区域并写盘，返回是否严格对齐。"""
    gray_box = _white_box(gray)
    swe_box = _white_box(swe)
    if gray_box is None and swe_box is None:
        swe_box = _elasticity_box(swe)
    if gray_box is None and swe_box is None:
        raise ValueError("White box not found")

    delta_x = gray.shape[1] - swe.shape[1]
    delta_y = gray.shape[0] - swe.shape[0]
    # 统一到灰阶坐标系后再比较
    if gray_box is None:
        gray_box = _shift_box(swe_box, delta_x, delta_y)
    if swe_box is None:
        swe_box = _shift_box(gray_box, -delta_x, -delta_y)
    swe_box_in_gray = _shift_box(swe_box, delta_x, delta_y)

    aligned = _boxes_agree(gray_box, swe_box_in_gray, box_tolerance)
    box = gray_box
    if not aligned:
        intersection = _intersect_boxes(gray_box, swe_box_in_gray)
        if intersection is not None and min(intersection[2], intersection[3]) >= 16:
            box = intersection
    gray_box = _inset_box(box, inset_ratio)
    swe_box = _shift_box(gray_box, -delta_x, -delta_y)

    gray_crop = _crop(gray, gray_box)
    swe_crop = _crop(swe, swe_box)
    if gray_crop.size == 0 or swe_crop.size == 0:
        raise ValueError("Empty crop")
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    swe_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(gray_path), gray_crop):
        raise ValueError("Gray image write failed")
    if not cv2.imwrite(str(swe_path), swe_crop):
        raise ValueError("SWE image write failed")
    return aligned


def _prepare_cohort(archive, config, output_dir, split_name=None):
    labels, label_duplicates = _read_labels(archive, config["workbook_suffix"])
    gray_members, gray_duplicates = _image_members(archive, config["gray_marker"])
    swe_members, swe_duplicates = _image_members(archive, config["swe_marker"])
    rows = []
    excluded = []
    for index, row in labels.reset_index(drop=True).iterrows():
        key = row["match_key"]
        if key in label_duplicates:
            excluded.append({"image_name": row["image_name"], "reason": "duplicate_label_match_key"})
            continue
        if key in gray_duplicates or key in swe_duplicates:
            excluded.append({"image_name": row["image_name"], "reason": "duplicate_match_key"})
            continue
        gray_member = gray_members.get(key)
        swe_member = swe_members.get(key)
        if gray_member is None or swe_member is None:
            excluded.append({"image_name": row["image_name"], "reason": "pair_not_found"})
            continue
        gray_path = output_dir / "images" / "gray" / f"{index:06d}.png"
        swe_path = output_dir / "images" / "swe" / f"{index:06d}.png"
        try:
            crop_aligned = _write_pair(
                _decode(archive.read(gray_member)),
                _decode(archive.read(swe_member)),
                gray_path,
                swe_path,
            )
        except ValueError as error:
            excluded.append({"image_name": row["image_name"], "reason": str(error)})
            continue
        rows.append(
            {
                "image_name": str(row["image_name"]),
                "patient_id": patient_id_from_name(row["image_name"]),
                "emean": float(row["emean"]),
                "gray_path": str(gray_path),
                "swe_path": str(swe_path),
                "crop_aligned": bool(crop_aligned),
                "split": split_name or "",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "image_name",
            "patient_id",
            "emean",
            "gray_path",
            "swe_path",
            "crop_aligned",
            "split",
        ],
    ), pd.DataFrame(
        excluded,
        columns=["image_name", "reason"],
    )


def prepare_dataset(zip_path, config, cache_dir, seed=42):
    zip_path = Path(zip_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(cache_dir / "internal" / "images", ignore_errors=True)
    shutil.rmtree(cache_dir / "external" / "images", ignore_errors=True)
    with ZipFile(zip_path) as archive:
        frame, excluded = _prepare_cohort(archive, config, cache_dir / "internal")
        frame = assign_patient_split(frame, seed=seed)
        frame.to_csv(cache_dir / "pairs_all.csv", index=False)
        frame.to_csv(cache_dir / f"pairs_seed_{seed}.csv", index=False)
        frame[["image_name", "patient_id", "split"]].to_csv(
            cache_dir / "patient_mapping.csv",
            index=False,
        )
        (
            frame.groupby("split", as_index=False)
            .agg(images=("image_name", "size"), patients=("patient_id", "nunique"))
            .to_csv(cache_dir / "split_summary.csv", index=False)
        )
        excluded.to_csv(cache_dir / "excluded.csv", index=False)
        frame.loc[~frame["crop_aligned"], ["image_name", "patient_id", "split"]].to_csv(
            cache_dir / "crop_mismatch.csv",
            index=False,
        )
        external = None
        if "external" in config:
            external, external_excluded = _prepare_cohort(
                archive,
                config["external"],
                cache_dir / "external",
                split_name="external_test",
            )
            external.to_csv(cache_dir / "external_pairs.csv", index=False)
            external[["image_name", "patient_id", "split"]].to_csv(
                cache_dir / "external_patient_mapping.csv",
                index=False,
            )
            external_excluded.to_csv(cache_dir / "external_excluded.csv", index=False)
    return frame, external


def _demo():
    """自检：配对必须裁到同一块内容；错位的配对被标记；白框内缩行为可复现。"""

    def canvas(width, height, box, thickness=3):
        image = np.full((height, width, 3), 30, np.uint8)
        x, y, box_width, box_height = box
        cv2.rectangle(
            image,
            (x, y),
            (x + box_width - 1, y + box_height - 1),
            (255, 255, 255),
            thickness,
        )
        cv2.rectangle(
            image,
            (x + thickness, y + thickness),
            (x + box_width - 1 - thickness, y + box_height - 1 - thickness),
            (90, 90, 90),
            -1,
        )
        # 非对称标记：两张图如果裁到的不是同一块内容，比对就会不相等
        cv2.rectangle(
            image,
            (x + thickness + 5, y + thickness + 7),
            (x + thickness + 15, y + thickness + 17),
            (20, 20, 20),
            -1,
        )
        return image

    import tempfile

    box = (30, 25, 60, 40)
    gray = canvas(120, 90, box)
    swe_aligned = canvas(146, 90, (56, 25, 60, 40))  # 画布宽 26px，ROI 内容位置相同
    swe_offset = canvas(146, 90, (66, 25, 60, 40))  # 内容整体右移 10px

    detected = _white_box(gray)
    assert _boxes_agree(detected, _shift_box(_white_box(swe_aligned), -26, 0))
    assert not _boxes_agree(detected, _shift_box(_white_box(swe_offset), -26, 0))
    inset = _inset_box(detected, 0.08)
    assert inset[2] < detected[2] and inset[3] < detected[3], inset

    # 机制说明：规则矩形白框要靠大内缩才能切掉，但代价是 ROI 面积大幅缩水
    assert frame_residue(gray, detected, 0.02) > 0.1, "2% 内缩后仍可检测到边框残留信号"
    assert frame_residue(gray, detected, 0.12) < 0.05, "足够大的内缩才能切掉规则矩形白框"
    kept = _inset_box(detected, 0.12)
    assert kept[2] * kept[3] < 0.7 * detected[2] * detected[3], "大内缩会吃掉三成以上 ROI"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        aligned = _write_pair(gray, swe_aligned, tmp / "g.png", tmp / "s.png")
        assert aligned is True, "内容一致的配对应判为已对齐"
        gray_crop = cv2.imread(str(tmp / "g.png"))
        swe_crop = cv2.imread(str(tmp / "s.png"))
        assert gray_crop.shape == swe_crop.shape, (gray_crop.shape, swe_crop.shape)
        assert np.array_equal(gray_crop, swe_crop), "对齐后两张裁剪图必须覆盖同一块内容"

        mismatch = _write_pair(gray, swe_offset, tmp / "g2.png", tmp / "s2.png")
        assert mismatch is False, "错位 10px 的配对应判为未对齐"
        gray_offset_crop = cv2.imread(str(tmp / "g2.png"))
        swe_offset_crop = cv2.imread(str(tmp / "s2.png"))
        assert gray_offset_crop.shape == swe_offset_crop.shape
        assert gray_offset_crop.size < gray_crop.size, "未对齐时取交集，裁剪区域应变小"

    print(
        "preprocess self-check ok：配对对齐判定、交集回退、白框内缩代价均符合预期"
    )


if __name__ == "__main__":
    _demo()
