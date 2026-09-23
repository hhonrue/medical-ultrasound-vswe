"""项目路径的唯一来源。

仓库里所有脚本/Notebook 都必须从这里取路径，不要再写死绝对路径。
需要换机器时设置环境变量即可：

    export MEDICAL_IMAGE_ROOT=/path/to/医疗图像
"""

import os
from pathlib import Path


ROOT = Path(os.environ.get("MEDICAL_IMAGE_ROOT", "/root/autodl-tmp/医疗图像")).expanduser()

RAW_ZIP = ROOT / "data" / "raw" / "数据发送2026.8.9(1).zip"
PROSPECTIVE_ZIP = ROOT / "data" / "raw" / "第四部分前瞻性灰阶预测9.6.zip"
PROSPECTIVE_GRAY_ROOT = ROOT / "data" / "input" / "kidney_prospective_gray_20260906"
REFERENCE = ROOT / "reference"
EXPERIMENT = ROOT / "experiment"

# 未随仓库分发：需要自行放置旧的 Map2Sat 生成器权重（见 reference/README.md）
PRETRAINED_GENERATOR = REFERENCE / "baseline" / "code" / "pretrained_models" / "map2sat.pth"

DATASET_CONFIG = {
    "kidney": {
        "workbook_suffix": "/肾脏/建模/肾脏_建模.xlsx",
        "gray_marker": "/肾脏/建模/肾脏_灰阶/",
        "swe_marker": "/肾脏/建模/肾脏_弹性/",
        "external": {
            "workbook_suffix": "/肾脏/肾脏_外部验证/肾脏外部验证89.xlsx",
            "gray_marker": "/肾脏/肾脏_外部验证/肾脏外部验证89_灰阶/",
            "swe_marker": "/肾脏/肾脏_外部验证/肾脏外部验证89_弹性/",
        },
    },
    "pancreas": {
        "workbook_suffix": "/其他器官/胰腺.xlsx",
        "gray_marker": "/其他器官/灰阶/胰腺_灰阶/",
        "swe_marker": "/其他器官/弹性/胰腺_弹性/",
    },
    "parotid": {
        "workbook_suffix": "/其他器官/腮腺.xlsx",
        "gray_marker": "/其他器官/灰阶/腮腺_灰阶/",
        "swe_marker": "/其他器官/弹性/腮腺_弹性/",
    },
    "submandibular": {
        "workbook_suffix": "/其他器官/颌下腺.xlsx",
        "gray_marker": "/其他器官/灰阶/颌下腺_灰阶/",
        "swe_marker": "/其他器官/弹性/颌下腺_弹性/",
    },
}


def dataset_root(name):
    return EXPERIMENT / name


def preprocessing_dir(name):
    return dataset_root(name) / "01_data_processing"


def cache_dir(name):
    return preprocessing_dir(name) / "cache"
