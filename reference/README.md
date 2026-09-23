# reference/ 目录说明

本目录原本用于存放基线稿件、返修需求和旧项目代码。**这些文件未随仓库发布**，
因此本目录只保留了占位结构：

| 路径 | 内容 | 是否随仓库发布 |
|---|---|---|
| `baseline/manuscript.docx` | 基线稿件 | ❌ 未发布 |
| `baseline/revision_requirements.docx` | 返修需求 | ❌ 未发布 |
| `baseline/code/utils.py`、`model_utils.py` 等 | 旧项目（基线）代码 | ❌ 未发布 |
| `baseline/code/pretrained_models/map2sat.pth` | Map2Sat 生成器预训练权重 | ❌ 未发布（体积大 + 版权归属） |

## 影响

1. 创新点 I2（Map2Sat 预训练初始化）与"外部验证集预训练消融"需要自备该权重：
   放到 `reference/baseline/code/pretrained_models/map2sat.pth`，或用环境变量
   `MEDICAL_IMAGE_ROOT` 指向包含该文件的项目根目录。
   `common/models.py::initialize_map2sat` 会在文件缺失或为 0 字节时给出明确报错。
2. 其它器官的预训练消融中 `using_pretrained=False` 的分支不依赖该文件，可直接运行。

## 数据

原始数据包同样不随仓库发布，需要放到 `data/raw/`（文件名见 `common/paths.py`）：

- `数据发送2026.8.9(1).zip`：肾脏建模 / 肾脏外部验证 / 胰腺 / 腮腺 / 颌下腺
- 第四部分前瞻性灰阶预测 zip：肾脏前瞻队列（仅灰阶，**已裁剪 ROI**）
- 肝脏 zip：肝脏补充实验
