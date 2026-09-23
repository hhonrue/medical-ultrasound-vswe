"""模型定义（唯一来源）。

仓库里只允许存在这一份生成器 / 判别器 / Emean 回归头定义，脚本不要再复制一份。
规模差异通过参数表达：
    UnetGenerator(base_channels=64)  -> 主模型     (54.41M 参数)
    UnetGenerator(base_channels=16)  -> 轻量变体   (3.41M 参数，前瞻/肝脏使用)
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_channels,
        inner_channels,
        input_channels=None,
        submodule=None,
        outermost=False,
        innermost=False,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        input_channels = input_channels or outer_channels
        down_convolution = nn.Conv2d(
            input_channels,
            inner_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        down_relu = nn.LeakyReLU(0.2, True)
        down_norm = nn.BatchNorm2d(inner_channels)
        up_relu = nn.ReLU(True)
        up_norm = nn.BatchNorm2d(outer_channels)
        if outermost:
            up_convolution = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            layers = [down_convolution, submodule, up_relu, up_convolution, nn.Tanh()]
        elif innermost:
            up_convolution = nn.ConvTranspose2d(
                inner_channels,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            layers = [down_relu, down_convolution, up_relu, up_convolution, up_norm]
        else:
            up_convolution = nn.ConvTranspose2d(
                inner_channels * 2,
                outer_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            layers = [
                down_relu,
                down_convolution,
                down_norm,
                submodule,
                up_relu,
                up_convolution,
                up_norm,
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.outermost:
            return self.model(inputs)
        return torch.cat([inputs, self.model(inputs)], dim=1)


class UnetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, base_channels=64):
        super().__init__()
        block = UnetSkipConnectionBlock(
            base_channels * 8,
            base_channels * 8,
            innermost=True,
        )
        for _ in range(3):
            block = UnetSkipConnectionBlock(
                base_channels * 8,
                base_channels * 8,
                submodule=block,
                use_dropout=True,
            )
        block = UnetSkipConnectionBlock(
            base_channels * 4,
            base_channels * 8,
            submodule=block,
        )
        block = UnetSkipConnectionBlock(
            base_channels * 2,
            base_channels * 4,
            submodule=block,
        )
        block = UnetSkipConnectionBlock(
            base_channels,
            base_channels * 2,
            submodule=block,
        )
        self.model = UnetSkipConnectionBlock(
            output_channels,
            base_channels,
            input_channels=input_channels,
            submodule=block,
            outermost=True,
        )

    def forward(self, inputs):
        return self.model(inputs)


class FeatureEnhancementModule(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        hidden_channels = max(1, channels // 4)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.refinement = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, inputs):
        channel_features = inputs * self.channel_attention(inputs)
        average = channel_features.mean(dim=1, keepdim=True)
        maximum = channel_features.max(dim=1, keepdim=True).values
        spatial_features = channel_features * self.spatial_attention(
            torch.cat([average, maximum], dim=1)
        )
        return spatial_features + self.refinement(spatial_features)


class TranslationNetwork(nn.Module):
    def __init__(self, use_fem=False, base_channels=64):
        super().__init__()
        self.generator = UnetGenerator(base_channels=base_channels)
        self.fem = FeatureEnhancementModule() if use_fem else nn.Identity()

    def forward(self, inputs):
        generated = self.generator(inputs)
        if isinstance(self.fem, nn.Identity):
            return generated
        return torch.tanh(self.fem(generated))


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=6, base_channels=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 1),
        )

    def forward(self, gray, swe):
        return self.model(torch.cat([gray, swe], dim=1))


class EmeanRegressor(nn.Module):
    """ResNet-18 回归头。

    head_width / dropout 决定最后一层结构，两种配置都保留过权重：
        EmeanRegressor()                        -> fc.0 / fc.2（主实验，11.43M）
        EmeanRegressor(head_width=256, dropout=0.2) -> fc.0 / fc.3（肝脏与前瞻）
    """

    def __init__(self, head_width=512, dropout=0.0):
        super().__init__()
        network = resnet18(weights=None)
        network.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        network.maxpool = nn.MaxPool2d(2, 2)
        layers = [nn.Linear(network.fc.in_features, head_width), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(head_width, 1))
        network.fc = nn.Sequential(*layers)
        self.network = network

    def forward(self, inputs):
        return self.network(inputs).flatten()


def _candidate_names(name):
    candidates = [name]
    current = name
    while True:
        changed = False
        for prefix in ("state_dict.", "module.", "generator.", "network."):
            if current.startswith(prefix):
                current = current[len(prefix) :]
                candidates.append(current)
                changed = True
        if not changed:
            break
    return candidates + [f"model.{candidate}" for candidate in list(candidates)]


def initialize_map2sat(generator, path, allow_slice=False):
    """用旧的 Map2Sat U-Net 权重初始化生成器。

    - allow_slice=False：只接受形状完全一致的张量（主实验用）。
    - allow_slice=True ：允许源张量比目标大，按通道切片载入（轻量变体用）。

    返回 {"loaded": 成功张量数, "sliced": 是否发生了通道切片}，便于写进 manifest。

    注意：map2sat.pth 未随仓库分发（见 reference/README.md）。文件缺失或为空时
    这里会直接给出可读的错误，而不是抛出 torch.load 的 EOFError。
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"预训练权重不可用：{path}（缺失或 0 字节）。该权重未随仓库分发，"
            "请按 reference/README.md 自备，或关闭 USE_PRETRAINED。"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    current_state = generator.state_dict()
    compatible = {}
    sliced = False
    for name, target in current_state.items():
        for candidate in _candidate_names(name):
            value = state.get(candidate)
            if value is None or value.ndim != target.ndim:
                continue
            if tuple(value.shape) == tuple(target.shape):
                compatible[name] = value
                break
            if allow_slice and all(
                target_size <= value_size
                for target_size, value_size in zip(target.shape, value.shape)
            ):
                slices = tuple(slice(0, size) for size in target.shape)
                compatible[name] = value[slices].clone()
                sliced = True
                break
    if len(compatible) < max(1, len(current_state) // 2):
        raise RuntimeError(
            f"Only {len(compatible)} of {len(current_state)} generator tensors are compatible"
        )
    generator.load_state_dict(compatible, strict=False)
    return {"loaded": len(compatible), "sliced": sliced}
