import csv
from pathlib import Path

import torch
from pytorch_lightning import Callback


class BestStateCallback(Callback):
    """只按验证指标保留最佳权重，训练结束后把最佳状态写回模型。

    monitor 必须是验证集指标（例如 val_loss / val_reconstruction）；
    绝不允许用测试集指标选权重。
    """

    def __init__(self, monitor, component):
        super().__init__()
        self.monitor = monitor
        self.component = component
        self.best = float("inf")
        self.state = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None or not torch.isfinite(value):
            return
        numeric = float(value.detach().cpu())
        if numeric < self.best:
            self.best = numeric
            module = getattr(pl_module, self.component)
            self.state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in module.state_dict().items()
            }

    def on_fit_end(self, trainer, pl_module):
        if self.state is None:
            raise RuntimeError(f"No finite {self.monitor} was observed")
        getattr(pl_module, self.component).load_state_dict(self.state)


class EpochRecorder(Callback):
    """把每个 epoch 的验证指标逐行落盘（epoch_metrics.csv）。

    存在的意义：只保留最终权重时，无法回答"实际训练了多少轮""early stopping
    在第几轮触发""最佳验证值是多少"。落盘后这些都能从文件复核。
    """

    def __init__(self, path, keys=None, when="validation"):
        super().__init__()
        if when not in {"validation", "train"}:
            raise ValueError(when)
        self.when = when
        self.path = Path(path)
        self.keys = list(
            keys
            or (
                "train_loss",
                "train_generator_loss",
                "train_discriminator_loss",
                "train_reconstruction",
                "train_l1",
                "val_loss",
                "val_reconstruction",
                "test_loss",
                "test_mae",
            )
        )
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.when != "validation" or trainer.sanity_checking:
            return
        self._record(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.when != "train":
            return
        self._record(trainer)

    def _record(self, trainer):
        row = {"epoch": int(trainer.current_epoch) + 1}
        for key in self.keys:
            value = trainer.callback_metrics.get(key)
            row[key] = None if value is None else float(value.detach().cpu())
        self.rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)


def _demo():
    """自检：EpochRecorder 必须在正确的钩子上落盘，非法参数要被拒绝。"""
    import csv
    import tempfile

    class FakeTrainer:
        sanity_checking = False
        current_epoch = 0

        def set_metrics(self, **kwargs):
            self.callback_metrics = {key: torch.tensor(float(value)) for key, value in kwargs.items()}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "validation.csv"
        recorder = EpochRecorder(path, keys=["val_loss", "train_loss"])
        trainer = FakeTrainer()
        for epoch in range(3):
            trainer.current_epoch = epoch
            trainer.set_metrics(val_loss=1.0 / (epoch + 1), train_loss=2.0 / (epoch + 1))
            recorder.on_validation_epoch_end(trainer, None)
        rows = list(csv.DictReader(path.open()))
        assert [row["epoch"] for row in rows] == ["1", "2", "3"], rows
        assert abs(float(rows[-1]["val_loss"]) - 1 / 3) < 1e-6, rows[-1]

        path = Path(tmp) / "train.csv"
        recorder = EpochRecorder(path, keys=["train_l1"], when="train")
        trainer.set_metrics(train_l1=0.5)
        recorder.on_train_epoch_end(trainer, None)
        recorder.on_validation_epoch_end(trainer, None)  # 不应写入
        assert len(list(csv.DictReader(path.open()))) == 1

        try:
            EpochRecorder(path, when="nope")
            raise AssertionError("非法 when 未被拒绝")
        except ValueError:
            pass

    print("train self-check ok：EpochRecorder 两种模式与参数校验均符合预期")


if __name__ == "__main__":
    _demo()
