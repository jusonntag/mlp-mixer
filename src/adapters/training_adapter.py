import torch

from src.ports.training import BaseTrainer
from src.ports.model import BaseModel


class PyTorchTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = self._build_loss(config.loss)
        self.optimizer = None
        self.scheduler = None

    def setup(self, model: BaseModel) -> None:
        # CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            torch.compile(model.forward)

        self.optimizer = self._build_optimizer(model.get_parameters())
        self.scheduler = self._build_scheduler()
        print("[INFO] Optimization successfull!")

    def train_step(self, model: BaseModel, X_batch, y_batch) -> dict:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        self.optimizer.zero_grad()
        predictions = model.forward(X_batch)
        loss = self.criterion(predictions, y_batch)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        metrics = {"loss": loss.item()}
        if self.device.type == "cuda":
            metrics["gpu_mem"] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            metrics["gpu_max_mem"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            metrics["gpu_total"] = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        return metrics

    def eval_step(self, model: BaseModel, X_batch, y_batch) -> dict:
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        with torch.no_grad():
            predictions = model.forward(X_batch)
            loss = self.criterion(predictions, y_batch)
            predicted = predictions.argmax(dim=1)
            correct = (predicted == y_batch).sum().item()

        metrics = {"loss": loss.item(), "correct": correct, "total": y_batch.size(0)}
        if self.device.type == "cuda":
            metrics["gpu_mem"] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            metrics["gpu_max_mem"] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            metrics["gpu_total"] = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
        return metrics

    def _build_optimizer(self, params):
        optimizers = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD,
        }
        cls = optimizers[self.config.optimizer.type]
        return cls(
            params,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.get("weight_decay", 0),
        )

    def _build_loss(self, loss_cfg):
        losses = {
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "MSELoss": torch.nn.MSELoss,
        }
        return losses[loss_cfg.type]()

    def _build_scheduler(self):
        schedulers = {
            "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
            "StepLR": torch.optim.lr_scheduler.StepLR,
        }
        cls = schedulers[self.config.scheduler.type]
        return cls(self.optimizer, T_max=self.config.scheduler.T_max)
