from abc import ABC, abstractmethod
from src.ports.model import BaseModel


class BaseTrainer(ABC):
    @abstractmethod
    def __init__(self, config) -> None:
        ...

    @abstractmethod
    def setup(self, model: BaseModel) -> None:
        """Bind optimizer, scheduler, loss to the model's parameters."""
        ...

    @abstractmethod
    def train_step(self, model: BaseModel, X_batch, y_batch) -> dict:
        """One training step: forward + loss + backward + optimizer step."""
        ...

    @abstractmethod
    def eval_step(self, model: BaseModel, X_batch, y_batch) -> dict:
        """One evaluation step: forward + loss (no gradients)."""
        ...
