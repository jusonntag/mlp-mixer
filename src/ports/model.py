from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, config) -> None:
        ...

    @abstractmethod
    def forward(self, X, y, X_test, y_test):
        ...

    @abstractmethod
    def train_mode(self) -> None:
        ...

    @abstractmethod
    def eval_mode(self) -> None:
        ...

    @abstractmethod
    def get_parameters(self):
        ...

    @abstractmethod
    def save(self) -> None:
        ...