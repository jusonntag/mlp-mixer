from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def get_train_data(self):
        ...

    @abstractmethod
    def get_test_data(self):
        ...
