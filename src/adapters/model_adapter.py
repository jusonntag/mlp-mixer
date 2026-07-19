import torch

from src.ports.model import BaseModel
from src.adapters.architectures import MLPMixer


class MLPMixerModel(BaseModel):
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.n_patches = (config.input_H // config.patch_size) * (config.input_W // config.patch_size)
        self.model = MLPMixer(config).to(self.device)

    def forward(self, X):
        return self.model(X)

    def train_mode(self) -> None:
        self.model.train()

    def eval_mode(self) -> None:
        self.model.eval()

    def get_parameters(self):
        return self.model.parameters()

    def save(self) -> None:
        torch.save(self.model.state_dict(), "best_model.pt")
