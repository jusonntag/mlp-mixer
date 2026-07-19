import torch
from torch import nn


class LayerNorem(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        ...
        pass
    
    #def forward(x: torch.Tensor) -> torch.Tensor:
    #    ...


class RMSNorm(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
    ...


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_features, out_features, bias=bias),
        )

    def forward(self, x):
        return self.f(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        n_patches,
        hidden_dim,
        token_mlp_dim,
        token_bias,
        channel_mlp_dim,
        channel_bias,
        NormType="Layer",
    ):
        super().__init__()
        self.norm1 = (
            nn.LayerNorm(hidden_dim)
            if NormType == "Layer"
            else nn.RMSNorm(hidden_dim)
        )
        self.token_mixing = MLP(n_patches, token_mlp_dim, n_patches, token_bias)
        self.norm2 = (
            nn.LayerNorm(hidden_dim)
            if NormType == "Layer"
            else nn.RMSNorm(hidden_dim)
        )
        self.channel_mixing = MLP(hidden_dim, channel_mlp_dim, hidden_dim, channel_bias)

    def forward(self, x):
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y
        y = self.norm2(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MLPMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.hidden_dim = config.hidden_dim

        self.conv = nn.Conv2d(
            config.input_size,
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=config.get("conv_bias", True),
        )
        self.blocks = nn.ModuleList([
                MixerBlock(
                    n_patches=config.n_patches,
                    hidden_dim=self.hidden_dim,
                    token_mlp_dim=config.token_mlp_dim,
                    token_bias=config.token_bias,
                    channel_mlp_dim=config.channel_mlp_dim,
                    channel_bias=config.channel_bias,
                    NormType=config.NormType,
                )
                for _ in range(config.num_blocks)
        ])
        self.norm = (
            nn.LayerNorm(self.hidden_dim)
            if config.NormType == "Layer"
            else nn.RMSNorm(self.hidden_dim)
        )
        self.head = nn.Linear(self.hidden_dim, config.n_classes)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = x.mean(dim=1) 
        x = self.head(x)
        #x = nn.Softmax(x)
        return x