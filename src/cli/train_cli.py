import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*instantiate\\(\\) resolved _target_=.*")

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from src.core.use_cases.train import TrainModelUseCase
from src.adapters.data_adapter import CIFAR100DataLoader


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Hydra entrypoint. Returns metric for Optunas hyperparameter sweeper."""
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.training)
    data_loader = CIFAR100DataLoader(batch_size=cfg.batch_size)
    print_run_config(cfg, model=model, data_loader=data_loader)
    use_case = TrainModelUseCase(
        model=model,
        trainer=trainer,
        data_loader=data_loader,
        epochs=cfg.epochs,
    )
    return use_case.execute()


def print_run_config(cfg, model=None, data_loader=None):
    print("=" * 50)
    print(" RUN CONFIGURATION ".center(50, "="))
    print("=" * 50)
    print(f"  Epochs:         {cfg.get('epochs')}")
    print(f"  Batch Size:     {cfg.get('batch_size')}")
    print(f"  Seed:           {cfg.get('seed')}")
    if data_loader is not None:
        print("-" * 50)
        print(" Dataset Splits: ".ljust(50, "-"))
        if hasattr(data_loader, "train_dataset"):
            print(f"  Train Samples:  {len(data_loader.train_dataset):,}")
        if hasattr(data_loader, "test_dataset"):
            print(f"  Test Samples:   {len(data_loader.test_dataset):,}")
    model_cfg = cfg.get("model", {}).get("config", {})
    print("-" * 50)
    print(" Model Architecture: ".ljust(50, "-"))
    print(f"  Hidden Dim:     {model_cfg.get('hidden_dim')}")
    print(f"  Token MLP Dim:  {model_cfg.get('token_mlp_dim')}")
    print(f"  Channel MLP Dim:{model_cfg.get('channel_mlp_dim')}")
    print(f"  Patch Size:     {model_cfg.get('patch_size')}")
    print(f"  Norm Type:      {model_cfg.get('NormType')}")
    print(f"  Num Blocks:     {model_cfg.get('num_blocks')}")
    if model is not None and hasattr(model, "model"):
        pytorch_model = model.model
        total_params = sum(p.numel() for p in pytorch_model.parameters())
        trainable_params = sum(p.numel() for p in pytorch_model.parameters() if p.requires_grad)
        def format_params(n):
            if n >= 1e6:
                return f"{n / 1e6:.2f}M"
            elif n >= 1e3:
                return f"{n / 1e3:.2f}K"
            return str(n)
        print(f"  Total Params:   {format_params(total_params)} ({total_params:,})")
        print(f"  Trainable:      {format_params(trainable_params)} ({trainable_params:,})")
    train_cfg = cfg.get("training", {}).get("config", {})
    opt_cfg = train_cfg.get("optimizer", {})
    print("-" * 50)
    print(" Training & Optimization: ".ljust(50, "-"))
    print(f"  Optimizer Type: {opt_cfg.get('type')}")
    print(f"  Learning Rate:  {opt_cfg.get('lr')}")
    print(f"  Weight Decay:   {opt_cfg.get('weight_decay')}")
    print(f"  Loss Function:  {train_cfg.get('loss', {}).get('type')}")
    print("=" * 50)

if __name__ == "__main__":
    main()
