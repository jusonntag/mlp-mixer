from src.ports.model import BaseModel
from src.ports.training import BaseTrainer
from src.ports.data_loader import BaseDataLoader


def progress_bar(current, total, prefix="", suffix="", width=30):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {percent:6.2%} {current}/{total} | {suffix}", end="", flush=True)


class TrainModelUseCase:
    def __init__(
        self,
        model: BaseModel,
        trainer: BaseTrainer,
        data_loader: BaseDataLoader,
        epochs: int,
    ):
        self.model = model
        self.trainer = trainer
        self.data_loader = data_loader
        self.epochs = epochs

    def execute(self) -> float:
        self.trainer.setup(self.model)

        print("[INFO] Loading data ...")
        train_data = self.data_loader.get_train_data()
        test_data = self.data_loader.get_test_data()
        best_metric = 0.0

        print("[INFO] Starting training ...")
        for epoch in range(self.epochs):
            self.model.train_mode()
            running_loss = 0.0
            total_batches = len(train_data)
            
            for i, batch in enumerate(train_data):
                step_metrics = self.trainer.train_step(self.model, batch[0], batch[1])
                loss = step_metrics.get("loss", 0.0)
                running_loss += loss
                avg_loss = running_loss / (i + 1)
                
                gpu_info = ""
                if "gpu_mem" in step_metrics:
                    gpu_info = f" | GPU: {step_metrics['gpu_mem']:.0f}MB (max: {step_metrics['gpu_max_mem']:.0f}MB) / {step_metrics['gpu_total']:.1f}GB"

                progress_bar(
                    current=i + 1,
                    total=total_batches,
                    prefix=f"Epoch {epoch+1:02d}/{self.epochs:02d} [Train]",
                    suffix=f"Avg Loss: {avg_loss:.4f} (Batch Loss: {loss:.4f}){gpu_info}",
                    width=20
                )

            self.model.eval_mode()
            total_correct = 0
            total_samples = 0
            test_running_loss = 0.0
            total_test_batches = len(test_data)
            
            for i, batch in enumerate(test_data):
                eval_metrics = self.trainer.eval_step(self.model, batch[0], batch[1])
                test_loss = eval_metrics.get("loss", 0.0)
                test_running_loss += test_loss
                total_correct += eval_metrics.get("correct", 0)
                total_samples += eval_metrics.get("total", 0)
                avg_test_loss = test_running_loss / (i + 1)
                current_acc = total_correct / max(total_samples, 1)
                
                gpu_info = ""
                if "gpu_mem" in eval_metrics:
                    gpu_info = f" | GPU: {eval_metrics['gpu_mem']:.0f}MB (max: {eval_metrics['gpu_max_mem']:.0f}MB) / {eval_metrics['gpu_total']:.1f}GB"

                progress_bar(
                    current=i + 1,
                    total=total_test_batches,
                    prefix=f"Epoch {epoch+1:02d}/{self.epochs:02d} [Test ]",
                    suffix=f"Loss: {avg_test_loss:.4f} | Acc: {current_acc:6.2%}{gpu_info}",
                    width=20
                )
            accuracy = total_correct / max(total_samples, 1)
            print()
            
            if accuracy > best_metric:
                best_metric = accuracy
                self.model.save()

        return best_metric
