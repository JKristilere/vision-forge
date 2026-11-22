"""Trainer module for training CNN models."""
import time

import mlflow
import torch

from src.visionforge.training.earlystopping import EarlyStopping
from src.visionforge.utils.checkpointing import save_checkpoint
from src.visionforge.utils.logging import setup_logger

logger = setup_logger(__name__)

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler,
                  train_dataloader, val_dataloader=None, 
                  checkpoint_dir="artifacts/checkpoints", log_to_mlflow=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stopping = EarlyStopping(verbose=True)
        self.checkpoint_dir = checkpoint_dir
        self.log_to_mlflow = log_to_mlflow

    def train_one_epoch(self):
        self.model.train()
        total_loss, n_batches = 0.0, 0
        data_wait = 0.0
        batch_start = time.perf_counter()
        for i, (images, labels) in enumerate(self.train_dataloader):
            # Time spent blocked on the DataLoader, i.e. waiting for the next batch.
            data_wait += time.perf_counter() - batch_start
            self.optimizer.zero_grad(set_to_none=True)
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if i % 10 == 0:
                logger.info(f"Training Loss: {loss.item()}")
            total_loss += loss.item()
            n_batches += 1
            # Start timing for the next batch
            batch_start = time.perf_counter()
        # Store the total data wait time for the epoch
        self.epoch_data_wait = data_wait
        return total_loss / n_batches

    def train(self, num_epochs, start_epoch=0,best_val_loss=float('inf')):
        history = []
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            # Timing for the epoch
            epoch_start = time.perf_counter()
            train_loss = self.train_one_epoch()
            # End of epoch timing calculations
            epoch_seconds = time.perf_counter() - epoch_start
            timing = {
                "epoch_seconds": epoch_seconds,
                "samples_per_sec": len(self.train_dataloader.dataset) / epoch_seconds,
                "data_wait_seconds": self.epoch_data_wait,
                "compute_seconds": epoch_seconds - self.epoch_data_wait,
            }
            logger.info(f"Epoch {epoch+1} Training Loss: {train_loss}, timing: {timing}")
            val_loss, val_accuracy = self.validate() if self.val_dataloader is not None else (float('inf'), 0)
            # Schedulers are configured in epochs (e.g. T_max), so step once per epoch.
            if self.scheduler is not None:
                self.scheduler.step()
            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })
            is_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            extra = None
            if self.log_to_mlflow:
                metrics = {"train_loss": train_loss, **timing}
                if self.val_dataloader is not None:
                    metrics.update(val_loss=val_loss, val_accuracy_pct=val_accuracy)
                mlflow.log_metrics(metrics, step=epoch)
                extra = {"mlflow_run_id": mlflow.active_run().info.run_id}
            save_checkpoint(self.checkpoint_dir, self.model, self.optimizer, epoch, val_loss, best_val_loss, is_best, extra=extra)
            self.early_stopping(val_loss)
            if self.early_stopping.stop_training:
                logger.info("Early stopping triggered. Stopping training.")
                if self.log_to_mlflow:
                    mlflow.set_tag("stopped_early_epoch", epoch)
                break
        return history



    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.val_dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            total_loss += self.criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        self.model.train()
        return total_loss / total, correct / total * 100