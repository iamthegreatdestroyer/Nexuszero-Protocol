"""Trainer implementation for Nexuszero Optimizer."""

from typing import Optional, Dict, Any
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .dataset import create_dataloaders
from .metrics import (
    parameter_mse,
    metrics_mse,
    security_penalty,
    MetricTracker,
)
from ..verification.validator import BatchSoundnessValidator
from ..verification.soundness import SoundnessVerifier
from ..models.gnn import ProofOptimizationGNN
from ..utils.config import Config
from pathlib import Path
import logging
from ..utils.tracing import init_tracer, get_tracer, span

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Data
        self.train_loader, self.val_loader, self.test_loader = (
            create_dataloaders(
                config.data_dir,
                batch_size=config.training.batch_size,
                num_workers=config.num_workers,
            )
        )

        # Model
        self.model = ProofOptimizationGNN(
            node_feat_dim=10,
            edge_feat_dim=4,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
        ).to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # Soundness
        self.soundness_verifier = SoundnessVerifier(
            n_min=config.optimization.n_min,
            n_max=config.optimization.n_max,
            q_min=config.optimization.q_min,
            q_max=config.optimization.q_max,
            sigma_min=config.optimization.sigma_min,
            sigma_max=config.optimization.sigma_max,
        )
        self.batch_validator = BatchSoundnessValidator(self.soundness_verifier)

        # Optional batch limiting for fast tuning runs
        self.max_batches_per_epoch: Optional[int] = getattr(
            config.training, "max_batches_per_epoch", None
        )

        # Tracking
        self.best_val_loss: Optional[float] = None
        self.early_stop_counter = 0

        # Logging (optional)
        self.tb_writer = None
        if config.training.tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=config.log_dir)
            except Exception:
                self.tb_writer = None

        # WandB
        self.wandb_run = None
        if config.training.wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.training.wandb_project,
                    entity=(
                        config.training.wandb_entity
                        if config.training.wandb_entity
                        else None
                    ),
                    name=config.training.wandb_run_name,
                    config=config.to_dict(),
                    tags=config.training.wandb_tags,
                )
            except Exception:
                self.wandb_run = None

        # Ensure checkpoint dir exists
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Initialize tracing (noop if not installed)
        try:
            init_tracer(service_name="nexuszero-optimizer")
        except Exception:
            pass
        self._tracer = get_tracer(__name__)

    def _build_scheduler(self):
        tcfg = self.config.training
        if tcfg.scheduler_type == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=tcfg.scheduler_factor,
                patience=tcfg.scheduler_patience,
            )
        if tcfg.scheduler_type == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=tcfg.num_epochs)
        return None

    def _forward_batch(self, batch) -> Dict[str, Any]:
        batch = batch.to(self.device)
        params_pred, metrics_pred = self.model(
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        return {
            "params_pred": params_pred,
            "metrics_pred": metrics_pred,
            "params_target": batch.y,
            "metrics_target": batch.metrics,
        }

    @span("train_epoch")
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        tracker = MetricTracker()
        total_batches = len(self.train_loader)
        logger.info(
            "Epoch %d start batches=%d limit=%s" % (
                epoch,
                total_batches,
                str(self.max_batches_per_epoch),
            )
        )
        with self._tracer.start_as_current_span("train_epoch.inner") as span:
            span.set_attribute("epoch", epoch)
            span.set_attribute("batch_size", self.config.training.batch_size)
            for batch_idx, batch in enumerate(self.train_loader):
                if (
                    self.max_batches_per_epoch is not None
                    and batch_idx >= self.max_batches_per_epoch
                ):
                    logger.info(
                        "Early stop batch %d limit=%d" % (
                            batch_idx,
                            self.max_batches_per_epoch,
                        )
                    )
                    break

                with self._tracer.start_as_current_span(
                    "train_epoch.batch"
                ) as batch_span:
                    batch_span.set_attribute("batch_idx", batch_idx)
                    out = self._forward_batch(batch)
                p_loss = parameter_mse(
                    out["params_pred"], out["params_target"]
                )
                m_loss = (
                    metrics_mse(out["metrics_pred"], out["metrics_target"])
                    * self.config.training.aux_metrics_loss_weight
                )
                batch_metrics = self.batch_validator.evaluate_batch(
                    out["params_pred"]
                )
                s_pen = security_penalty(batch_metrics["security_score_mean"])
                total_loss = p_loss + m_loss + s_pen

                self.optimizer.zero_grad()
                total_loss.backward()
                if self.config.training.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.grad_clip
                    )
                self.optimizer.step()

                tracker.update("loss", float(total_loss.item()))
                tracker.update("param_loss", float(p_loss.item()))
                tracker.update("metrics_loss", float(m_loss.item()))
                tracker.update("security_penalty", float(s_pen))
                tracker.update(
                    "security_score", batch_metrics["security_score_mean"]
                )
                tracker.update(
                    "bit_security", batch_metrics["bit_security_mean"]
                )
                tracker.update("hardness", batch_metrics["hardness_mean"])

                # Track proof size predictions
                # Metrics are [proof_size, prove_time, verify_time]
                tracker.update(
                    "proof_size_norm",
                    float(out["metrics_pred"][:, 0].mean().item()),
                )

                if batch_idx % 10 == 0:
                    logger.info(
                        "Epoch %d batch %d loss=%.4f sec=%.4f" % (
                            epoch,
                            batch_idx,
                            total_loss.item(),
                            batch_metrics["security_score_mean"],
                        )
                    )

        avg = tracker.to_dict()
        logger.info(
            "Epoch %d train summary loss=%.4f param=%.4f sec=%.4f" % (
                epoch,
                avg.get("loss", 0.0),
                avg.get("param_loss", 0.0),
                avg.get("security_score", 0.0),
            )
        )
        if self.tb_writer:
            for k, v in avg.items():
                self.tb_writer.add_scalar(f"train/{k}", v, epoch)
        if self.wandb_run:
            import wandb
            wandb.log({f"train_{k}": v for k, v in avg.items()}, step=epoch)
        return avg

    @torch.no_grad()
    @span("validate_epoch")
    def validate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        for batch in self.val_loader:
            out = self._forward_batch(batch)
            p_loss = parameter_mse(out["params_pred"], out["params_target"])
            m_loss = (
                metrics_mse(out["metrics_pred"], out["metrics_target"])
                * self.config.training.aux_metrics_loss_weight
            )
            batch_metrics = self.batch_validator.evaluate_batch(
                out["params_pred"]
            )
            s_pen = security_penalty(batch_metrics["security_score_mean"])
            total_loss = p_loss + m_loss + s_pen

            tracker.update("loss", float(total_loss.item()))
            tracker.update("param_loss", float(p_loss.item()))
            tracker.update("metrics_loss", float(m_loss.item()))
            tracker.update("security_penalty", float(s_pen))
            tracker.update(
                "security_score", batch_metrics["security_score_mean"]
            )
            tracker.update("bit_security", batch_metrics["bit_security_mean"])
            tracker.update("hardness", batch_metrics["hardness_mean"])

        avg = tracker.to_dict()
        logger.info(
            "Epoch %d val summary loss=%.4f sec=%.4f" % (
                epoch,
                avg.get("loss", 0.0),
                avg.get("security_score", 0.0),
            )
        )
        if self.tb_writer:
            for k, v in avg.items():
                self.tb_writer.add_scalar(f"val/{k}", v, epoch)
        if self.wandb_run:
            import wandb
            wandb.log({f"val_{k}": v for k, v in avg.items()}, step=epoch)
        return avg

    @span("checkpoint")
    def _maybe_checkpoint(self, val_metrics: Dict[str, float], epoch: int):
        val_loss = val_metrics["loss"]
        tcfg = self.config.training
        ckpt_dir = Path(self.config.checkpoint_dir)
        save_path = ckpt_dir / f"epoch_{epoch}.pt"

        improved = self.best_val_loss is None or val_loss < self.best_val_loss
        if improved:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            if tcfg.checkpoint_best_only:
                torch.save(self.model.state_dict(), ckpt_dir / "best.pt")
        else:
            self.early_stop_counter += 1

        if not tcfg.checkpoint_best_only:
            torch.save(self.model.state_dict(), save_path)

    @span("training_run")
    def fit(self):
        for epoch in range(1, self.config.training.num_epochs + 1):
            self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self._maybe_checkpoint(val_metrics, epoch)

            if self.scheduler:
                if hasattr(self.scheduler, "step"):
                    # Plateau scheduler expects metric
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()

            # Early stopping
            if (
                self.early_stop_counter
                >= self.config.training.early_stopping_patience
            ):
                break

        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    @torch.no_grad()
    @span("evaluate_test")
    def evaluate_test(self) -> Dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()
        for batch in self.test_loader:
            out = self._forward_batch(batch)
            p_loss = parameter_mse(out["params_pred"], out["params_target"])
            m_loss = (
                metrics_mse(out["metrics_pred"], out["metrics_target"])
                * self.config.training.aux_metrics_loss_weight
            )
            batch_metrics = self.batch_validator.evaluate_batch(
                out["params_pred"]
            )
            s_pen = security_penalty(batch_metrics["security_score_mean"])
            total_loss = p_loss + m_loss + s_pen

            tracker.update("loss", float(total_loss.item()))
            tracker.update("param_loss", float(p_loss.item()))
            tracker.update("metrics_loss", float(m_loss.item()))
            tracker.update("security_penalty", float(s_pen))
            tracker.update(
                "security_score", batch_metrics["security_score_mean"]
            )
            tracker.update("bit_security", batch_metrics["bit_security_mean"])
            tracker.update("hardness", batch_metrics["hardness_mean"])

        return tracker.to_dict()


__all__ = ["Trainer"]
