"""Evaluate a trained model checkpoint on the test split."""

import argparse
import torch
from nexuszero_optimizer import Config, ProofOptimizationGNN, SoundnessVerifier
from nexuszero_optimizer.training.dataset import create_dataloaders
from nexuszero_optimizer.verification.validator import BatchSoundnessValidator
from nexuszero_optimizer.training.metrics import (
    parameter_mse,
    metrics_mse,
    security_penalty,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--checkpoint", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = config.device if torch.cuda.is_available() else "cpu"
    _, _, test_loader = create_dataloaders(
        config.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.num_workers,
    )
    model = ProofOptimizationGNN(
        node_feat_dim=10,
        edge_feat_dim=4,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
    )
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    verifier = SoundnessVerifier(
        n_min=config.optimization.n_min,
        n_max=config.optimization.n_max,
        q_min=config.optimization.q_min,
        q_max=config.optimization.q_max,
        sigma_min=config.optimization.sigma_min,
        sigma_max=config.optimization.sigma_max,
    )
    batch_validator = BatchSoundnessValidator(verifier)

    losses = []
    sec_scores = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            params_pred, metrics_pred = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )
            p_loss = parameter_mse(params_pred, batch.y)
            m_loss = (
                metrics_mse(metrics_pred, batch.metrics)
                * config.training.aux_metrics_loss_weight
            )
            batch_metrics = batch_validator.evaluate_batch(params_pred)
            s_pen = security_penalty(batch_metrics["security_score_mean"])
            total = p_loss + m_loss + s_pen
            losses.append(float(total.item()))
            sec_scores.append(batch_metrics["security_score_mean"])

    print("Avg total loss:", sum(losses) / len(losses))
    print("Avg security score:", sum(sec_scores) / len(sec_scores))


if __name__ == "__main__":
    main()
