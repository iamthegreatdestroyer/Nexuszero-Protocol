import tempfile
import shutil
from pathlib import Path
from nexuszero_optimizer.utils.config import Config
from nexuszero_optimizer.training.dataset import ProofCircuitGenerator
from nexuszero_optimizer.training.trainer import Trainer


def _make_small_dataset(tmpdir: str):
    gen = ProofCircuitGenerator(min_nodes=5, max_nodes=15, seed=123)
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        gen.generate_dataset(n, tmpdir, split=split, show_progress=False)


def test_trainer_smoke():
    tmpdir = tempfile.mkdtemp()
    try:
        _make_small_dataset(tmpdir)
        cfg = Config()
        cfg.data_dir = tmpdir
        cfg.training.num_epochs = 2
        cfg.training.batch_size = 4
        cfg.training.tensorboard_enabled = False
        trainer = Trainer(cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        assert "loss" in metrics
    finally:
        shutil.rmtree(tmpdir)
