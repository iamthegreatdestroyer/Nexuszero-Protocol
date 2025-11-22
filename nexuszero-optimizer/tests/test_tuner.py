import pytest

def test_optuna_tuner_import():
    try:
        import optuna  # noqa: F401
    except Exception:
        pytest.skip("Optuna not installed")

    from nexuszero_optimizer.utils.config import Config
    from nexuszero_optimizer.training.tuner import OptunaTuner

    cfg = Config()
    cfg.training.num_epochs = 1  # speed
    tuner = OptunaTuner(cfg, n_trials=1)
    study = tuner.run()
    assert len(study.trials) == 1
