import tempfile
import shutil
import pytest

from nexuszero_optimizer.utils.tracing import init_tracer, get_tracer
from nexuszero_optimizer.utils.config import Config
from nexuszero_optimizer.training.dataset import ProofCircuitGenerator


def test_tracing_smoke():
    # Basic smoke test: init tracer and start a simple span
    init_tracer(service_name="nexuszero-test", export_to_console=True)
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("test_span") as span:
        assert span is not None
        span.set_attribute("key", "value")
    # If no exception was raised, tracing is functional
    assert True


def _make_small_dataset(tmpdir: str):
    gen = ProofCircuitGenerator(min_nodes=5, max_nodes=15, seed=123)
    for split, n in [("train", 8), ("val", 4), ("test", 4)]:
        gen.generate_dataset(n, tmpdir, split=split, show_progress=False)


def test_tracing_trainer_export_console():
    """
    Smoke test: run a minimal trainer fit with tracing console export enabled.

    This ensures the SafeConsoleSpanExporter doesn't raise export errors during
    training runs and the tracer behaves safely in test environments.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        _make_small_dataset(tmpdir)
        cfg = Config()
        cfg.data_dir = tmpdir
        cfg.training.num_epochs = 1
        cfg.training.batch_size = 4
        cfg.training.tensorboard_enabled = False

        init_tracer(
            service_name="nexuszero-trainer-test",
            export_to_console=True,
        )
        tracer = get_tracer(__name__)
        assert tracer is not None

        from nexuszero_optimizer.training.trainer import Trainer

        trainer = Trainer(cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        assert "loss" in metrics
    finally:
        shutil.rmtree(tmpdir)


def test_tracing_trainer_export_otlp():
    """Smoke test: initialize tracer with OTLP exporter endpoint and run a minimal Trainer fit.

    This test ensures the OTLP exporter initialization path does not raise errors
    when an OTLP endpoint is present (a collector is expected at the endpoint).
    """
    pytest.importorskip("opentelemetry")
    tmpdir = tempfile.mkdtemp()
    try:
        _make_small_dataset(tmpdir)
        cfg = Config()
        cfg.data_dir = tmpdir
        cfg.training.num_epochs = 1
        cfg.training.batch_size = 4
        cfg.training.tensorboard_enabled = False

        # Initialize tracer - OTLP exporter will be used if OTEL_EXPORTER_OTLP_ENDPOINT is set
        init_tracer(service_name="nexuszero-trainer-otlp")
        tracer = get_tracer(__name__)
        assert tracer is not None

        from nexuszero_optimizer.training.trainer import Trainer

        trainer = Trainer(cfg)
        trainer.fit()
        metrics = trainer.evaluate_test()
        assert "loss" in metrics
    finally:
        shutil.rmtree(tmpdir)


def test_tracing_decorator_integration(monkeypatch):
    """Validate span decorator and attributes using an isolated in-memory provider.

    This test ensures the decorator uses the patched `get_tracer` and that
    child spans record expected attributes which our InMemory exporter receives.
    """
    pytest.importorskip("opentelemetry")

    # dynamic imports
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    try:
        from opentelemetry.sdk.trace.export import InMemorySpanExporter
    except Exception:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Monkeypatch get_tracer and init_tracer in tracing helper
    monkeypatch.setattr(
        "nexuszero_optimizer.utils.tracing.get_tracer",
        lambda name=None: provider.get_tracer(name),
    )
    monkeypatch.setattr(
        "nexuszero_optimizer.utils.tracing.init_tracer",
        lambda *a, **k: None,
    )

    # Import decorator after patching
    from nexuszero_optimizer.utils.tracing import span, get_tracer

    @span("my_test")
    def instrumented():
        t = get_tracer(__name__)
        with t.start_as_current_span("my_test.inner") as s:
            s.set_attribute("epoch", 42)
            s.set_attribute("batch_idx", 0)

    # Call the instrumented function
    instrumented()

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "my_test" in names
    assert "my_test.inner" in names
    assert any("epoch" in s.attributes for s in spans)


def test_tracing_span_attributes(monkeypatch):
    """Use an in-memory exporter to assert spans and attributes are recorded.

    This test sets up an InMemorySpanExporter and a TracerProvider and then
    creates and runs a small Trainer instance. We monkeypatch the module-level
    `init_tracer` to be a no-op so Trainer doesn't replace our provider.
    """
    pytest.importorskip("opentelemetry")

    # Import SDK test exporter dynamically to avoid import-time failures
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        try:
            from opentelemetry.sdk.trace.export import (
                InMemorySpanExporter,
            )
        except Exception:
            # Try alternate path for older/newer SDKs
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
                InMemorySpanExporter,
            )
    except Exception:
        pytest.skip("OpenTelemetry runtime not available for this test")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Ensure trainer uses this provider by monkeypatching the helper get_tracer
    # so Trainer's `get_tracer` returns a tracer from our test provider.
    monkeypatch.setattr(
        "nexuszero_optimizer.utils.tracing.get_tracer",
        lambda name=None: provider.get_tracer(name),
    )

    # Avoid trainer calling init_tracer and overriding provider
    monkeypatch.setattr(
        "nexuszero_optimizer.utils.tracing.init_tracer",
        lambda *a, **k: None,
    )

    tmpdir = tempfile.mkdtemp()
    try:
        _make_small_dataset(tmpdir)
        cfg = Config()
        cfg.data_dir = tmpdir
        cfg.training.num_epochs = 1
        cfg.training.batch_size = 4
        cfg.training.tensorboard_enabled = False

        # Import Trainer after we set any monkeypatches
        from nexuszero_optimizer.training.trainer import Trainer

        trainer = Trainer(cfg)
        trainer.fit()

        spans = exporter.get_finished_spans()
        names = [s.name for s in spans]
        assert any(
            n in names for n in ("train_epoch", "train_epoch.inner", "train_epoch.batch")
        )
        # At least one of the batch/epoch spans should be exported
        assert spans, "Expected at least one exported span"
    finally:
        shutil.rmtree(tmpdir)
