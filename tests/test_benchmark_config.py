from pathlib import Path

import pytest

from seqtrainer.benchmarks import (
    ConfigValidationError,
    REQUIRED_CLASSIFICATION_METRICS,
    load_benchmark_config,
)
from seqtrainer.benchmarks.config import parse_benchmark_config


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config-examples" / "benchmarks"


def test_example_benchmark_configs_load():
    for name in ("cnn.toml", "dnabert2.toml", "ipromp.toml"):
        config = load_benchmark_config(CONFIG_DIR / name)
        assert config.dataset.name == "ep_dnabert2_genomic_order"
        assert config.split.strategy == "predefined"
        assert set(REQUIRED_CLASSIFICATION_METRICS).issubset(config.evaluation.metrics)


def test_model_examples_share_dataset_and_split_contract():
    configs = [load_benchmark_config(CONFIG_DIR / name) for name in ("cnn.toml", "dnabert2.toml", "ipromp.toml")]
    dataset_names = {config.dataset.name for config in configs}
    split_files = {tuple(sorted(config.dataset.split_files.items())) for config in configs}
    split_strategies = {config.split.strategy for config in configs}

    assert dataset_names == {"ep_dnabert2_genomic_order"}
    assert len(split_files) == 1
    assert split_strategies == {"predefined"}


def test_missing_required_section_fails_clearly():
    with pytest.raises(ConfigValidationError, match="missing required section"):
        parse_benchmark_config({}, source="demo.toml")


def test_invalid_model_family_fails_clearly():
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    raw = {
        "experiment": config.experiment.__dict__,
        "dataset": {**config.dataset.__dict__, "split_files": dict(config.dataset.split_files)},
        "label": config.label.__dict__,
        "split": config.split.__dict__,
        "preprocessing": config.preprocessing.__dict__,
        "model": {**config.model.__dict__, "family": "random_forest"},
        "training": config.training.__dict__,
        "evaluation": {**config.evaluation.__dict__, "metrics": list(config.evaluation.metrics)},
        "outputs": config.outputs.__dict__,
        "environment": config.environment.__dict__,
    }

    with pytest.raises(ConfigValidationError, match="model.family"):
        parse_benchmark_config(raw, source="demo.toml")


def test_train_val_test_ratio_validation():
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    raw = {
        "experiment": config.experiment.__dict__,
        "dataset": {**config.dataset.__dict__, "split_files": {}},
        "label": config.label.__dict__,
        "split": {
            **config.split.__dict__,
            "strategy": "train_val_test",
            "train_size": 0.8,
            "validation_size": 0.3,
            "test_size": 0.1,
        },
        "preprocessing": config.preprocessing.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "evaluation": {**config.evaluation.__dict__, "metrics": list(config.evaluation.metrics)},
        "outputs": config.outputs.__dict__,
        "environment": config.environment.__dict__,
    }

    with pytest.raises(ConfigValidationError, match="sum to 1.0"):
        parse_benchmark_config(raw, source="demo.toml")


def test_required_metric_suite_is_enforced():
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    raw = {
        "experiment": config.experiment.__dict__,
        "dataset": {**config.dataset.__dict__, "split_files": dict(config.dataset.split_files)},
        "label": config.label.__dict__,
        "split": config.split.__dict__,
        "preprocessing": config.preprocessing.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "evaluation": {**config.evaluation.__dict__, "metrics": ["accuracy", "mcc"]},
        "outputs": config.outputs.__dict__,
        "environment": config.environment.__dict__,
    }

    with pytest.raises(ConfigValidationError, match="evaluation.metrics"):
        parse_benchmark_config(raw, source="demo.toml")
