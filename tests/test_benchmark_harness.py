from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pandas as pd
import pytest

from seqtrainer.benchmarks import (
    build_run_manifest,
    compare_benchmark_outputs,
    decide_imbalance_policy,
    load_benchmark_config,
    load_predefined_split_frames,
    prepare_dnabert2_tokenized_splits,
    run_benchmark,
    summarize_split_frames,
    threshold_metric_from_strategy,
    write_benchmark_outputs,
)
from seqtrainer.cli.main import main
from seqtrainer.metrics import best_threshold_by_metric, binary_classification_metrics


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config-examples" / "benchmarks"


def test_binary_metrics_include_required_fields_and_confusion_matrix():
    metrics = binary_classification_metrics(
        y_true=[0, 1, 1, 0],
        y_score=[0.1, 0.8, 0.4, 0.3],
        threshold=0.5,
    )

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["sensitivity"] == 0.5
    assert metrics["specificity"] == 1.0
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 1, "tp": 1}


def test_binary_metrics_handle_one_class_split_gracefully():
    metrics = binary_classification_metrics(
        y_true=[0, 0, 0],
        y_score=[0.1, 0.2, 0.3],
        threshold=0.5,
    )

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["sensitivity"] is None
    assert metrics["specificity"] == 1.0
    assert "warning" in metrics


def test_best_threshold_uses_score_extremes_not_only_middle_grid():
    threshold, score = best_threshold_by_metric(
        y_true=[0, 1],
        y_score=[0.01, 0.04],
        metric="mcc",
    )

    assert score == 1.0
    assert ((pd.Series([0.01, 0.04], dtype=float) >= threshold).astype(int).tolist()) == [0, 1]


def test_predefined_split_loader_and_summary(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for split, filename in {
        "train": "train_EP_DNA_BERT2_genomic_order.csv",
        "validation": "eval_EP_DNA_BERT2_genomic_order.csv",
        "test": "test_EP_DNA_BERT2_genomic_order.csv",
    }.items():
        pd.DataFrame(
            {
                "sequence": ["ACGT", "TGCA", "AAAA"],
                "label": [0, 1, 1],
                "split_name": split,
            }
        ).to_csv(split_dir / filename, index=False)

    frames = load_predefined_split_frames(config, base_dir=tmp_path)
    summary = summarize_split_frames(config, frames)

    assert set(frames) == {"train", "validation", "test"}
    assert summary["train"]["rows"] == 3
    assert summary["train"]["class_counts"] == {"0": 1, "1": 2}
    assert summary["train"]["sequence_length"]["max"] == 4


def test_predefined_split_loader_validates_required_columns(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for filename in (
        "train_EP_DNA_BERT2_genomic_order.csv",
        "eval_EP_DNA_BERT2_genomic_order.csv",
        "test_EP_DNA_BERT2_genomic_order.csv",
    ):
        pd.DataFrame({"sequence": ["ACGT"]}).to_csv(split_dir / filename, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_predefined_split_frames(config, base_dir=tmp_path)


def test_benchmark_artifact_writers_create_json_and_csv(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    metrics = {
        "validation": binary_classification_metrics(
            y_true=[0, 1],
            y_score=[0.2, 0.8],
            threshold=0.5,
        )
    }
    manifest = build_run_manifest(
        config,
        split_summary={"validation": {"rows": 2, "class_counts": {"0": 1, "1": 1}}},
        threshold=0.5,
    )
    predictions = pd.DataFrame(
        {
            "split": ["validation", "validation"],
            "idx": [0, 1],
            "label": [0, 1],
            "probability": [0.2, 0.8],
            "threshold": [0.5, 0.5],
            "prediction": [0, 1],
        }
    )

    written = write_benchmark_outputs(
        tmp_path,
        manifest=manifest,
        metrics=metrics,
        predictions=predictions,
        config=config,
    )

    assert written["manifest"].exists()
    assert written["metrics_json"].exists()
    assert written["metrics_csv"].exists()
    assert written["predictions"].exists()

    metrics_csv = pd.read_csv(tmp_path / "metrics.csv")
    assert {"tn", "fp", "fn", "tp"}.issubset(metrics_csv.columns)
    assert metrics_csv.loc[0, "split"] == "validation"


def test_benchmark_manifest_cli_writes_shared_manifest(tmp_path, capsys):
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    for filename in (
        "train_EP_DNA_BERT2_genomic_order.csv",
        "eval_EP_DNA_BERT2_genomic_order.csv",
        "test_EP_DNA_BERT2_genomic_order.csv",
    ):
        pd.DataFrame({"sequence": ["ACGT", "TGCA"], "label": [0, 1]}).to_csv(
            split_dir / filename,
            index=False,
        )

    output_dir = tmp_path / "manifest_run"
    exit_code = main(
        [
            "benchmark-manifest",
            "--config",
            str(CONFIG_DIR / "cnn.toml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "train: rows=2" in captured.out
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "config.json").exists()


def test_benchmark_run_cli_runs_cnn_and_writes_common_outputs(tmp_path, capsys):
    split_dir = tmp_path / "data" / "promoter_classification"
    split_dir.mkdir(parents=True)
    sequences = ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GGGGTTTT"]
    for split, filename in {
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    }.items():
        pd.DataFrame({"sequence": sequences, "label": [0, 1, 0, 1]}).to_csv(
            split_dir / filename,
            index=False,
        )

    config_path = tmp_path / "cnn_smoke.toml"
    config_path.write_text(
        """
[experiment]
name = "cnn_smoke"
task = "bacterial_promoter_prediction"
seed = 42

[dataset]
name = "synthetic"
format = "csv"
sequence_field = "sequence"
label_field = "label"

[dataset.split_files]
train = "data/promoter_classification/train.csv"
validation = "data/promoter_classification/validation.csv"
test = "data/promoter_classification/test.csv"

[label]
source = "provided_binary"

[split]
strategy = "predefined"
seed = 42

[preprocessing]
encoding = "one_hot"
sequence_length = 32

[model]
family = "cnn"
name = "tiny_dna_cnn"

[model.params]
variant = "tiny"

[training]
seed = 42
batch_size = 2
max_epochs = 1
learning_rate = 0.001

[training.params]
optimizer = "adamw"
scheduler = "one_cycle"
select_best_by_mcc = true

[evaluation]
primary_metric = "mcc"
threshold_strategy = "validation_mcc"
metrics = [
  "accuracy",
  "balanced_accuracy",
  "auroc",
  "auprc",
  "f1",
  "mcc",
  "precision",
  "sensitivity",
  "specificity",
  "confusion_matrix",
]

[outputs]
output_dir = "outputs/ignored"
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "benchmark_run"
    exit_code = main(["benchmark", "run", str(config_path), "--base-dir", str(tmp_path), "--output-dir", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "status=completed" in captured.out
    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "predictions.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "checkpoints" / "best_model.pt").exists()


def test_dnabert2_benchmark_gracefully_skips_without_local_model_files(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "dnabert2_frozen.toml")
    _write_configured_split_files(config, tmp_path)

    result = run_benchmark(config, base_dir=tmp_path, output_dir=tmp_path / "dnabert2")

    assert result.status in {"skipped", "completed"}
    assert (tmp_path / "dnabert2" / "manifest.json").exists()
    if result.status == "skipped":
        assert result.manifest["extra"]["status"] == "skipped"


def test_dnabert2_pad_token_patch_defaults_to_tokenizer_or_zero():
    from seqtrainer.torch.dnabert2_benchmark import _ensure_pad_token_id

    config = SimpleNamespace()
    tokenizer = SimpleNamespace(pad_token_id=None)
    _ensure_pad_token_id(config, tokenizer)
    assert config.pad_token_id == 0

    tokenizer.pad_token_id = 7
    _ensure_pad_token_id(config, tokenizer)
    assert config.pad_token_id == 7


def test_dnabert2_setter_handles_config_without_pad_token_attribute():
    from seqtrainer.torch.dnabert2_benchmark import _set_pad_token_id

    class _StrictConfig:
        def __getattribute__(self, name):
            if name == "pad_token_id" and "pad_token_id" not in self.__dict__:
                raise AttributeError(name)
            return super().__getattribute__(name)

    config = _StrictConfig()
    with pytest.raises(AttributeError):
        _ = config.pad_token_id

    _set_pad_token_id(config, 3)

    assert config.pad_token_id == 3


def test_dnabert2_loader_uses_state_dict_fallback_for_meta_device_model(monkeypatch):
    import seqtrainer.torch.dnabert2_benchmark as dnabert2_benchmark
    from seqtrainer.torch.dnabert2_benchmark import _load_huggingface_dnabert2

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(pad_token_id=5)

    class _FakeConfig:
        def __init__(self):
            self.hidden_size = 4

        def update(self, payload):
            for key, value in payload.items():
                setattr(self, key, value)

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeConfig()

    class _FakeParameter:
        def __init__(self, is_meta):
            self.is_meta = is_meta

    class _FakeModel:
        def __init__(self, config, *, is_meta):
            self.config = config
            self.is_meta = is_meta

        def named_parameters(self):
            return [("encoder.weight", _FakeParameter(self.is_meta))]

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.eval_called = True
            return self

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel(kwargs["config"], is_meta=True)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    fallback_calls = []

    def _fake_state_dict_loader(*args, **kwargs):
        fallback_calls.append((args, kwargs))
        return _FakeModel(kwargs["config"], is_meta=False)

    monkeypatch.setattr(dnabert2_benchmark, "_load_dnabert2_from_state_dict", _fake_state_dict_loader)

    tokenizer, encoder = _load_huggingface_dnabert2(
        "fake/dnabert2",
        device="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )

    assert tokenizer.pad_token_id == 5
    assert encoder.is_meta is False
    assert encoder.config.pad_token_id == 5
    assert fallback_calls


def test_dnabert2_loader_skips_when_state_dict_fallback_still_has_meta_params(monkeypatch):
    import seqtrainer.torch.dnabert2_benchmark as dnabert2_benchmark
    from seqtrainer.benchmarks.runner import BenchmarkSkipped
    from seqtrainer.torch.dnabert2_benchmark import _load_huggingface_dnabert2

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return SimpleNamespace(pad_token_id=0)

    class _FakeConfig:
        hidden_size = 4

        def update(self, payload):
            for key, value in payload.items():
                setattr(self, key, value)

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeConfig()

    class _FakeParameter:
        is_meta = True

    class _MetaModel:
        config = _FakeConfig()

        def named_parameters(self):
            return [("encoder.weight", _FakeParameter())]

        def to(self, device):
            return self

        def eval(self):
            return self

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _MetaModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.AutoConfig = _FakeAutoConfig
    fake_transformers.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(dnabert2_benchmark, "_load_dnabert2_from_state_dict", lambda *args, **kwargs: _MetaModel())

    with pytest.raises(BenchmarkSkipped, match="state-dict fallback"):
        _load_huggingface_dnabert2(
            "fake/dnabert2",
            device="cpu",
            trust_remote_code=True,
            local_files_only=True,
        )


def test_dnabert2_tokenization_pipeline_uses_shared_splits_and_metadata(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "dnabert2_smoke.toml")
    _write_configured_split_files(config, tmp_path)

    result = prepare_dnabert2_tokenized_splits(
        config,
        base_dir=tmp_path,
        output_dir=tmp_path / "dnabert2_tokens",
        tokenizer=_StubTokenizer(),
    )

    assert result.metadata_path.exists()
    assert result.metadata["model_name"] == "zhihan1996/DNABERT-2-117M"
    assert result.metadata["max_length"] == 104
    assert set(result.tokenized_paths) == {"train", "validation", "test"}
    tokenized = pd.read_csv(result.tokenized_paths["train"])
    assert {"input_ids", "attention_mask", "token_count", "label"}.issubset(tokenized.columns)
    assert len(tokenized) == 4


def test_dnabert2_frozen_embedding_baseline_uses_encoder_and_caches_embeddings(tmp_path):
    torch = pytest.importorskip("torch")
    from seqtrainer.torch.dnabert2_benchmark import run_dnabert2_csv_splits

    config = load_benchmark_config(CONFIG_DIR / "dnabert2_frozen.toml")
    _write_configured_split_files(config, tmp_path)
    config = replace(
        config,
        training=replace(config.training, max_epochs=2, batch_size=2, learning_rate=0.01),
        model=replace(config.model, params={**dict(config.model.params), "classifier_dropout": 0.0}),
    )

    result = run_dnabert2_csv_splits(
        config,
        base_dir=tmp_path,
        output_dir=tmp_path / "dnabert2_frozen",
        tokenizer=_TorchStubTokenizer(torch),
        encoder=_TinyEncoder(torch),
    )

    assert result.status == "completed"
    assert (tmp_path / "dnabert2_frozen" / "embeddings" / "train_embeddings.pt").exists()
    assert (tmp_path / "dnabert2_frozen" / "checkpoints" / "best_model.pt").exists()
    assert (tmp_path / "dnabert2_frozen" / "history.csv").exists()
    history = pd.read_csv(tmp_path / "dnabert2_frozen" / "history.csv")
    assert {"train_loss", "validation_loss", "validation_mcc", "learning_rate"}.issubset(history.columns)
    assert result.manifest["model"]["metadata"]["embedding_cache_dir"]


def test_ipromp_benchmark_writes_fastas_and_skipped_manifest(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    _write_configured_split_files(config, tmp_path)

    result = run_benchmark(config, base_dir=tmp_path, output_dir=tmp_path / "ipromp")

    assert result.status == "skipped"
    assert (tmp_path / "ipromp" / "manifest.json").exists()
    assert (tmp_path / "ipromp" / "ipromp_fasta" / "train.fasta").exists()
    assert (tmp_path / "ipromp" / "ipromp_id_mapping.csv").exists()
    assert (tmp_path / "ipromp" / "ipromp_run_commands.sh").exists()
    assert (tmp_path / "ipromp" / "external_prediction_schema.md").exists()
    assert "FASTA prepared" in result.manifest["extra"]["skip_reason"]
    fasta_text = (tmp_path / "ipromp" / "ipromp_fasta" / "validation.fasta").read_text()
    assert ">seqtrainer|split=validation|row_index=0|sequence_id=validation_000000|label=0" in fasta_text


def test_prepare_ipromp_cli_writes_mapping_and_command_script(tmp_path, capsys):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    _write_configured_split_files(config, tmp_path)

    code = main(
        [
            "benchmark",
            "prepare-ipromp",
            str(CONFIG_DIR / "ipromp_external.toml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "prepared_ipromp"),
        ]
    )

    assert code == 0
    assert (tmp_path / "prepared_ipromp" / "ipromp_id_mapping.csv").exists()
    assert (tmp_path / "prepared_ipromp" / "ipromp_fasta" / "test.fasta").exists()
    command_text = (tmp_path / "prepared_ipromp" / "ipromp_run_commands.sh").read_text()
    assert "--species-id 10" in command_text
    assert "--max-length 300" in command_text
    assert "--batch-size 16" in command_text
    assert "seqtrainer.adapters.ipromp_inference" in command_text
    assert "mapping_csv=" in capsys.readouterr().out


def test_ipromp_fasta_writer_rejects_invalid_bases(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    for split, relative_path in config.dataset.split_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sequence = "ACGTXB" if split == "validation" else "ACGTACGT"
        pd.DataFrame(
            {
                config.dataset.sequence_field: [sequence],
                config.dataset.label_field: [1],
            }
        ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Invalid DNA bases"):
        main(
            [
                "benchmark",
                "prepare-ipromp",
                str(CONFIG_DIR / "ipromp_external.toml"),
                "--base-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "prepared_ipromp"),
            ]
        )


def test_ipromp_inference_reads_multiline_seqtrainer_fasta(tmp_path):
    from seqtrainer.adapters.ipromp_inference import read_seqtrainer_fasta

    fasta = tmp_path / "validation.fasta"
    fasta.write_text(
        ">seqtrainer|split=validation|row_index=0|sequence_id=validation_000000|label=1\n"
        "ACGT\n"
        "TGCA\n",
        encoding="utf-8",
    )

    records = read_seqtrainer_fasta(fasta, expected_split="validation")

    assert len(records) == 1
    assert records[0].sequence_id == "validation_000000"
    assert records[0].sequence == "ACGTTGCA"


def test_ipromp_inference_requires_all_five_fold_checkpoints(tmp_path):
    from seqtrainer.adapters.ipromp_inference import fold_checkpoint_paths

    for fold in range(1, 5):
        (tmp_path / f"10_fold_{fold}.pth").touch()

    with pytest.raises(FileNotFoundError, match="10_fold_5.pth"):
        fold_checkpoint_paths(tmp_path, species_id=10)


def test_ipromp_inference_normalizes_wrapped_parallel_state_dict():
    from seqtrainer.adapters.ipromp_inference import normalize_state_dict

    normalized = normalize_state_dict({"state_dict": {"module.fc1.weight": "value"}})

    assert normalized == {"fc1.weight": "value"}


def test_ipromp_mapping_honors_configured_string_labels():
    from seqtrainer.adapters.ipromp import build_ipromp_mapping

    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    config = replace(
        config,
        label=replace(config.label, negative_label="background", positive_label="promoter"),
    )
    frame = pd.DataFrame(
        {
            config.dataset.sequence_field: ["ACGTACGT", "TGCATGCA"],
            config.dataset.label_field: ["background", "promoter"],
        }
    )

    mapping = build_ipromp_mapping(
        config,
        {"train": frame, "validation": frame, "test": frame},
    )

    assert mapping["label"].tolist() == [0, 1, 0, 1, 0, 1]


def test_ipromp_external_predictions_are_evaluated_with_validation_threshold(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    _write_configured_split_files(config, tmp_path)
    predictions_path = tmp_path / "ipromp_predictions.csv"
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            [
                {"split": split, "label": 0, "score": 0.10},
                {"split": split, "label": 1, "score": 0.80},
                {"split": split, "label": 0, "score": 0.30},
                {"split": split, "label": 1, "score": 0.70},
            ]
        )
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    config_with_predictions = replace(
        config,
        model=replace(
            config.model,
            params={**dict(config.model.params), "predictions_csv": str(predictions_path)},
        ),
    )

    result = run_benchmark(config_with_predictions, base_dir=tmp_path, output_dir=tmp_path / "ipromp_eval")

    assert result.status == "completed"
    assert result.metrics["test"]["mcc"] == 1.0
    predictions = pd.read_csv(tmp_path / "ipromp_eval" / "predictions.csv")
    assert "probability" in predictions.columns
    assert "imbalance_policy" in result.manifest["extra"]


def test_ipromp_official_predictions_are_normalized_with_mapping(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    _write_configured_split_files(config, tmp_path)
    prep = main(
        [
            "benchmark",
            "prepare-ipromp",
            str(CONFIG_DIR / "ipromp_external.toml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "ipromp"),
        ]
    )
    assert prep == 0
    mapping = pd.read_csv(tmp_path / "ipromp" / "ipromp_id_mapping.csv")
    external_dir = tmp_path / "ipromp" / "external_predictions"
    external_dir.mkdir(exist_ok=True)
    for split in ("validation", "test"):
        split_mapping = mapping[mapping["split"] == split]
        pd.DataFrame(
            {
                "Sequence": split_mapping["sequence"],
                "Prediction": [0, 1, 0, 1],
                "Probability": [0.10, 0.80, 0.30, 0.70],
            }
        ).to_csv(external_dir / f"{split}_predictions.csv", index=False)
    config_with_predictions = replace(
        config,
        model=replace(
            config.model,
            params={
                **dict(config.model.params),
                "mapping_csv": str(tmp_path / "ipromp" / "ipromp_id_mapping.csv"),
                "validation_predictions_csv": str(external_dir / "validation_predictions.csv"),
                "test_predictions_csv": str(external_dir / "test_predictions.csv"),
            },
        ),
    )

    result = run_benchmark(config_with_predictions, base_dir=tmp_path, output_dir=tmp_path / "ipromp")

    assert result.status == "completed"
    assert result.metrics["test"]["mcc"] == 1.0
    predictions = pd.read_csv(tmp_path / "ipromp" / "predictions.csv")
    assert {"sequence_id", "row_index", "source_prediction", "threshold"}.issubset(predictions.columns)
    assert predictions[predictions["split"] == "test"]["threshold"].nunique() == 1


def test_ipromp_official_predictions_fail_on_duplicate_sequence_ambiguity(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    for split, relative_path in config.dataset.split_files.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                config.dataset.sequence_field: ["ACGTACGT", "ACGTACGT", "AAAACCCC", "GGGGTTTT"],
                config.dataset.label_field: [0, 1, 0, 1],
            }
        ).to_csv(path, index=False)
    main(
        [
            "benchmark",
            "prepare-ipromp",
            str(CONFIG_DIR / "ipromp_external.toml"),
            "--base-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "ipromp"),
        ]
    )
    external_dir = tmp_path / "ipromp" / "external_predictions"
    external_dir.mkdir(exist_ok=True)
    for split in ("validation", "test"):
        pd.DataFrame(
            {
                "Sequence": ["ACGTACGT", "AAAACCCC", "GGGGTTTT"],
                "Prediction": [1, 0, 1],
                "Probability": [0.80, 0.30, 0.70],
            }
        ).to_csv(external_dir / f"{split}_predictions.csv", index=False)
    config_with_predictions = replace(
        config,
        model=replace(
            config.model,
            params={
                **dict(config.model.params),
                "mapping_csv": str(tmp_path / "ipromp" / "ipromp_id_mapping.csv"),
                "validation_predictions_csv": str(external_dir / "validation_predictions.csv"),
                "test_predictions_csv": str(external_dir / "test_predictions.csv"),
            },
        ),
    )

    with pytest.raises(ValueError, match="duplicate sequences"):
        run_benchmark(config_with_predictions, base_dir=tmp_path, output_dir=tmp_path / "ipromp", allow_skip=False)


def test_ipromp_external_hard_labels_are_evaluated_without_faking_rank_metrics(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "ipromp_external.toml")
    _write_configured_split_files(config, tmp_path)
    predictions_path = tmp_path / "ipromp_predictions.tsv"
    rows = []
    for split in ("train", "validation", "test"):
        rows.extend(
            [
                {"split": split, "label": 0, "prediction": 0},
                {"split": split, "label": 1, "prediction": 1},
                {"split": split, "label": 0, "prediction": 0},
                {"split": split, "label": 1, "prediction": 1},
            ]
        )
    pd.DataFrame(rows).to_csv(predictions_path, index=False, sep="\t")
    config_with_predictions = replace(
        config,
        model=replace(
            config.model,
            params={**dict(config.model.params), "predictions_csv": str(predictions_path)},
        ),
    )

    result = run_benchmark(config_with_predictions, base_dir=tmp_path, output_dir=tmp_path / "ipromp_hard_labels")

    assert result.status == "completed"
    assert result.metrics["test"]["mcc"] == 1.0
    assert result.metrics["test"]["auroc"] is None
    assert result.metrics["test"]["auprc"] is None
    assert "hard labels" in result.metrics["test"]["warning"].lower()
    predictions = pd.read_csv(tmp_path / "ipromp_hard_labels" / "predictions.csv")
    assert "prediction" in predictions.columns
    assert predictions["threshold"].isna().all()


def test_benchmark_compare_cli_and_helper_rank_test_metrics(tmp_path, capsys):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    first = tmp_path / "model_a"
    second = tmp_path / "model_b"
    for out_dir, mcc, auprc in ((first, 0.2, 0.6), (second, 0.8, 0.9)):
        manifest = build_run_manifest(
            config,
            split_summary={"test": {"rows": 2, "class_counts": {"0": 1, "1": 1}}},
            threshold=0.5,
        )
        metrics = {
            "test": {
                "threshold": 0.5,
                "accuracy": 0.75,
                "balanced_accuracy": 0.75,
                "precision": 0.75,
                "recall": 0.75,
                "f1": 0.75,
                "mcc": mcc,
                "sensitivity": 0.75,
                "specificity": 0.75,
                "auroc": 0.8,
                "auprc": auprc,
                "confusion_matrix": {"tn": 1, "fp": 0, "fn": 1, "tp": 2},
            }
        }
        write_benchmark_outputs(out_dir, manifest=manifest, metrics=metrics, config=config)

    helper_written = compare_benchmark_outputs([first, second], output_dir=tmp_path / "comparison_helper")
    assert helper_written["comparison_metrics"].exists()
    helper_frame = pd.read_csv(helper_written["comparison_metrics"])
    assert helper_frame.iloc[0]["mcc"] == 0.8

    exit_code = main(["benchmark", "compare", str(first), str(second), "--output-dir", str(tmp_path / "comparison_cli")])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "comparison_summary" in captured.out
    assert (tmp_path / "comparison_cli" / "comparison_summary.md").exists()


def test_imbalance_policy_uses_training_split_only():
    split_summary = {
        "train": {"class_counts": {"0": 10, "1": 20}},
        "validation": {"class_counts": {"0": 1, "1": 100}},
    }

    policy = decide_imbalance_policy(split_summary, ratio_threshold=1.5)

    assert policy.apply_to_training is True
    assert policy.strategy == "class_weighting"
    assert policy.imbalance_ratio == 2.0

    with pytest.raises(ValueError, match="training split"):
        decide_imbalance_policy(split_summary, split="validation")


def test_imbalance_policy_does_not_weight_balanced_training_data():
    split_summary = {"train": {"class_counts": {"0": 20, "1": 22}}}

    policy = decide_imbalance_policy(split_summary, ratio_threshold=1.5)

    assert policy.apply_to_training is False
    assert policy.strategy == "none"


def test_threshold_metric_from_strategy_maps_validation_strategies():
    assert threshold_metric_from_strategy("validation_mcc") == "mcc"
    assert threshold_metric_from_strategy("validation_f1") == "f1"
    assert threshold_metric_from_strategy("validation_balanced_accuracy") == "balanced_accuracy"
    assert threshold_metric_from_strategy("fixed_0_5") is None


def test_ai_x_bio_fasta_parsing_and_source_split_preservation(tmp_path):
    from seqtrainer.benchmarks.ai_x_bio import prepare_ai_x_bio_splits

    source = tmp_path / "ai x bio.fasta"
    source.write_text(
        "\n".join(
            [
                ">seq1 label=promoter split=train",
                "acgu",
                ">seq2 label=negative split=train",
                "ttxx",
                ">seq3 label=1 split=validation",
                "cccc",
                ">seq4 label=0 split=validation",
                "gggg",
                ">seq5 label=positive split=test",
                "aaaa",
                ">seq6 label=non-promoter split=test",
                "nnnn",
            ]
        ),
        encoding="utf-8",
    )

    result = prepare_ai_x_bio_splits(source_file=source, output_dir=tmp_path / "prepared")

    assert result.metadata["source_format"] == "fasta"
    assert result.metadata["split_strategy"] == "preserved_source_split"
    train = pd.read_csv(result.split_paths["train"])
    assert list(train.columns) == ["sequence", "label", "id"]
    assert train.loc[0, "sequence"] == "ACGT"
    assert train["label"].tolist() == [1, 0]


def test_ai_x_bio_stratified_split_creation_and_schema(tmp_path):
    from seqtrainer.benchmarks.ai_x_bio import prepare_ai_x_bio_splits

    source = tmp_path / "ai x bio.csv"
    pd.DataFrame(
        {
            "seq": ["ACGT", "TGCA", "AAAA", "CCCC", "GGGG", "TTTT", "ACAC", "GTGT", "CACA", "TGTG"],
            "class": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    ).to_csv(source, index=False)

    result = prepare_ai_x_bio_splits(source_file=source, output_dir=tmp_path / "prepared", seed=42)

    assert result.metadata["split_strategy"] == "seeded_stratified_70_15_15"
    for split in ("train", "validation", "test"):
        frame = pd.read_csv(result.split_paths[split])
        assert list(frame.columns) == ["sequence", "label", "id"]
        assert set(frame["label"]).issubset({0, 1})
    assert result.metadata_path.exists()


def _write_configured_split_files(config, base_dir):
    for split, relative_path in config.dataset.split_files.items():
        path = base_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                config.dataset.sequence_field: ["ACGTACGT", "TGCATGCA", "AAAACCCC", "GGGGTTTT"],
                config.dataset.label_field: [0, 1, 0, 1],
                "split": split,
            }
        ).to_csv(path, index=False)


class _StubTokenizer:
    def __call__(
        self,
        sequences,
        *,
        padding,
        truncation,
        max_length,
        pad_to_multiple_of=None,
    ):
        tokenized = []
        masks = []
        for sequence in sequences:
            ids = [ord(base) % 7 + 1 for base in str(sequence)[:max_length]]
            if pad_to_multiple_of:
                target_length = ((len(ids) + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
            else:
                target_length = len(ids)
            mask = [1] * len(ids) + [0] * (target_length - len(ids))
            ids = ids + [0] * (target_length - len(ids))
            tokenized.append(ids)
            masks.append(mask)
        return {"input_ids": tokenized, "attention_mask": masks}

    def __len__(self):
        return 7


class _TorchStubTokenizer:
    def __init__(self, torch):
        self.torch = torch

    def __call__(self, sequences, **kwargs):
        max_length = int(kwargs.get("max_length", 8))
        ids, masks = [], []
        for sequence in sequences:
            row = [ord(base) % 7 + 1 for base in str(sequence)[:max_length]]
            mask = [1] * len(row)
            if kwargs.get("padding") in {"longest", "max_length"}:
                target = max_length if kwargs.get("padding") == "max_length" else max(len(row), 1)
                row = row + [0] * (target - len(row))
                mask = mask + [0] * (target - len(mask))
            ids.append(row)
            masks.append(mask)
        return {
            "input_ids": self.torch.tensor(ids, dtype=self.torch.long),
            "attention_mask": self.torch.tensor(masks, dtype=self.torch.long),
        }


class _TinyEncoder:
    def __init__(self, torch):
        self.torch = torch
        self.config = SimpleNamespace(hidden_size=4)

    def parameters(self):
        return []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, *, input_ids, attention_mask):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4) / 10.0
        return SimpleNamespace(last_hidden_state=hidden)


def test_split_summary_supports_configured_string_labels(tmp_path):
    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    config = replace(config, label=replace(config.label, negative_label="background", positive_label="promoter"))
    for filename in config.dataset.split_files.values():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"sequence": ["ACGT", "TGCA", "AAAA"], "label": ["background", "promoter", "promoter"]}
        ).to_csv(path, index=False)

    summary = summarize_split_frames(config, load_predefined_split_frames(config, base_dir=tmp_path))

    assert summary["train"]["class_counts"] == {"background": 1, "promoter": 2}


def test_direct_cnn_cli_propagates_configured_cnn_v2_params(tmp_path, monkeypatch):
    import seqtrainer.torch.cnn_baseline as cnn_baseline

    captured = {}

    def fake_run(run_config):
        captured["config"] = run_config
        return SimpleNamespace(
            output_dir=tmp_path,
            metrics={},
            manifest={"threshold_selection": {"threshold": 0.5}},
        )

    monkeypatch.setattr(cnn_baseline, "run_cnn_csv_splits", fake_run)

    assert main(["run-cnn-benchmark", "--config", str(CONFIG_DIR / "cnn_v2.toml")]) == 0
    run_config = captured["config"]
    assert run_config.model_variant == "enhanced"
    assert run_config.optimizer_name == "adamw"
    assert run_config.scheduler_name == "one_cycle"
    assert run_config.threshold_strategy == "validation_mcc"


def test_cnn_runner_preserves_explicit_zero_training_values(tmp_path, monkeypatch):
    import seqtrainer.benchmarks.runner as benchmark_runner
    import seqtrainer.torch.cnn_baseline as cnn_baseline

    config = load_benchmark_config(CONFIG_DIR / "cnn.toml")
    config = replace(config, training=replace(config.training, max_epochs=0, learning_rate=0.0))
    captured = {}
    monkeypatch.setattr(
        benchmark_runner,
        "resolve_split_paths",
        lambda *args, **kwargs: {
            "train": tmp_path / "train.csv",
            "validation": tmp_path / "validation.csv",
            "test": tmp_path / "test.csv",
        },
    )

    def fake_run(run_config):
        captured["config"] = run_config
        return SimpleNamespace(output_dir=tmp_path, metrics={}, manifest={})

    monkeypatch.setattr(cnn_baseline, "run_cnn_csv_splits", fake_run)

    result = benchmark_runner.run_benchmark(config, base_dir=tmp_path)

    assert result.status == "completed"
    assert captured["config"].cycles == 0
    assert captured["config"].learning_rate == 0.0


def test_direct_cnn_cli_preserves_zero_overrides(tmp_path, monkeypatch):
    import seqtrainer.torch.cnn_baseline as cnn_baseline

    captured = {}

    def fake_run(run_config):
        captured["config"] = run_config
        return SimpleNamespace(
            output_dir=tmp_path,
            metrics={},
            manifest={"threshold_selection": {"threshold": 0.5}},
        )

    monkeypatch.setattr(cnn_baseline, "run_cnn_csv_splits", fake_run)

    assert main(
        [
            "run-cnn-benchmark",
            "--config",
            str(CONFIG_DIR / "cnn.toml"),
            "--seed",
            "0",
            "--cycles",
            "0",
            "--learning-rate",
            "0",
        ]
    ) == 0
    assert captured["config"].seed == 0
    assert captured["config"].cycles == 0
    assert captured["config"].learning_rate == 0.0

