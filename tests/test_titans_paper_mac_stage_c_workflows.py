from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import materialize_token_stream_dataset  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c import SeqTrainerBaseTokenizer  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.colab_cli import main as colab_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.capacity_cli import (  # noqa: E402
    _write_capacity_artifacts,
    parse_args as parse_capacity_args,
)
from seqtrainer.torch.titans_paper_mac_stage_c.evaluate_cli import main as evaluate_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.resume_verify_cli import (  # noqa: E402
    main as resume_verify_main,
)
from seqtrainer.torch.titans_paper_mac_stage_c.train_cli import main as train_main  # noqa: E402


def test_colab_wrapper_persists_streamed_log_manifest_and_failure_marker(tmp_path) -> None:
    run_dir = tmp_path / "run"
    assert colab_main(
        [
            "--run-dir",
            str(run_dir),
            "--label",
            "passing",
            "--",
            sys.executable,
            "-c",
            "print('persisted output')",
        ]
    ) == 0
    assert "persisted output" in (run_dir / "logs" / "passing.log").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "colab_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"][-1]["status"] == "passed"
    assert not (run_dir / "FAILED.txt").exists()

    with pytest.raises(SystemExit):
        colab_main(
            [
                "--run-dir",
                str(run_dir),
                "--label",
                "failing",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(7)",
            ]
        )
    assert "failing" in (run_dir / "FAILED.txt").read_text(encoding="utf-8")
    manifest = json.loads((run_dir / "colab_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"][-1]["return_code"] == 7
    assert manifest["steps"][-1]["status"] == "failed"


def test_capacity_artifacts_are_safe_to_append_after_an_isolated_probe(tmp_path) -> None:
    args = parse_capacity_args(
        [
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "capacity"),
            "--require",
            "A100",
            "--horizons",
            "2",
            "--variants",
            "exact_sdpa_fp32",
            "--append",
        ]
    )
    args.output_dir.mkdir()
    result = {
        "horizon": 2,
        "variant": "exact_sdpa_fp32",
        "available": True,
        "validation_bpb": 2.0,
        "bases_per_second": 100.0,
        "peak_allocated_bytes": 1_000_000,
        "functional_state_bytes_per_stream": 512,
        "projected_one_train_pass_hours": 1.0,
        "written_state_gradient_norm": 3.0,
    }
    payload = _write_capacity_artifacts(
        args=args,
        device_name="NVIDIA A100",
        train_predictable_bases=1000,
        results=[result],
    )

    assert payload["results"] == [result]
    assert json.loads((args.output_dir / "capacity_matrix.json").read_text())["results"] == [result]
    for name in (
        "capacity_throughput.svg",
        "capacity_memory.svg",
        "capacity_validation_bpb.svg",
        "capacity_matrix.md",
    ):
        assert (args.output_dir / name).is_file(), name


def test_production_stream_dataset_trains_checkpoints_and_reports_on_cpu(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    run_dir = tmp_path / "run"
    tokenizer = SeqTrainerBaseTokenizer()
    records = [
        {
            "accession": "train-1",
            "contig_id": "chromosome",
            "sequence": "ACGT" * 24,
            "split": "train",
            "clade_group": "ani99:train-1",
        },
        {
            "accession": "val-1",
            "contig_id": "chromosome",
            "sequence": "TGCA" * 12,
            "split": "val",
            "clade_group": "ani99:val-1",
        },
    ]
    materialize_token_stream_dataset(records, tokenizer, dataset_dir, tokens_per_shard=256)

    assert train_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--run-dir",
            str(run_dir),
            "--max-optimizer-steps",
            "1",
            "--checkpoint-every",
            "1",
            "--validation-streams",
            "1",
            "--device",
            "cpu",
            "--horizon",
            "1",
            "--block-count",
            "1",
            "--d-model",
            "4",
            "--num-heads",
            "2",
            "--persistent-tokens",
            "2",
            "--memory-depth",
            "1",
            "--memory-surprise-clip-norm",
            "none",
            "--memory-associative-loss-reduction",
            "mean",
            "--memory-max-gradient-rms-ratio",
            "10",
            "--memory-theta-max",
            "0.5",
            "--memory-theta-initial",
            "0.25",
        ]
    ) == 0

    for name in (
        "latest.pt",
        "run_manifest.json",
        "validation.json",
        "training_history.json",
        "training_bpb.svg",
        "memory_diagnostics.svg",
        "gradient_diagnostics.svg",
        "memory_conditioning.svg",
        "gate_diagnostics.svg",
        "training_throughput.svg",
    ):
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    live_status = json.loads((run_dir / "LIVE_STATUS.json").read_text(encoding="utf-8"))
    assert manifest["optimizer_steps"] == 1
    assert manifest["learning_rate"] == pytest.approx(3e-5)
    assert manifest["gradient_clip_norm"] == pytest.approx(0.5)
    assert manifest["memory_surprise_clip_norm"] is None
    assert manifest["memory_associative_loss_reduction"] == "mean"
    assert manifest["memory_max_gradient_rms_ratio"] == pytest.approx(10.0)
    assert manifest["memory_theta_max"] == pytest.approx(0.5)
    assert manifest["memory_theta_initial"] == pytest.approx(0.25)
    assert live_status["state"] == "completed"
    assert manifest["processed_bases"] > 0
    assert manifest["validation"]["bits_per_base"] > 0
    assert manifest["validation"]["perplexity"] > 0
    assert manifest["validation"]["per_gc_bin_bpb"]
    assert manifest["validation"]["gate_statistics"]
    assert manifest["validation"]["memory_gradient_statistics"]
    assert manifest["stop_reason"] == "optimizer_step_budget"
    verification_path = tmp_path / "resume_verification.json"
    assert resume_verify_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--checkpoint",
            str(run_dir / "latest.pt"),
            "--output",
            str(verification_path),
            "--device",
            "cpu",
            "--expected-step",
            "1",
        ]
    ) == 0
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    assert verification["read_only_source_checkpoint"] is True
    assert verification["deterministic_continuation"] is True
    assert (
        verification["first_continuation"]["continued_optimizer_step"]
        == verification["second_continuation"]["continued_optimizer_step"]
        == 2
    )
    with pytest.raises(ValueError, match="separately trained runs"):
        evaluate_main(
            [
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--run",
                f"adaptive={run_dir / 'latest.pt'}",
            ]
        )


def test_all_colab_notebooks_are_thin_pinned_logged_handoffs() -> None:
    root = Path(__file__).parents[1]
    notebooks = sorted((root / "notebooks" / "titans_stage_c").glob("*.ipynb"))
    assert len(notebooks) == 6
    for notebook in notebooks:
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell.get("cell_type") == "code"
        )
        assert "GIT_REF" in source
        assert "seqtrainer-titans-stage-c-colab-run" in source
        assert "DRIVE_ROOT" in source
        assert len(payload["cells"]) <= 5


def test_stream_dataset_notebook_preserves_the_colab_numeric_abi_stack() -> None:
    root = Path(__file__).parents[1]
    notebook = json.loads(
        (root / "notebooks" / "titans_stage_c" / "00b_stage_c_stream_dataset.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "virtualenv" in source
    assert "--system-site-packages" in source
    assert "bootstrap.log" in source
    assert "run_bootstrap" in source
    assert "stage_c_python" in source
    assert "stage_c_runner" in source
    assert "--no-deps" in source
    assert "numpy==1.26.4" not in source
    assert "pandas==2.2.2" not in source
    assert "pyarrow==18.1.0" not in source
    assert "[sys.executable,'-m','venv'" not in source
