from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import materialize_token_stream_dataset  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c import SeqTrainerBaseTokenizer  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.colab_cli import main as colab_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.evaluate_cli import main as evaluate_main  # noqa: E402
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
        "gate_diagnostics.svg",
        "training_throughput.svg",
    ):
        assert (run_dir / name).exists(), name
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["optimizer_steps"] == 1
    assert manifest["processed_bases"] > 0
    assert manifest["validation"]["bits_per_base"] > 0
    assert manifest["validation"]["perplexity"] > 0
    assert manifest["validation"]["per_gc_bin_bpb"]
    assert manifest["validation"]["gate_statistics"]
    assert manifest["stop_reason"] == "optimizer_step_budget"
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


def test_stream_dataset_notebook_uses_an_isolated_colab_numeric_abi_stack() -> None:
    root = Path(__file__).parents[1]
    notebook = json.loads(
        (root / "notebooks" / "titans_stage_c" / "00b_stage_c_stream_dataset.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "--system-site-packages" in source
    assert "stage_c_python" in source
    assert "stage_c_runner" in source
    assert "--force-reinstall" in source
    assert "numpy==1.26.4" in source
    assert "pandas==2.2.2" in source
    assert "pyarrow==18.1.0" in source
    assert "[sys.executable,'-m','pip'" not in source
