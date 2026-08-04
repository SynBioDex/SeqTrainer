from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import (  # noqa: E402
    TokenStreamDataset,
    materialize_token_stream_dataset,
)
from seqtrainer.torch.titans_paper_mac_stage_c import (  # noqa: E402
    SeqTrainerBaseTokenizer,
    evaluate_ordered_streams,
)
from seqtrainer.torch.titans_paper_mac_stage_c.colab_cli import main as colab_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.capacity_cli import (  # noqa: E402
    _write_capacity_artifacts,
    parse_args as parse_capacity_args,
)
from seqtrainer.torch.titans_paper_mac_stage_c.evaluate_cli import main as evaluate_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.generation_cli import (  # noqa: E402
    _find_orfs,
    _jensen_shannon,
    _kmer_counts,
    _sequence_metrics,
    generate_continuation,
    parse_args as parse_generation_args,
)
from seqtrainer.torch.titans_paper_mac_stage_c.memory_trace_cli import (  # noqa: E402
    _pca,
    _scatter_svg,
    _taxonomy_labels,
)
from seqtrainer.torch.titans_paper_mac_stage_c.resume_verify_cli import (  # noqa: E402
    main as resume_verify_main,
)
from seqtrainer.torch.titans_paper_mac_stage_c.train_cli import main as train_main  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.config import StageCModelConfig  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c.model import (  # noqa: E402
    StageCPaperMACForCausalLM,
)


def test_generation_distribution_and_orf_diagnostics_are_deterministic() -> None:
    reference = "ACGT" * 30
    divergent = "A" * 120

    assert _jensen_shannon(_kmer_counts([reference], 3), _kmer_counts([reference], 3)) == 0.0
    assert _jensen_shannon(_kmer_counts([reference], 3), _kmer_counts([divergent], 3)) > 0.0
    metrics = _sequence_metrics("example", reference, "reference")
    assert metrics["gc_fraction"] == 0.5
    assert metrics["max_homopolymer"] == 1
    assert metrics["aligned_unique_6mer_fraction"] == 0.1

    coding = "ATG" + "AAA" * 29 + "TAA"
    orfs = _find_orfs(coding)
    assert any(record["length"] == 93 for record in orfs)
    coding_metrics = _sequence_metrics("coding", coding, "reference")
    assert coding_metrics["heuristic_orfs_at_least_90bp"] >= 1
    assert coding_metrics["heuristic_longest_orf_bases"] >= 93


def test_evaluation_is_accession_balanced_resumable_and_contract_locked(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    tokenizer = SeqTrainerBaseTokenizer()
    records = [
        {
            "accession": accession,
            "contig_id": "chromosome",
            "sequence": sequence,
            "split": "val",
            "clade_group": f"ani99:{accession}",
        }
        for accession, sequence in (
            ("val-a", "ACGT" * 150),
            ("val-b", "TGCA" * 150),
        )
    ]
    materialize_token_stream_dataset(records, tokenizer, dataset_dir, tokens_per_shard=512)
    streams = TokenStreamDataset(dataset_dir).streams(split="val")
    config = StageCModelConfig(
        vocab_size=tokenizer.spec.vocab_size,
        pad_token_id=tokenizer.spec.pad_token_id,
        tokenizer_name=tokenizer.spec.name,
        tokenizer_checksum=tokenizer.spec.checksum,
        block_count=1,
        d_model=4,
        num_heads=2,
        persistent_tokens=2,
        memory_depth=1,
        gradient_horizon=1,
    )
    base = StageCPaperMACForCausalLM(config)
    uninterrupted_model = copy.deepcopy(base)
    resumed_model = copy.deepcopy(base)
    common = {
        "device": torch.device("cpu"),
        "max_segments_per_accession": 3,
        "checkpoint_every_segments": 1,
        "progress_every_segments": 1,
        "resume_contract": {"fixture": "balanced-v1"},
    }
    expected = evaluate_ordered_streams(
        uninterrupted_model,
        streams,
        progress_path=tmp_path / "expected_status.json",
        run_label="expected",
        **common,
    )
    checkpoint = tmp_path / "resume" / "evaluation.pt"
    original_forward = resumed_model.forward_segment
    calls = 0

    def interrupted_forward(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated Colab disconnect")
        return original_forward(*args, **kwargs)

    resumed_model.forward_segment = interrupted_forward  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated Colab disconnect"):
        evaluate_ordered_streams(
            resumed_model,
            streams,
            resume_checkpoint=checkpoint,
            progress_path=tmp_path / "resumed_status.json",
            run_label="resumed",
            **common,
        )
    interrupted_status = json.loads(
        (tmp_path / "resumed_status.json").read_text(encoding="utf-8")
    )
    assert interrupted_status["state"] == "interrupted"
    assert interrupted_status["completed_segments"] == 1
    assert checkpoint.is_file()

    resumed_model.forward_segment = original_forward  # type: ignore[method-assign]
    actual = evaluate_ordered_streams(
        resumed_model,
        streams,
        resume_checkpoint=checkpoint,
        progress_path=tmp_path / "resumed_status.json",
        run_label="resumed",
        **common,
    )

    assert actual.to_dict() == expected.to_dict()
    assert actual.segments == 6
    assert set(actual.per_accession_bpb) == {"val-a", "val-b"}
    assert actual.per_accession_segments == {"val-a": 3, "val-b": 3}
    completed_status = json.loads(
        (tmp_path / "resumed_status.json").read_text(encoding="utf-8")
    )
    assert completed_status["state"] == "completed"
    assert completed_status["resumed"] is True
    assert completed_status["progress_fraction"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="contract changed"):
        evaluate_ordered_streams(
            resumed_model,
            streams,
            resume_checkpoint=checkpoint,
            resume_contract={"fixture": "different"},
            max_segments_per_accession=3,
            device="cpu",
        )


def test_generation_cli_accepts_an_unrestricted_top_k(tmp_path) -> None:
    args = parse_generation_args(
        [
            "--dataset-dir",
            str(tmp_path / "dataset"),
            "--taxonomy-manifest",
            str(tmp_path / "taxonomy.csv"),
            "--checkpoint",
            str(tmp_path / "latest.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--top-k",
            "none",
        ]
    )

    assert args.top_k is None


def test_generation_preserves_the_inner_surprise_gradient() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    config = StageCModelConfig.paper_deep(
        vocab_size=tokenizer.spec.vocab_size,
        pad_token_id=tokenizer.spec.pad_token_id,
        tokenizer_name=tokenizer.spec.name,
        tokenizer_checksum=tokenizer.spec.checksum,
        recurrence_policy="paper_exact",
        block_count=1,
        d_model=4,
        num_heads=2,
        persistent_tokens=2,
        gradient_horizon=1,
    )
    model = StageCPaperMACForCausalLM(config).eval()

    generated, diagnostics = generate_continuation(
        model,
        [2, 3],
        new_tokens=2,
        temperature=1.0,
        top_k=2,
        top_p=1.0,
        seed=17,
        device=torch.device("cpu"),
    )

    assert len(generated) == 2
    assert all(np.isfinite(value) for value in diagnostics.values())


def test_memory_trace_pca_uses_numpy_singular_values_correctly() -> None:
    features = np.array(
        [[1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [3.0, 0.0, -1.0]],
        dtype=float,
    )

    points, variance = _pca(features)

    assert points.shape == (3, 2)
    assert len(variance) == 2
    assert all(0.0 <= value <= 1.0 for value in variance)
    assert sum(variance) <= 1.0


def test_embedding_taxonomy_labels_and_svg_are_explicit(tmp_path) -> None:
    manifest = tmp_path / "accession_manifest.csv"
    manifest.write_text(
        "accession,gtdb_taxonomy\n"
        "GCF_1,d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_coli\n"
        "GCF_2,d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_albertii\n",
        encoding="utf-8",
    )

    labels = _taxonomy_labels(manifest, "species")
    svg = _scatter_svg(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        [labels["GCF_1"], labels["GCF_2"]],
        [0.75, 0.25],
        title="Contextual sequence-embedding PCA",
        color_label="GTDB species",
    )

    assert labels == {"GCF_1": "Escherichia coli", "GCF_2": "Escherichia albertii"}
    assert "Contextual sequence-embedding PCA" in svg
    assert "GTDB species" in svg
    assert "Escherichia coli" in svg


def test_embedding_taxonomy_uses_the_canonical_stage_c_assembly_accession(tmp_path) -> None:
    manifest = tmp_path / "source_accession_manifest.csv"
    manifest.write_text(
        "accession,assembly_accession,gtdb_taxonomy\n"
        "GB_GCA_000249155.2,GCA_000249155.2,d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia_coli\n",
        encoding="utf-8",
    )

    assert _taxonomy_labels(manifest, "species") == {
        "GCA_000249155.2": "Escherichia coli"
    }


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

    # A hard Colab disconnect cannot run the wrapper's finalizer.  On the next
    # invocation, the stale same-label attempt must no longer remain "running".
    manifest["steps"][-1]["status"] = "running"
    (run_dir / "colab_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert colab_main(
        [
            "--run-dir",
            str(run_dir),
            "--label",
            "passing",
            "--",
            sys.executable,
            "-c",
            "print('resumed invocation')",
        ]
    ) == 0
    manifest = json.loads((run_dir / "colab_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["steps"][-2]["status"] == "interrupted"
    assert manifest["steps"][-1]["status"] == "passed"

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
    evaluation_dir = tmp_path / "evaluation_partial"
    assert evaluate_main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--output-dir",
            str(evaluation_dir),
            "--run",
            f"c16={run_dir / 'latest.pt'}",
            "--comparison-mode",
            "partial",
            "--max-segments-per-accession",
            "1",
            "--resume",
            "--checkpoint-every-segments",
            "1",
            "--progress-every-segments",
            "1",
            "--device",
            "cpu",
        ]
    ) == 0
    evaluation = json.loads((evaluation_dir / "evaluation.json").read_text())
    evaluation_status = json.loads((evaluation_dir / "c16_LIVE_STATUS.json").read_text())
    assert evaluation["execution"]["resumable"] is True
    assert evaluation["execution"]["max_segments_per_accession"] == 1
    assert evaluation["results"]["c16"]["segments"] == 1
    assert evaluation_status["state"] == "completed"
    assert not (evaluation_dir / "resume" / "c16.evaluation.pt").exists()
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
    assert len(notebooks) >= 7
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


def test_v3_training_and_evaluation_notebooks_preserve_resume_paths() -> None:
    root = Path(__file__).parents[1] / "notebooks" / "titans_stage_c"

    def source(name: str) -> str:
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        return "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])

    baseline = source("03j_stage_c_v3_freeze_panels_and_c16_baseline.ipynb")
    assert "--resume" in baseline
    assert "PILOT_SEGMENTS_PER_ACCESSION=256" in baseline
    assert "RUN_FULL_A100=False" in baseline
    assert "c17_v3_c16_broad_baseline_resumable" in baseline

    e25 = source("03l_stage_c_v3_medium_adaptive_e25.ipynb")
    assert "RUN_SEED=str(RUN_SPEC['seed'])" in e25
    assert "'--seed',RUN_SEED" in e25

    e100 = source("03n_stage_c_v3_medium_adaptive_e100_increment.ipynb")
    assert "startup=[] if resume_checkpoint.is_file()" in e100
    assert "else ['--warm-start-checkpoint',str(parent),'--no-resume']" in e100
    assert "RUN_SEED=str(RUN_SPEC['seed'])" in e100
    assert "'--seed',RUN_SEED" in e100
