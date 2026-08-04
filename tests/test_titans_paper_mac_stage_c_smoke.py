from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import build_stream_segments  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c import (  # noqa: E402
    MemoryMode,
    SeqTrainerBaseTokenizer,
    StageCModelConfig,
    StageCPaperMACForCausalLM,
)
from seqtrainer.torch.titans_paper_mac_stage_c.smoke_cli import (  # noqa: E402
    _forward_backward,
    parse_args,
    resolve_stage_c_dataset,
)
from seqtrainer.torch.titans_paper_mac_stage_c.t4_evidence_cli import (  # noqa: E402
    build_t4_evidence,
    write_t4_evidence,
)


def test_smoke_cli_uses_the_production_t4_geometry_by_default(tmp_path) -> None:
    args = parse_args(
        ["--dataset-dir", str(tmp_path / "dataset"), "--output", str(tmp_path / "gpu_smoke.json")]
    )

    assert args.require == "T4"
    assert args.block_count == 8
    assert args.d_model == 384
    assert args.num_heads == 8
    assert args.persistent_tokens == 4
    assert args.memory_depth == 1
    assert args.gradient_horizon == 3


def test_resolve_stage_c_dataset_uses_the_frozen_tokenizer_selection(tmp_path) -> None:
    selection = tmp_path / "runs" / "c1_tokenizers_cpu" / "tokenizer_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"selected_tokenizer": "nonoverlap_6mer_v1"}))
    dataset = tmp_path / "stage_c_dataset" / "ordered_streams" / "nonoverlap_6mer_v1"
    dataset.mkdir(parents=True)
    (dataset / "token_stream_manifest.json").write_text("{}")

    assert resolve_stage_c_dataset(tmp_path) == dataset


def test_resolve_stage_c_dataset_explains_when_00b_has_not_built_it(tmp_path) -> None:
    selection = tmp_path / "runs" / "c1_tokenizers_cpu" / "tokenizer_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"selected_tokenizer": "nonoverlap_6mer_v1"}))

    with pytest.raises(FileNotFoundError, match="run Notebook 00b first"):
        resolve_stage_c_dataset(tmp_path)


def test_parity_probe_allows_first_segment_write_only_parameters() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
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
    segments = build_stream_segments(
        sequence="ACGT" * 16,
        accession="smoke",
        contig_id="contig",
        split="train",
        clade_group="smoke",
        tokenizer=tokenizer,
    )

    _, loss, gradients, inactive = _forward_backward(
        StageCPaperMACForCausalLM(config), segments[0], torch.device("cpu")
    )

    assert torch.isfinite(loss)
    assert any(".attention." in name for name in gradients)
    assert not any(".attention." in name for name in inactive)
    assert any(".memory." in name for name in inactive)


def test_no_grad_causal_probe_uses_attention_without_functional_memory_writes() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
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
    segment = build_stream_segments(
        sequence="ACGT" * 16,
        accession="causal-smoke",
        contig_id="contig",
        split="train",
        clade_group="causal-smoke",
        tokenizer=tokenizer,
    )[0]
    model = StageCPaperMACForCausalLM(config)
    inputs = torch.tensor([segment.input_ids], dtype=torch.long)
    valid_mask = torch.tensor([segment.valid_mask], dtype=torch.bool)

    with torch.no_grad():
        output = model.forward_segment(
            (model.initial_states("causal-smoke"),),
            inputs,
            valid_mask=valid_mask,
            memory_mode=MemoryMode.NONE,
        )

    assert torch.isfinite(output.logits).all()


def test_t4_evidence_preserves_the_passed_smoke_contract(tmp_path) -> None:
    smoke = {
        "classification": "stage_c_gpu_smoke",
        "passed": True,
        "hardware": {"device_name": "Tesla T4"},
        "geometry": {"block_count": 8, "d_model": 384, "gradient_horizon": 3},
        "dataset": {"path": "/content/dataset"},
        "fp32_training": {"optimizer_steps": 1, "valid_bases": 96},
        "fp16_training": {"optimizer_steps": 1, "valid_bases": 96},
        "fp32_cpu_gpu_parity": {"passed": True},
        "fp16_causal_mask": {"passed": True},
    }
    smoke_path = tmp_path / "gpu_smoke.json"
    smoke_path.write_text(json.dumps(smoke))

    evidence = build_t4_evidence(smoke)
    paths = write_t4_evidence(smoke_path, tmp_path / "evidence")

    assert evidence["passed"] is True
    assert "deferred_to_a100" in evidence["scope"]
    assert all(path.exists() for path in paths.values())


def test_t4_evidence_rejects_a_failed_smoke_report() -> None:
    with pytest.raises(ValueError, match="passed stage_c_gpu_smoke"):
        build_t4_evidence({"classification": "stage_c_gpu_smoke", "passed": False})
