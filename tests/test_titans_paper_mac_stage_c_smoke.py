from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import build_stream_segments  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_c import (  # noqa: E402
    SeqTrainerBaseTokenizer,
    StageCModelConfig,
    StageCPaperMACForCausalLM,
)
from seqtrainer.torch.titans_paper_mac_stage_c.smoke_cli import (  # noqa: E402
    _forward_backward,
    parse_args,
    resolve_stage_c_dataset,
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
