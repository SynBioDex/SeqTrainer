from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_paper_mac_stage_c.architecture_cli import (  # noqa: E402
    main,
)
from seqtrainer.torch.titans_paper_mac_stage_c.model import (  # noqa: E402
    StageCPaperMACForCausalLM,
)
from tests.test_titans_paper_mac_stage_c_model import paper_deep_config  # noqa: E402


def test_architecture_cli_reconstructs_the_checkpointed_paper_deep_model(
    tmp_path, capsys
) -> None:
    config = paper_deep_config(recurrence="paper_exact")
    model = StageCPaperMACForCausalLM(config)
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {
            "format_version": 2,
            "code_commit": "test-commit",
            "model_config": config.to_dict(),
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    output_dir = tmp_path / "architecture"
    assert main(["--checkpoint", str(checkpoint), "--output-dir", str(output_dir)]) == 0

    report = json.loads((output_dir / "MODEL_ARCHITECTURE.json").read_text())
    text = (output_dir / "MODEL_ARCHITECTURE.txt").read_text()
    assert report["code_commit"] == "test-commit"
    assert report["model_config"]["memory_architecture"] == "paper_residual_mlp_v2"
    assert report["model_config"]["memory_depth"] == 2
    assert report["tied_input_output_embeddings"] is True
    assert report["total_parameters"] == sum(parameter.numel() for parameter in model.parameters())
    memory = report["functional_memory_by_block"][0]
    assert memory["memory_module"] == "PaperResidualMemory"
    assert memory["gate_module"] == "PerLayerChannelUpdateGates"
    assert memory["projection_convolution_kernel"] == 4
    assert memory["projection_history_shapes"] == [[3, 4], [3, 4]]
    assert memory["functional_state_bytes_per_stream"] > 0
    assert "StageCPaperMACForCausalLM" in text
    assert "PaperResidualMemory" in text
    assert "PerLayerChannelUpdateGates" in text
    assert "Stage C checkpoint architecture" in capsys.readouterr().out
