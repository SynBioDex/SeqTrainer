import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch import (
    DNATokenizer,
    TitansMIRASConfig,
    TitansMemoryAsContextClassifier,
)


def test_titans_import_smoke() -> None:
    cfg = TitansMIRASConfig()
    assert cfg.memory_architecture == "mlp"


def test_dna_tokenizer_shapes() -> None:
    tok = DNATokenizer(max_length=8)
    input_ids, mask = tok.batch_encode(["ACGTN", "TGCA"])
    assert input_ids.shape == (2, 8)
    assert mask.shape == (2, 8)
    assert input_ids[0, 0].item() == 2
    assert input_ids[0, 4].item() == DNATokenizer.UNK_TOKEN_ID


def test_forward_and_loss_shapes() -> None:
    cfg = TitansMIRASConfig(d_model=32, num_heads=4, num_layers=1, max_length=16, memory_slots=6, memory_context_tokens=3)
    model = TitansMemoryAsContextClassifier(cfg)
    tok = DNATokenizer(max_length=16)
    input_ids, mask = tok.batch_encode(["ACGT" * 4, "TGCATGCA", "NNNN"])

    logits = model(input_ids=input_ids, attention_mask=mask)
    assert logits.shape == (3, cfg.num_classes)

    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
    assert out["loss"].ndim == 0
    assert out["logits"].shape == (3, cfg.num_classes)


def test_memory_reset_and_update() -> None:
    cfg = TitansMIRASConfig(d_model=32, num_heads=4, num_layers=1, max_length=8, memory_slots=4, memory_context_tokens=2)
    model = TitansMemoryAsContextClassifier(cfg)
    tok = DNATokenizer(max_length=8)
    input_ids, mask = tok.batch_encode(["ACGTACGT", "TTTTAAAA"])

    before = model.long_term_memory.memory_state.clone()
    _ = model(input_ids=input_ids, attention_mask=mask, update_memory=True)
    after = model.long_term_memory.memory_state.clone()
    assert not torch.allclose(before, after)

    model.reset_memory()
    reset = model.long_term_memory.memory_state.clone()
    assert torch.allclose(reset, torch.zeros_like(reset))


def test_tiny_cpu_optimization_step() -> None:
    cfg = TitansMIRASConfig(d_model=32, num_heads=4, num_layers=1, max_length=12, memory_slots=4, memory_context_tokens=2)
    model = TitansMemoryAsContextClassifier(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tok = DNATokenizer(max_length=12)
    input_ids, mask = tok.batch_encode(["ACGTACGTACGT", "TTTTCCCCAAAA", "GGGGAAAATTTT", "ACACACACACAC"])
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    out = model(input_ids=input_ids, attention_mask=mask, labels=labels)
    out["loss"].backward()
    opt.step()
    opt.zero_grad()

    assert torch.isfinite(out["loss"]).item()