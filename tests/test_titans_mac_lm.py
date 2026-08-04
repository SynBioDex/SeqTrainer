import pytest

torch = pytest.importorskip("torch")

from seqtrainer.torch.titans_mac import (  # noqa: E402
    DNABaseTokenizer,
    TitansMACForCausalLM,
    TitansMACLMConfig,
    count_parameters,
    generate_dna,
    load_training_checkpoint,
    save_training_checkpoint,
)


def tiny_config() -> TitansMACLMConfig:
    return TitansMACLMConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        max_length=16,
        memory_slots=6,
        memory_depth=2,
        memory_context_tokens=2,
        persistent_tokens=2,
        dropout=0.0,
    )


def test_tokenizer_contract_and_round_trip(tmp_path) -> None:
    tokenizer = DNABaseTokenizer(max_length=8)
    assert tokenizer.encode("NACGTX") == [1, 2, 3, 4, 5, 1]
    assert tokenizer.decode([0, 1, 2, 3, 4, 5]) == "NACGT"
    input_ids, mask = tokenizer.batch_encode(["ACGT", "N"], padding=True)
    assert input_ids.tolist() == [[2, 3, 4, 5, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]
    assert mask.sum(dim=1).tolist() == [4, 1]
    restored = DNABaseTokenizer.from_file(tokenizer.save(tmp_path / "tokenizer.json"))
    assert restored.to_dict() == tokenizer.to_dict()


def test_model_forward_loss_embeddings_and_memory_diagnostics() -> None:
    config = tiny_config()
    model = TitansMACForCausalLM(config)
    input_ids = torch.randint(1, 6, (3, 16))
    labels = torch.randint(1, 6, (3, 16))
    output = model(
        input_ids,
        labels=labels,
        output_hidden_states=True,
        output_memory_context=True,
        output_memory_diagnostics=True,
        update_memory=False,
    )
    assert output["loss"].ndim == 0
    assert output["logits"].shape == (3, 16, 6)
    assert output["hidden_states"].shape == (3, 16, config.d_model)
    assert output["memory_context"].shape == (3, config.memory_context_tokens, config.d_model)
    embeddings = model.extract_sequence_embeddings(input_ids)
    diagnostics = model.get_memory_diagnostics(input_ids)
    assert embeddings.shape == (3, config.d_model)
    assert diagnostics["slot_cosine_similarity"].shape == (config.memory_slots, config.memory_slots)
    assert count_parameters(model) > 0


def test_generation_smoke() -> None:
    model = TitansMACForCausalLM(tiny_config())
    generated = generate_dna(model, prefix="AC", max_new_tokens=4, top_k=2)
    assert generated.sequence.startswith("AC")
    assert len(generated.token_ids) == 6
    assert set(generated.sequence).issubset(set("NACGT"))


def test_future_bases_do_not_change_earlier_logits() -> None:
    model = TitansMACForCausalLM(tiny_config()).eval()
    first = torch.tensor([[2, 3, 4, 5, 2, 3, 4, 5]])
    second = first.clone()
    second[:, 5:] = torch.tensor([[5, 5, 5]])
    with torch.no_grad():
        first_logits = model(first, update_memory=False)["logits"]
        second_logits = model(second, update_memory=False)["logits"]
    assert torch.allclose(first_logits[:, :5], second_logits[:, :5], atol=1e-6)


def test_checkpoint_round_trip(tmp_path) -> None:
    config = tiny_config()
    model = TitansMACForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(1, 6, (2, 8))
    output = model(input_ids, labels=input_ids)
    output["loss"].backward()
    optimizer.step()
    path = save_training_checkpoint(
        tmp_path / "latest.pt", model, optimizer, epoch=2, step=9, history=[{"loss": 1.0}]
    )
    restored = TitansMACForCausalLM(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    checkpoint = load_training_checkpoint(path, restored, restored_optimizer, trusted=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["step"] == 9
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
