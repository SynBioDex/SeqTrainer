from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from seqtrainer.data.bacteria_titan import build_stream_segments  # noqa: E402
from seqtrainer.torch.titans_paper_mac_stage_b import (  # noqa: E402
    AttentionBackend,
    StageBBackendConfig,
)
from seqtrainer.torch.titans_paper_mac_stage_c import (  # noqa: E402
    MemoryMode,
    SeqTrainerBaseTokenizer,
    StageCModelConfig,
    StageCPaperMACForCausalLM,
    StageCTrainer,
    StreamBatchScheduler,
    compute_stage_c_metrics,
    detach_stream_states,
    run_statistical_baselines,
    load_stage_c_checkpoint,
    save_stage_c_checkpoint,
)
from seqtrainer.torch.titans_paper_mac_stage_c.trainer import (  # noqa: E402
    NonFiniteTrainingError,
)


def tiny_config(*, horizon: int = 2) -> StageCModelConfig:
    tokenizer = SeqTrainerBaseTokenizer()
    return StageCModelConfig(
        vocab_size=tokenizer.spec.vocab_size,
        pad_token_id=tokenizer.spec.pad_token_id,
        tokenizer_name=tokenizer.spec.name,
        tokenizer_checksum=tokenizer.spec.checksum,
        block_count=1,
        d_model=4,
        num_heads=2,
        persistent_tokens=2,
        memory_depth=1,
        gradient_horizon=horizon,
        backend=StageBBackendConfig(),
    )


def tensors(sequence: str = "ACGT" * 8):
    tokenizer = SeqTrainerBaseTokenizer()
    encoded = tokenizer.encode(sequence + "A")
    inputs = torch.tensor([encoded.ids[:32]])
    labels = torch.tensor([encoded.ids[1:33]])
    mask = torch.ones((1, 32), dtype=torch.bool)
    bases = torch.ones((1, 32), dtype=torch.long)
    return inputs, labels, mask, bases


def test_stage_c_lm_has_tied_head_finite_meta_gradients_and_bpb() -> None:
    torch.manual_seed(41)
    model = StageCPaperMACForCausalLM(tiny_config())
    inputs, labels, mask, bases = tensors()
    output = model.forward_segment(
        (model.initial_states("stream"),),
        inputs,
        labels=labels,
        valid_mask=mask,
        loss_mask=mask,
        represented_base_counts=bases,
    )

    assert model.lm_head.weight is model.token_embeddings.weight
    assert output.logits.shape == (1, 32, 6)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert output.memory_update_norm > 0
    assert output.surprise_norm > 0
    assert output.state_drift_norm > 0
    assert set(output.gate_statistics) == {
        "alpha_mean",
        "alpha_std",
        "alpha_min",
        "alpha_max",
        "eta_mean",
        "eta_std",
        "eta_min",
        "eta_max",
        "theta_mean",
        "theta_std",
        "theta_min",
        "theta_max",
    }
    assert all(0 <= value <= 1 for value in output.gate_statistics.values())
    # The write happens after this segment's logits, so outer gradients through
    # gates are intentionally observed when the next segment reads the state.
    second = model.forward_segment(
        output.states,
        inputs,
        labels=labels,
        valid_mask=mask,
        loss_mask=mask,
        represented_base_counts=bases,
    )
    assert second.loss is not None
    (output.loss + second.loss).backward()
    assert model.stack.blocks[0].memory.gates.projection.weight.grad is not None
    assert torch.isfinite(model.stack.blocks[0].memory.gates.projection.weight.grad).all()
    metrics = compute_stage_c_metrics(output.logits.detach(), labels, mask, bases)
    assert metrics.valid_tokens == metrics.valid_bases == 32
    assert metrics.bits_per_base > 0


def test_cpu_basal_routes_backward_through_math_sdpa() -> None:
    """Colab CPU pilots must not reach MultiheadAttention's Flash backward."""

    tokenizer = SeqTrainerBaseTokenizer()
    config = StageCModelConfig.cpu_basal(
        vocab_size=tokenizer.spec.vocab_size,
        pad_token_id=tokenizer.spec.pad_token_id,
        tokenizer_name=tokenizer.spec.name,
        tokenizer_checksum=tokenizer.spec.checksum,
    )
    assert config.backend.attention_backend is AttentionBackend.SDPA

    model = StageCPaperMACForCausalLM(config)
    inputs, labels, mask, bases = tensors()
    output = model.forward_segment(
        (model.initial_states("cpu"),),
        inputs,
        labels=labels,
        valid_mask=mask,
        loss_mask=mask,
        represented_base_counts=bases,
    )

    assert output.loss is not None
    output.loss.backward()
    gradient = model.stack.blocks[0].attention.in_proj_weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


def test_cpu_basal_inner_memory_updates_remain_finite_across_training_steps() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    config = StageCModelConfig.cpu_basal(
        vocab_size=tokenizer.spec.vocab_size,
        pad_token_id=tokenizer.spec.pad_token_id,
        tokenizer_name=tokenizer.spec.name,
        tokenizer_checksum=tokenizer.spec.checksum,
    )
    segments = build_stream_segments(
        sequence="ACGTN" * 80,
        accession="cpu-basal",
        contig_id="contig",
        split="train",
        clade_group="cpu-basal",
        tokenizer=tokenizer,
    )
    model = StageCPaperMACForCausalLM(config)
    trainer = StageCTrainer(model, torch.optim.AdamW(model.parameters(), lr=3e-4))
    history = trainer.train(
        StreamBatchScheduler({segments[0].stream_id: segments}, batch_size=1, shuffle=False),
        max_optimizer_steps=4,
    )

    assert len(history) == 4
    assert all(torch.isfinite(torch.tensor(record.loss_per_token)) for record in history)
    assert all(torch.isfinite(torch.tensor(record.gradient_norm)) for record in history)


def test_trainer_reports_the_first_nonfinite_forward_value() -> None:
    model = StageCPaperMACForCausalLM(tiny_config())
    with torch.no_grad():
        model.token_embeddings.weight.fill_(float("nan"))
    tokenizer = SeqTrainerBaseTokenizer()
    segments = build_stream_segments(
        sequence="ACGT" * 16,
        accession="nonfinite",
        contig_id="contig",
        split="train",
        clade_group="nonfinite",
        tokenizer=tokenizer,
    )
    trainer = StageCTrainer(model, torch.optim.AdamW(model.parameters(), lr=1e-3))

    with pytest.raises(NonFiniteTrainingError, match="non-finite forward output.*loss_sum"):
        trainer.train(
            StreamBatchScheduler({segments[0].stream_id: segments}, batch_size=1, shuffle=False),
            max_optimizer_steps=1,
        )


def test_future_tokens_do_not_change_earlier_logits_or_valid_memory_state() -> None:
    torch.manual_seed(43)
    model = StageCPaperMACForCausalLM(tiny_config()).double()
    inputs, _, mask, _ = tensors()
    inputs = inputs.long()
    changed = inputs.clone()
    changed[:, 20:] = torch.tensor([5, 4, 3, 2] * 3)[:12]
    initial = model.initial_states("stream")

    first = model.forward_segment((initial,), inputs, valid_mask=mask)
    second = model.forward_segment((initial,), changed, valid_mask=mask)

    assert torch.equal(first.logits[:, :20], second.logits[:, :20])

    tail_mask = mask.clone()
    tail_mask[:, 20:] = False
    first_tail = model.forward_segment((initial,), inputs, valid_mask=tail_mask)
    second_tail = model.forward_segment((initial,), changed, valid_mask=tail_mask)
    for first_state, second_state in zip(first_tail.states[0], second_tail.states[0]):
        for name in first_state.fast_weights:
            assert torch.equal(first_state.fast_weights[name], second_state.fast_weights[name])


def test_batched_streams_equal_isolated_and_memory_controls_are_explicit() -> None:
    torch.manual_seed(47)
    batched = StageCPaperMACForCausalLM(tiny_config())
    isolated = copy.deepcopy(batched)
    inputs, _, mask, _ = tensors()
    second_inputs = inputs.roll(1, dims=1)
    batch = batched.forward_segment(
        (batched.initial_states("a"), batched.initial_states("b")),
        torch.cat((inputs, second_inputs)),
        valid_mask=torch.cat((mask, mask)),
    )
    alone_a = isolated.forward_segment((isolated.initial_states("a"),), inputs, valid_mask=mask)
    alone_b = isolated.forward_segment((isolated.initial_states("b"),), second_inputs, valid_mask=mask)

    assert torch.equal(batch.logits[0], alone_a.logits[0])
    assert torch.equal(batch.logits[1], alone_b.logits[0])
    frozen = isolated.forward_segment(
        (isolated.initial_states("frozen"),), inputs, valid_mask=mask, memory_mode=MemoryMode.FROZEN
    )
    none = isolated.forward_segment(
        (isolated.initial_states("none"),), inputs, valid_mask=mask, memory_mode=MemoryMode.NONE
    )
    assert frozen.memory_update_norm == 0
    assert frozen.surprise_norm == frozen.state_drift_norm == 0
    assert none.retrieval_norm == 0


def test_detach_preserves_values_and_truncates_history() -> None:
    model = StageCPaperMACForCausalLM(tiny_config())
    inputs, _, mask, _ = tensors()
    output = model.forward_segment((model.initial_states("stream"),), inputs, valid_mask=mask)
    detached = detach_stream_states(output.states[0])

    for original, replacement in zip(output.states[0], detached):
        for name in original.fast_weights:
            assert torch.equal(original.fast_weights[name], replacement.fast_weights[name])
            assert replacement.fast_weights[name].grad_fn is None
            assert replacement.fast_weights[name].requires_grad


def test_scheduler_orders_streams_and_trainer_honors_horizon() -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    streams = {}
    for accession, sequence in (("a", "ACGT" * 20), ("b", "TGCA" * 20)):
        segments = build_stream_segments(
            sequence=sequence,
            accession=accession,
            contig_id="contig",
            split="train",
            clade_group=f"group-{accession}",
            tokenizer=tokenizer,
        )
        streams[segments[0].stream_id] = segments
    model = StageCPaperMACForCausalLM(tiny_config(horizon=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StreamBatchScheduler(streams, batch_size=2, shuffle=False)
    trainer = StageCTrainer(model, optimizer)

    history = trainer.train(scheduler, max_optimizer_steps=1)

    assert len(history) == 1
    assert history[0].segment_steps == 2
    assert history[0].segments == 4
    assert trainer.optimizer_step == 1
    assert trainer.processed_segments == 4
    assert history[0].valid_bases == 128
    assert history[0].gradient_norm > 0
    assert history[0].written_state_gradient_norm > 0
    assert history[0].surprise_norm > 0
    assert history[0].state_drift_norm > 0
    assert 0 < history[0].alpha_mean < 1


def test_cpu_statistical_baselines_are_informative() -> None:
    results = run_statistical_baselines(
        ["ACGT" * 100],
        ["ACGT" * 20],
    )
    by_name = {result.name: result for result in results}

    assert set(by_name) == {
        "uniform",
        "nucleotide_frequency",
        "markov_order_1",
        "markov_order_3",
        "markov_order_5",
    }
    assert by_name["markov_order_1"].bits_per_base < by_name["uniform"].bits_per_base


def test_checkpoint_restores_optimizer_cursor_rng_and_functional_states(tmp_path) -> None:
    tokenizer = SeqTrainerBaseTokenizer()
    segments = build_stream_segments(
        sequence="ACGT" * 30,
        accession="resume",
        contig_id="contig",
        split="train",
        clade_group="group-resume",
        tokenizer=tokenizer,
    )
    streams = {segments[0].stream_id: segments}
    model = StageCPaperMACForCausalLM(tiny_config(horizon=1))
    uninterrupted_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = StreamBatchScheduler(streams, batch_size=1, shuffle=False)
    trainer = StageCTrainer(model, optimizer)
    trainer.train(scheduler, max_optimizer_steps=1)
    checkpoint = save_stage_c_checkpoint(
        tmp_path / "latest.pt",
        trainer,
        scheduler,
        dataset_fingerprint="fixture-data",
        code_commit="fixture-code",
    )

    restored_model = StageCPaperMACForCausalLM(tiny_config(horizon=1))
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = StreamBatchScheduler(streams, batch_size=1, shuffle=False)
    restored = StageCTrainer(restored_model, restored_optimizer)
    load_stage_c_checkpoint(
        checkpoint,
        restored,
        restored_scheduler,
        dataset_fingerprint="fixture-data",
        trusted=True,
    )

    assert restored.optimizer_step == trainer.optimizer_step == 1
    assert restored_scheduler.to_state_dict() == scheduler.to_state_dict()
    assert restored.stream_states.keys() == trainer.stream_states.keys()
    for expected, actual in zip(
        trainer.stream_states["resume:contig"], restored.stream_states["resume:contig"]
    ):
        for name in expected.fast_weights:
            assert torch.equal(expected.fast_weights[name], actual.fast_weights[name])
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(model.state_dict().values(), restored_model.state_dict().values())
    )

    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_model.parameters(), lr=1e-3)
    uninterrupted_scheduler = StreamBatchScheduler(streams, batch_size=1, shuffle=False)
    uninterrupted = StageCTrainer(uninterrupted_model, uninterrupted_optimizer)
    uninterrupted.train(uninterrupted_scheduler, max_optimizer_steps=2)
    restored.train(restored_scheduler, max_optimizer_steps=2)

    assert restored.optimizer_step == uninterrupted.optimizer_step == 2
    assert restored.processed_bases == uninterrupted.processed_bases
    assert restored_scheduler.to_state_dict() == uninterrupted_scheduler.to_state_dict()
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            uninterrupted_model.state_dict().values(), restored_model.state_dict().values()
        )
    )
