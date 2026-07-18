# Stage B named-A100 pilot

This pilot is the target-hardware evidence package required by the Stage C gate. It does not train on the 15 Gbp corpus. It measures the synthetic paper-MAC mechanisms and repeated two-segment outer-training steps on a named NVIDIA A100 without replacing the canonical CPU artifacts.

## What the command proves

The runner refuses non-A100 devices before benchmarking, then writes an isolated bundle under `artifacts/titans_stage_b/a100/`. Its strict verifier independently reloads the JSON and requires:

- a device name containing `A100`, version/capability provenance, and BF16 support metadata;
- at least three matched B3 repetitions with tensor-exact reference and exact-functional outputs;
- FP64/FP32 attention parity, the authoritative `[persistent, retrieval, sequence]` mask, and zero future-prefix leakage;
- BF16 and FP16 behavioral runs whose published sequence and neural-memory state return to FP32;
- a forced Flash-SDPA attempt with the exact additive mask (a documented rejection is valid evidence that Flash is unsafe to enable; it is never silently substituted);
- finite, causal long streams with CUDA peak allocation and adaptive-memory advantage over frozen/no-memory controls;
- at least three complete forward/backward/AdamW steps for reference, exact memory, and exact memory plus SDPA, including a finite nonzero gradient through segment one's written state;
- a default chosen only from exact FP32 candidates and only when it is at least 5% faster than reference.

Every source artifact is SHA-256 recorded. `--verify-only` recomputes the hashes and all semantic checks instead of trusting the manifest's verdict.

## Colab Pro A100 procedure with uv

1. In Colab, select **Runtime → Change runtime type → A100 GPU**. Confirm the allocation with `!nvidia-smi`; a T4, L4, V100, or CPU does not satisfy this pilot.
2. Clone SeqTrainer and check out the exact branch/commit being evaluated.
3. Set up an environment that reuses Colab's CUDA-enabled PyTorch:

   ```bash
   python -m pip install uv
   uv venv --system-site-packages .venv
   uv pip install --python .venv/bin/python --no-deps -e .
   uv pip install --python .venv/bin/python pytest ruff
   .venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(), torch.version.cuda)"
   ```

4. Run CPU-safe contract tests, then the preflight:

   ```bash
   .venv/bin/python -m pytest -q tests/test_titans_paper_mac_stage_b_a100_pilot.py tests/test_titans_paper_mac_stage_b_training_step.py
   .venv/bin/seqtrainer-titans-stage-b-a100-pilot --preflight-only
   ```

5. Capture the evidence. Saving directly to mounted Drive is recommended so a Colab disconnect does not lose completed artifacts:

   ```bash
   .venv/bin/seqtrainer-titans-stage-b-a100-pilot \
     --output-dir /content/drive/MyDrive/seqtrainer-a100 \
     --warmup-runs 1 --repetitions 3
   .venv/bin/seqtrainer-titans-stage-b-a100-pilot \
     --output-dir /content/drive/MyDrive/seqtrainer-a100 --verify-only
   ```

6. Copy the complete verified directory to `artifacts/titans_stage_b/a100/` in the repository. Do not copy only the manifest. Run the full suite and regenerate the audit:

   ```bash
   uv run pytest -q
   uv run seqtrainer-titans-stage-b-audit --test-count <passed> --warning-count <warnings>
   ```

The audit remains `NO_GO` when the directory is absent, incomplete, modified after capture, or fails any semantic check. A successful pilot removes the hardware/performance blockers but does not authorize Stage C unless every other audit criterion also passes.
