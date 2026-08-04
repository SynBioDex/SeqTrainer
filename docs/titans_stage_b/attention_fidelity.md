# B6 mask-preserving attention backends

## Authoritative edge set

Stage B does not redesign MAC attention. The Stage A
`block_causal_attention_mask` remains authoritative for the complete
`[persistent, retrieval, sequence]` layout:

- persistent queries see persistent keys only;
- retrieval and sequence query `i` see every persistent key;
- those queries see retrieval and sequence positions only through `i`; and
- every future retrieval and future sequence edge is blocked.

PyTorch uses opposite boolean conventions in the two relevant APIs:
`MultiheadAttention` uses `True = blocked`, while functional SDPA uses
`True = allowed`. `sdpa_allowed_attention_mask` is therefore the exact boolean
complement of the Stage A mask. Before execution, the adapter converts that edge
set into the same additive `0/-inf` representation used by the reference MHA
path. Tests enumerate every edge and require zero complement mismatches.

## Functional SDPA adapter

`integrate_sdpa_attention` uses the existing `PaperMACBlock` parameters rather
than introducing a second attention module:

1. concatenate `[P,H,S]` exactly as Stage A does;
2. apply the packed Q/K/V projection from `nn.MultiheadAttention`;
3. split the same number of heads;
4. call `scaled_dot_product_attention` with the exact additive mask;
5. apply the existing output projection, sequence residual, and LayerNorm; and
6. return only the 32 sequence positions.

This path is opt-in through `AttentionBackend.SDPA`; MultiheadAttention remains
the default. Neural-memory read-before-write timing is unchanged, and the
attention backend never sees or mutates stream state directly.

## Precision boundary

FP64 CPU is the oracle and FP32 is numerical parity. `ActivationDType.BF16` and
`FP16` are classified separately as behavioral mixed-precision modes:

- input, module parameters, fast weights, surprise, and published memory state
  must be FP32;
- only Q/K/V attention, output projection, residual, and norm activations are
  cast down;
- the 32-position sequence is cast back to FP32 before the memory write; and
- CPU FP16 is rejected. BF16 runs only if the installed PyTorch/device supports
  the required operations.

Thus reduced precision cannot silently turn the neural-memory recurrence or
its persistent state into BF16/FP16.

## Flash/A100 policy

`probe_flash_mask_support` disables math and memory-efficient SDP fallbacks and
forces the Flash kernel with the exact non-triangular `[P,H,S]` additive mask.
It reports support only when attached to an NVIDIA A100 and that forced call
succeeds. It never substitutes a simpler causal mask. On the current Intel
MacBook, CUDA and an A100 are unavailable, so Flash remains unregistered and
the artifact records an unavailable result rather than a performance claim.

## Evidence

`artifacts/titans_stage_b/b6_attention_backend_evidence.json` records:

- FP64 and FP32 sequence, retrieval, fast-weight, surprise, input-gradient,
  persistent-token-gradient, and attention-parameter-gradient errors;
- exhaustive edge counts and persistent-query restriction;
- future-token prefix error;
- BF16/FP16 behavioral results and published memory-state dtypes;
- raw reference/SDPA CPU timings with geometry, warmups, repetitions, seed, and
  hardware/software provenance; and
- the exact-mask Flash/A100 capability probe.

The Markdown companion is a concise summary. These local CPU results do not
satisfy the Stage B A100 exit criterion; B8 must retain that as an explicit
blocker unless an A100 artifact is later produced.

