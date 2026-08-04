# Stage A minimal causal MAC block

`PaperMACBlock` is the Stage A paper-reference integration point.  It is
separate from the legacy `seqtrainer.torch.titans_mac` slot/EMA baseline.

For a fixed 32-position segment `S_t`, the block first calls
`FunctionalNeuralMemory.read_segment`. This produces all 32 retrievals
`H_t = M_(t-1)(Q_t)` from the same pre-write memory snapshot. The core then
causally integrates `[P, H_t, S_t]`; only its resulting 32 representations are
passed to `update_segment`. The returned `M_t` is therefore the single
post-core committed state required by MAC equations 24–25.

The attention input layout is:

```text
[ P_1 ... P_n | H_1 ... H_32 | S_1 ... S_32 ]
```

`P` is a learned persistent-token bank.  The explicit boolean attention mask
uses PyTorch's convention: `True` blocks an edge.

- A persistent query sees persistent-token keys only.
- Either `H_i` or `S_i` sees every persistent token plus `H_1:i` and `S_1:i`.
- It cannot see `H_j` or `S_j` when `j > i`.

The public `block_causal_attention_mask` makes this contract inspectable; the
`integrate` primitive permits perturbation tests independently of memory
updates.  The full `PaperMACBlock` path additionally tests that changing a
future input token cannot affect an earlier output through either retrieval or
attention.  This scope deliberately excludes convolution, scan updates,
flash/fused attention, and throughput optimization.
