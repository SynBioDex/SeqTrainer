# Titans paper-MAC Stage B backend benchmark

- Memory backend: `reference`
- Attention backend: `multihead_attention`
- Activation dtype: `float32`
- Seed: `20260727`
- Device: `CPU x86_64` (`cpu`)
- Torch: `2.2.2`
- Geometry: ModelGeometry(block_count=1, d_model=8, num_heads=2, persistent_tokens=4, memory_depth=1, segment_length=32, parameter_count=627)
- Segments/tokens: 1/32
- Warmups/repetitions: 1/3
- Median wall time: 0.097953565 s
- Throughput: 326.685 tokens/s
- State payload: 576 bytes
- CUDA memory: unavailable (non-CUDA execution)
- Parity passed: True

Raw per-repetition timings and complete parity metrics are in the JSON artifact.
