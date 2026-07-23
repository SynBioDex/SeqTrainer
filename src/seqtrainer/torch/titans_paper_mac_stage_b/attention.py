"""Mask-preserving SDPA adapters for the Stage B paper-MAC block."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import ContextManager

import torch
from torch import Tensor
from torch.nn import functional as F

from seqtrainer.torch.titans_paper_mac.mac import PaperMACBlock

from .config import ActivationDType


def sdpa_allowed_attention_mask(block: PaperMACBlock, *, device: torch.device) -> Tensor:
    """Return SDPA's boolean convention: ``True`` means the edge is allowed."""

    return ~block.attention_mask(device=device)


def _activation_torch_dtype(dtype: ActivationDType, original: torch.dtype) -> torch.dtype:
    if dtype is ActivationDType.FP32:
        return original
    if dtype is ActivationDType.BF16:
        return torch.bfloat16
    if dtype is ActivationDType.FP16:
        return torch.float16
    raise ValueError(f"unsupported activation dtype: {dtype}")


def _validate_mixed_precision_boundary(
    block: PaperMACBlock,
    retrieval: Tensor,
    segment: Tensor,
    activation_dtype: ActivationDType,
) -> None:
    if activation_dtype is ActivationDType.FP32:
        return
    if retrieval.dtype is not torch.float32 or segment.dtype is not torch.float32:
        raise ValueError("reduced-precision attention requires FP32 memory/core inputs")
    if any(parameter.dtype is not torch.float32 for parameter in block.parameters()):
        raise ValueError("reduced-precision attention requires an FP32 PaperMACBlock")
    if activation_dtype is ActivationDType.FP16 and segment.device.type != "cuda":
        raise RuntimeError("FP16 SDPA is available only on CUDA in Stage B")
    if (
        activation_dtype is ActivationDType.BF16
        and segment.device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("this CUDA device does not support BF16 attention")


def _forced_flash_context() -> ContextManager[object]:
    return torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=False,
        enable_mem_efficient=False,
    )


def integrate_sdpa_attention(
    block: PaperMACBlock,
    retrieval: Tensor,
    segment_embeddings: Tensor,
    activation_dtype: ActivationDType = ActivationDType.FP32,
    *,
    force_flash: bool = False,
) -> Tensor:
    """Reproduce ``PaperMACBlock.integrate`` with functional SDPA.

    The authoritative MultiheadAttention boolean mask is converted first to
    SDPA's opposite boolean convention and then to the additive mask used by
    the reference MHA implementation. Reduced precision is confined to this
    attention/norm computation; the returned sequence is restored to the FP32
    memory-island dtype before the neural-memory write.
    """

    block._validate_layout_inputs(retrieval, segment_embeddings)
    activation_dtype = ActivationDType(activation_dtype)
    _validate_mixed_precision_boundary(
        block,
        retrieval,
        segment_embeddings,
        activation_dtype,
    )
    original_dtype = segment_embeddings.dtype
    compute_dtype = _activation_torch_dtype(activation_dtype, original_dtype)
    persistent = block.persistent_tokens.to(
        device=segment_embeddings.device,
        dtype=compute_dtype,
    )
    retrieval_activation = retrieval.to(dtype=compute_dtype)
    segment_activation = segment_embeddings.to(dtype=compute_dtype)
    layout = torch.cat(
        (persistent, retrieval_activation, segment_activation), dim=0
    ).unsqueeze(0)

    attention = block.attention
    packed = F.linear(
        layout,
        attention.in_proj_weight.to(dtype=compute_dtype),
        None
        if attention.in_proj_bias is None
        else attention.in_proj_bias.to(dtype=compute_dtype),
    )
    query, key, value = packed.chunk(3, dim=-1)
    batch, length, embed_dim = query.shape
    heads = attention.num_heads
    head_dim = embed_dim // heads

    def split_heads(value_to_split: Tensor) -> Tensor:
        return value_to_split.view(batch, length, heads, head_dim).transpose(1, 2)

    query = split_heads(query)
    key = split_heads(key)
    value = split_heads(value)
    allowed = sdpa_allowed_attention_mask(block, device=layout.device)
    additive_mask = torch.zeros_like(allowed, dtype=compute_dtype).masked_fill(
        ~allowed, float("-inf")
    )
    if layout.device.type == "cpu" and not force_flash:
        # Current Colab builds can select CPU Flash SDPA even though its
        # backward is unimplemented. The basal gate is intentionally small,
        # so use the exact scaled dot-product definition directly instead.
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)
        weights = torch.softmax(scores + additive_mask.unsqueeze(0).unsqueeze(0), dim=-1)
        attended = torch.matmul(weights, value)
    else:
        context = _forced_flash_context() if force_flash else nullcontext()
        with context:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=additive_mask.unsqueeze(0).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
            )
    merged = attended.transpose(1, 2).contiguous().view(batch, length, embed_dim)
    projected = F.linear(
        merged,
        attention.out_proj.weight.to(dtype=compute_dtype),
        None
        if attention.out_proj.bias is None
        else attention.out_proj.bias.to(dtype=compute_dtype),
    )
    sequence_start = block.persistent_token_count + block.segment_length
    residual = segment_activation + projected[0, sequence_start:]
    normalized = F.layer_norm(
        residual,
        block.output_norm.normalized_shape,
        block.output_norm.weight.to(dtype=compute_dtype),
        block.output_norm.bias.to(dtype=compute_dtype),
        block.output_norm.eps,
    )
    return normalized.to(dtype=original_dtype)


def integrate_flash_attention(
    block: PaperMACBlock,
    retrieval: Tensor,
    segment_embeddings: Tensor,
    activation_dtype: ActivationDType = ActivationDType.FP32,
) -> Tensor:
    """Force the CUDA Flash SDP kernel; fail instead of weakening the mask."""

    return integrate_sdpa_attention(
        block,
        retrieval,
        segment_embeddings,
        activation_dtype,
        force_flash=True,
    )


def probe_flash_mask_support(
    block: PaperMACBlock,
    *,
    activation_dtype: ActivationDType = ActivationDType.FP16,
) -> dict[str, object]:
    """Test the exact block mask with math and memory-efficient fallbacks off."""

    try:
        parameter = next(block.parameters())
    except StopIteration:
        return {"available": False, "reason": "block has no parameters"}
    if parameter.device.type != "cuda" or not torch.cuda.is_available():
        return {
            "available": False,
            "reason": "unavailable: no CUDA device attached",
            "device": str(parameter.device),
        }
    properties = torch.cuda.get_device_properties(parameter.device)
    if "A100" not in properties.name:
        return {
            "available": False,
            "reason": "Stage B Flash claims require an A100 probe",
            "device": properties.name,
        }
    try:
        candidate = copy.deepcopy(block).float()
        retrieval = torch.randn(
            32, candidate.d_model, device=parameter.device, dtype=torch.float32
        )
        segment = torch.randn_like(retrieval)
        with torch.no_grad():
            output = integrate_flash_attention(
                candidate,
                retrieval,
                segment,
                activation_dtype,
            )
        torch.cuda.synchronize(parameter.device)
        return {
            "available": True,
            "reason": "forced Flash SDP accepted the exact additive [P,H,S] mask",
            "device": properties.name,
            "activation_dtype": activation_dtype.value,
            "output_shape": list(output.shape),
        }
    except (RuntimeError, ValueError) as error:
        return {
            "available": False,
            "reason": f"forced Flash SDP rejected the exact mask: {error}",
            "device": properties.name,
            "activation_dtype": activation_dtype.value,
        }
