"""Training checkpoint persistence with PyTorch 2.6 compatibility."""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Optional

import torch
from torch import nn

from .configuration import TitansMACLMConfig


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    step: int = 0,
    history: Optional[list[dict[str, Any]]] = None,
    **metadata: Any,
) -> Path:
    """Atomically save a complete, resumable training checkpoint."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = getattr(model, "config", None)
    state = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "config": config.to_dict() if isinstance(config, TitansMACLMConfig) else config,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "history": history or [],
        "metadata": metadata,
        "rng_state": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, output)
    return output


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str | torch.device] = "cpu",
    strict: bool = True,
    trusted: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint; trusted full training states use ``weights_only=False``."""

    kwargs: dict[str, Any] = {"map_location": map_location}
    if trusted:
        kwargs["weights_only"] = False
    try:
        checkpoint = torch.load(Path(path), **kwargs)
    except TypeError:  # PyTorch before weights_only was added
        kwargs.pop("weights_only", None)
        checkpoint = torch.load(Path(path), **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    rng_state = checkpoint.get("rng_state")
    if restore_rng and rng_state:
        random.setstate(rng_state["python"])
        torch.set_rng_state(rng_state["torch"].cpu())
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return checkpoint
