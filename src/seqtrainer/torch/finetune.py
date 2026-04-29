"""Fine-tuning helper placeholders for PyTorch workflows."""


def build_finetune_config(*, backbone: str, head: str, learning_rate: float = 1e-4, epochs: int = 5) -> dict:
    """Return a minimal fine-tuning configuration dictionary."""
    return {
        "framework": "torch",
        "backbone": backbone,
        "head": head,
        "learning_rate": learning_rate,
        "epochs": epochs,
    }
