"""Paper-traceable functional neural-memory primitives for Titans MAC.

This package is intentionally separate from :mod:`seqtrainer.torch.titans_mac`.
The latter is a slot-retrieval/EMA baseline; these classes implement only the
functional long-term-memory update needed for the Stage A paper reference.
"""

from .memory import AdaptiveUpdateGates, FunctionalNeuralMemory, GateValues
from .state import PaperMACStreamState

__all__ = [
    "AdaptiveUpdateGates",
    "FunctionalNeuralMemory",
    "GateValues",
    "PaperMACStreamState",
]
