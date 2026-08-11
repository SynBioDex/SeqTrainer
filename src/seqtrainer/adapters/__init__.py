"""External model adapters for benchmark workflows."""

from .ipromp import write_ipromp_fastas
from .ipromp_inference import read_seqtrainer_fasta

__all__ = ["read_seqtrainer_fasta", "write_ipromp_fastas"]
