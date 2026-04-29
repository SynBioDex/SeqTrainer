"""Backward-compatible preprocessing wrappers.

Deprecated: import from ``seqtrainer.transforms.dna`` instead.
"""

from seqtrainer.transforms.dna import gc_content, kmer_counts, one_hot_encode, pad_or_trim


def pad_sequence(seq: str, max_length: int) -> str:
    return pad_or_trim(seq, max_length)


def calc_gc(df, seq_col_name: str):
    import pandas as pd

    return pd.DataFrame([gc_content(seq) for seq in df[seq_col_name]], columns=["gc_content"])


def generate_kmer_counts(df, seq_col_name: str, k: int, normalize: bool = True):
    import pandas as pd

    return pd.DataFrame([kmer_counts(seq, k=k, normalize=normalize) for seq in df[seq_col_name]])


def process_seqs(df, seq_length: int, seq_col_name: str, pad_seq: bool = True):
    seqs = [pad_or_trim(s, seq_length) for s in df[seq_col_name]] if pad_seq else list(df[seq_col_name])
    return one_hot_encode(seqs)
