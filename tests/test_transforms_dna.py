import numpy as np

from seqtrainer.transforms.dna import gc_content, kmer_counts, normalize_sequence, one_hot_encode, pad_or_trim


def test_normalize_sequence_replaces_unknowns():
    assert normalize_sequence("acgtxyz") == "ACGTNNN"


def test_pad_or_trim():
    assert pad_or_trim("ACGT", 6) == "NACGTN"
    assert pad_or_trim("AACCGGTT", 4) == "CCGG"


def test_one_hot_encode_shape():
    encoded = one_hot_encode(["AC", "GT"])
    assert encoded.shape == (2, 2, 5)
    assert np.isclose(encoded.sum(), 4.0)


def test_gc_content():
    assert gc_content("AGGC") == 0.75


def test_kmer_counts_normalized_sum():
    values = kmer_counts("ACGT", k=2, normalize=True)
    assert np.isclose(sum(values.values()), 1.0)
