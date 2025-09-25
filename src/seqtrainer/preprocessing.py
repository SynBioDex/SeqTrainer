import sbol2
import os
import pandas as pd
import numpy as np
import itertools
from collections import Counter 
import sys 
from sklearn.preprocessing import OneHotEncoder
import numpy as np

python_executable = sys.executable
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..','data')
sbol_path = os.path.join(current_dir, '..', 'sbol_data')
downloaded_sbol_path = os.path.join(current_dir, '..', 'downloaded_sbol')
original_data_path = os.path.join(data_path, 'original_data')
nt_path = os.path.join(current_dir, '..', 'nt_data')
model_data_path = os.path.join(data_path, 'processed_data', 'replicated_models')

def one_hot_encode(sequences, defined_categories=['A', 'C', 'G', 'T', 'N']):
    """
    One-hot encodes a list of nucleotide sequences.

    Parameters
    ----------
    sequences : list of str
        A list of nucleotide sequences of equal length.
    defined_categories : list of str, optional
        The allowed nucleotide categories for encoding.
        Defaults to ['A', 'C', 'G', 'T', 'N'].

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (n_sequences, sequence_length * n_categories),
        where each nucleotide in the sequences is represented by a one-hot vector.
    """

    sequences_array = np.array([[char for char in seq] for seq in sequences])
    sequence_length = sequences_array.shape[1]
    ohe = OneHotEncoder(sparse_output=False,
                        categories=[defined_categories] * sequence_length,
                        dtype=np.float32) 

    return ohe.fit_transform(sequences_array)

def pad_sequence(seq, max_length):
    """
    Pads or trims a nucleotide sequence to a fixed length.

    If the sequence is longer than `max_length`, it is symmetrically trimmed
    from both ends (extra base removed from the right if odd difference).
    If shorter, it is padded with 'N' characters on both sides until centered.

    Parameters
    ----------
    seq : str
        The nucleotide sequence to adjust.
    max_length : int
        The target sequence length.

    Returns
    -------
    str
        The sequence adjusted to the specified length.
    """
    if len(seq) > max_length:
        diff = len(seq) - max_length
        trim_length = int(diff / 2)
        seq = seq[trim_length : -(trim_length + diff % 2)]
    else:
        seq = seq.center(max_length, 'N')
    return seq
	

def process_seqs(df, seq_length, seq_col_name, pad_seq=True):
    """
    Pads and one-hot encodes sequences from a DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing sequences.
    seq_length : int
        Target sequence length for padding/trimming.
    seq_col_name : str
        Name of the column in the DataFrame containing sequences.
    pad_seq : bool, optional
        Whether to pad/trim sequences to `seq_length` before encoding.
        Defaults to True.

    Returns
    -------
    numpy.ndarray
        One-hot encoded representation of the sequences with shape
        (n_sequences, seq_length * n_categories).
    """
    padded_seqs = [pad_sequence(x, seq_length) for x in df[seq_col_name]] if pad_seq else df[seq_col_name]
    return one_hot_encode(np.array(padded_seqs))

def calc_gc(df, seq_col_name):
    """
    Calculates GC content for sequences in a DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing sequences.
    seq_col_name : str
        Name of the column in the DataFrame containing sequences.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with one column, 'gc_content', containing the
        fraction of G and C bases in each sequence.
    """
    sequences = df[seq_col_name]
    gc_all = []
    for seq in sequences:
        seq = seq.upper()  
        seq_length = len(seq)
        num_gc = seq.count('G') + seq.count('C')
        gc_content = num_gc / seq_length if seq_length > 0 else 0
        gc_all.append(gc_content)
    return pd.DataFrame(gc_all, columns=['gc_content'])

def generate_kmer_counts(df, seq_col_name, k, normalize=True):
    """
    Generates k-mer frequency counts for sequences in a DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing sequences.
    seq_col_name : str
        Name of the column in the DataFrame containing sequences.
    k : int
        Length of the k-mers to compute.
    normalize : bool, optional
        If True, returns relative frequencies (counts divided by total k-mers per sequence).
        If False, returns raw counts. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each column corresponds to a possible k-mer
        (composed of A, C, G, T) and each row corresponds to a sequence,
        containing either normalized frequencies or raw counts.
    """
    sequences = df[seq_col_name]
    all_kmers = [''.join(x) for x in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    
    kmer_counts = []
    for seq in sequences:
        counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
        total = max(len(seq) - k + 1, 1) 
        if normalize:
            kmer_counts.append({kmer: counts.get(kmer, 0) / total for kmer in all_kmers})
        else:
            kmer_counts.append({kmer: counts.get(kmer, 0) for kmer in all_kmers})
    
    kmer_df = pd.DataFrame(kmer_counts, columns=all_kmers)  
    return kmer_df
