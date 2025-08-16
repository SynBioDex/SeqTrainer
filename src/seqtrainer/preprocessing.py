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
    sequences_array = np.array([[char for char in seq] for seq in sequences])
    sequence_length = sequences_array.shape[1]
    ohe = OneHotEncoder(sparse_output=False,
                        categories=[defined_categories] * sequence_length,
                        dtype=np.float32) 

    return ohe.fit_transform(sequences_array)

def pad_sequence(seq, max_length):
	if len(seq) > max_length:
		diff = len(seq) - max_length
		trim_length = int(diff / 2)
		seq = seq[trim_length : -(trim_length + diff%2)]
	else:
		seq = seq.center(max_length, 'N')
	return seq

def process_seqs(df, seq_length, seq_col_name, pad_seq=True):
	padded_seqs = [pad_sequence(x, seq_length) for x in df[seq_col_name]] if pad_seq else df[seq_col_name]
	return one_hot_encode(np.array(padded_seqs))

def calc_gc(df, seq_col_name):
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

# df3 = pd.read_csv(os.path.join(original_data_path, "fLP3_Endo2_lb_expression_formatted.txt"), delimiter=" ")
# df3 = df3[['variant', 'expn_med']]
# # print(process_seqs(df3, 150, "variant")[0]) # .reshape(-1, 5))
# kmer = generate_kmer_counts(df3, 'variant', 3)
# gc = calc_gc(df3, 'variant')
# gc.to_csv("gc.csv", index=False)

