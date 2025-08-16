import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio import SeqIO
import sys

pwm_minus10 = {
    'A': [0.0097, 1.0000, 0.3363, 0.5335, 0.4963, 0.0781],
    'C': [0.0618, 0.0000, 0.1190, 0.1094, 0.2299, 0.0268],
    'G': [0.1042, 0.0000, 0.0856, 0.1317, 0.1399, 0.0000],
    'T': [0.8244, 0.0000, 0.4591, 0.2254, 0.1339, 0.8951]
}

pwm_minus35 = {
    'A': [0.0000, 0.0784, 0.0362, 0.4894, 0.3605, 0.4208],
    'C': [0.1109, 0.0656, 0.0747, 0.2851, 0.3605, 0.0769],
    'G': [0.1267, 0.0181, 0.6192, 0.1041, 0.0000, 0.2225],
    'T': [0.7624, 0.8379, 0.2700, 0.1214, 0.2790, 0.2798]
}

NUCLEOTIDES = ['A', 'C', 'G', 'T']

def pwm_score(seq, pwm):
    score = 0
    for i, base in enumerate(seq):
        if base in pwm:
            score += pwm[base][i]
        else:
            return -np.inf  # Invalid base (e.g., N)
    return score

def find_best_pwm_match(seq, pwm):
    seq = seq.upper()
    best_score = -np.inf
    best_pos = -1
    best_kmer = 'NNNNNN'
    for i in range(len(seq) - 6 + 1):
        kmer = seq[i:i+6]
        score = pwm_score(kmer, pwm)
        if score > best_score:
            best_score = score
            best_pos = i + 1 # One-based
            best_kmer = kmer
    return best_score, best_pos, best_kmer

def paired_pwm_score_exact(seq, pwm_10, pwm_35, spacer_len):
    seq = seq.upper()
    n = len(seq)

    # Score every position for -10 and -35
    scores_10 = [pwm_score(seq[i:i+6], pwm_10) if i + 6 <= n else -np.inf for i in range(n)]
    scores_35 = [pwm_score(seq[i:i+6], pwm_35) if i + 6 <= n else -np.inf for i in range(n)]

    paired_scores = []
    for i in range(n - (6 + spacer_len + 6) + 1):
        score_35 = scores_35[i]
        score_10 = scores_10[i + 6 + spacer_len]
        if -np.inf not in (score_10, score_35):
            paired_scores.append(score_10 + score_35)
    print("Done")

    return max(paired_scores) if paired_scores else -np.inf

def load_sequences(filename):
    df = pd.read_csv(filename, sep="\t", header=None, names=["variant", "expn_med"])
    return df

# def process_data(train_file, test_file, output_file):
train_file = r"C:\Users\Sai\Documents\GitHub\SBOLtrainer\tss_expression_model_format_train_genome_split.txt"
test_file = r"C:\Users\Sai\Documents\GitHub\SBOLtrainer\tss_expression_model_format_test_genome_split.txt"
train_df = load_sequences(train_file)
test_df = load_sequences(test_file)

train_df["dataset"] = "train"
test_df["dataset"] = "test"

data = pd.concat([train_df, test_df], ignore_index=True)

data["minus10_max_score"] = 0.0
data["minus10_start"] = -1
data["minus10"] = "NNNNNN"

data["minus35_max_score"] = 0.0
data["minus35_start"] = -1
data["minus35"] = "NNNNNN"

for idx, row in data.iterrows():
    seq = row["variant"][-150:]  # last 150 bp

    # -10
    s10, p10, k10 = find_best_pwm_match(seq, pwm_minus10)
    data.loc[idx, "minus10_max_score"] = s10
    data.loc[idx, "minus10_start"] = p10
    data.loc[idx, "minus10"] = k10

    # -35
    s35, p35, k35 = find_best_pwm_match(seq, pwm_minus35)
    data.loc[idx, "minus35_max_score"] = s35
    data.loc[idx, "minus35_start"] = p35
    data.loc[idx, "minus35"] = k35
# # Save output
# keep_cols = [c for c in data.columns if not c.startswith("pwm_paired_max_1")]  # keep final max only
data.to_csv("pwmoutput.csv", index=False)

