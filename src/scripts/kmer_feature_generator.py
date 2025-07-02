from __future__ import division
import timeit
from sklearn.metrics import r2_score
import argparse
from collections import Counter
import pandas as pd
import numpy as np


def read_file(dataset_filename):
    """
    Load sequences and expression values for training data
    """
    sequences = []
    expression = []
    with open(dataset_filename, 'r') as input_f:
        for line in input_f:
            data = (line.strip('\n')).split('\t')
            sequence, exp_value = data[:]
            sequences.append(sequence)
            exp = float(exp_value)
            expression.append(exp)
    return sequences, expression


def generate_kmer_counts(sequences, k):
    
    # all_kmers = [''.join(x) for x in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    
    start = timeit.default_timer()
    print("Counting k-mers of length", k, "in all sequences...")
    kmer_counts = [dict(Counter([seq[i:i+k] for i in range(len(seq)-k+1)])) for seq in sequences]
    elapsed = timeit.default_timer() - start
    print("Elapsed time:", elapsed)
    # ensures all columns are present, even if some keys not present in all dictionaries
    print("Creating features...")
    kmer_df = pd.DataFrame.from_records(kmer_counts)
    kmer_df.fillna(0, inplace=True)
    
    return kmer_df


def kmer_filter_random(df, df_random, y_train):

    filtered_kmers = []

    for x in df.columns:
        # print(len(y_train), df[x].shape, df_random[x].shape)
        kmer_corr = abs(np.corrcoef(y_train, df[x])[0, 1])
        random_corr = abs(np.corrcoef(y_train, df_random[x])[0, 1])
        
        if kmer_corr >= random_corr:
            filtered_kmers.append(x)
        
    return filtered_kmers


# generate random sequences
# bedtools random -n 10000 -l 150 -seed 123 -g ../../ref/U00096.2.chrom.sizes > genome_random_10000.bed
# bedtools getfasta -fi ../../ref/U00096.2.fasta -bed genome_random_10000.bed -fo genome_random_10000.fasta -name -s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', help='Filename of training set')
    parser.add_argument('test', help='Filename of test set')
    parser.add_argument('output_train', help='Output file name train')
    parser.add_argument('output_test', help='Output file name test')
    parser.add_argument('min_k', type=int, help='Minimum k-mer length')
    parser.add_argument('max_k', type=int, help='Max k-mer length')
    parser.add_argument('--random', help='''Text file of random control sequences to 
        generate random k-mer counts''')

    args = parser.parse_args()
    
    sequences, y_train = read_file(args.train)
    # sequences, y_train = read_file('tss_scramble_peak_expression_model_format_floored_train_genome_split.txt')
    print('loaded training data')
    test_sequences, y_test = read_file(args.test)
    # test_sequences, y_test = read_file('tss_scramble_peak_expression_model_format_floored_test_genome_split.txt')
    print('loaded test data')

    X_train_all = pd.DataFrame()
    X_test_all = pd.DataFrame()

    for k in range(args.min_k, args.max_k+1):
    # for k in range(min_k, max_k+1):
        print("Train:")
        X_train = generate_kmer_counts(sequences, k)
        X_train_all = pd.concat([X_train_all, X_train], axis=1)
        print("Test:")
        X_test = generate_kmer_counts(test_sequences, k)
        X_test_all = pd.concat([X_test_all, X_test], axis=1)


    if args.random:
        # read in random sequences
        random_sequences = [line.strip().split('\t')[1] for line in open(args.random)]
        X_random_all = pd.DataFrame()
        for k in range(args.min_k, args.max_k+1):
        # for k in range(min_k, max_k+1):
            X_random = generate_kmer_counts(random_sequences, k)
            X_random_all = pd.concat([X_random_all, X_random], axis=1)

        print("Filtering k-mers...")
        filtered_kmers = kmer_filter_random(X_train_all, X_random_all, y_train)
        X_train_filtered = X_train_all[filtered_kmers]
        X_test_filtered = X_test_all[filtered_kmers]

        print("Writing training set...")
        X_train_filtered['y'] = y_train
        X_train_filtered.to_csv(args.output_train, sep='\t', index=False)

        print("Writing test set...")
        X_test_filtered['y'] = y_test
        X_test_filtered.to_csv(args.output_test, sep='\t', index=False)
    else:
        print("Writing training set...")
        X_train_all['y'] = y_train
        X_train_all.to_csv(args.output_train, sep='\t', index=False)

        print("Writing test set...")
        X_test_all['y'] = y_test
        X_test_all.to_csv(args.output_test, sep='\t', index=False)

