import argparse
import random
import pandas as pd
import numpy as np
from math import floor
from copy import deepcopy


def create_splits(train_fraction):

	test_fraction = 1 - train_fraction
	split_fraction = test_fraction / 2
	split_size = int(floor(genome_size * split_fraction))

	# genome coordinates are 1-based
	splits = [(i, i + split_size - 1) for i in range(1, genome_size + 1, split_size)]

	# remove last split which contains the remainder of the split size
	splits.pop()
	# fix last remaining split
	last_split = splits.pop()
	splits.append((last_split[0], genome_size))

	return splits


def find_equidistant_pair(splits, split1):
	'''
	Input: list of tuples corresponding to the start and end of the genome split
	Output: two splits that are equidistant from the origin. The genome is circular
	and the origin is at coordinate 1 and coordinate <genome size>, the second 
	element in the last tuple
	'''


	start = splits[0][0]
	end = splits[-1][1]

	# to accurately compare distances, determine how much longer the last split
	# is since it contains the modulus of genome % split_size
	split_size = splits[0][1] - splits[0][0] + 1
	remainder = end % split_size

	# # randomly choose split
	# split1 = random.choice(splits)
	# determine distance from origin
	distance1 = split1[0] - start 
	# iterate through splits until we find out that is equidistant
	for i in range(len(splits)):
		split2 = splits[i]
		distance2 = abs(split2[1] - end + remainder)
		if distance1 == distance2:
			return split2

	return None


def in_range(split,  x, y):
	# Check if (x, y) is contained in split, a tuple (start, end)
	if x >= split[0] and y <= split[1]:
		return True
	else:
		return False


if __name__ == '__main__':

	parser = argparse.ArgumentParser('''Define genome positions used for train/test.
		Test and train locations will be equidistant from origin of replication.''')
	parser.add_argument('train_fraction', type=float, help='''Fraction of genomic
		positions used for training''')
	parser.add_argument('genome_size', type=int, help='Size of genome (bp)')
	parser.add_argument('infile_train',  help='''filename of formatted expression data.
		No column names.
		Must include columns for variant, expn_med_fitted_scaled, start, end''')
	parser.add_argument('infile_test', help='''filename of formatted expression data.
		Must include columns for variant, expn_med_fitted_scaled, start, end''')
	parser.add_argument('--floor', action='store_true', help='''Flatten all scores below 1 and set to 1.
		Decreases importance of inactive sequences''')
	parser.add_argument('--classification', action='store_true',
		help='Split datasets for classification')
	# parser.add_argument('--neg_threshold', type=float, nargs='?', const=1,
	# 	help='Only consider negatives below this threshold for train/test')
	# parser.add_argument('--pos_threshold', type=float, nargs='?', const=1,
	# 	help='Only consider positives above this threshold for train/test')

	args = parser.parse_args()

	train_fraction = args.train_fraction
	genome_size = args.genome_size

	splits = create_splits(train_fraction)	

	# assign test splits so they are equidistant from the origin (coordinate 1, circular)
	# we created the splits so two splits are equal to the test fraction
	# we must choose two splits equidistant and assign to test, the rest to train
	test_split1 = splits[1]
	test_splits = [test_split1, find_equidistant_pair(splits, test_split1)]

	# create train/test look-up for each split
	split_lookup = {x : 'test' if x in test_splits else 'train' for x in splits}

	# read in datasets
	train = pd.read_table(args.infile_train, sep='\t', header=None)
	test = pd.read_table(args.infile_test, sep='\t', header=None)

	train.columns = ['variant', 'expn_med_fitted_scaled', 'start', 'end', 'name', 'active']
	test.columns = ['variant', 'expn_med_fitted_scaled', 'start', 'end', 'name', 'active']


	if args.floor:
		# set all train values less than 1 equal to 1. This 
		# decreases the importance of the inactive/noisy values
		train['expn_med_fitted_scaled'] = np.where(train['expn_med_fitted_scaled'] < 1, 1, train['expn_med_fitted_scaled'])

	
	# define output prefixes
	train_prefix = args.infile_train.replace('.txt', '') 
	test_prefix = args.infile_test.replace('.txt', '')
	if args.floor:
		train_prefix += '_floored'
		test_prefix += '_floored'

	# For train, only sequences that fall into train splits will be written
	if not args.classification:
		outfile_train_name = train_prefix + '_train_genome_split.txt'
		with open(outfile_train_name, 'w') as outfile_train:
			for i in range(len(train)):
				
				x = train.start.iloc[i]
				y = train.end.iloc[i]
				
				for j in range(len(splits)):
					if in_range(splits[j], x, y):
						if split_lookup[splits[j]] == 'train':
							outfile_train.write(train.variant.iloc[i] + '\t')
							outfile_train.write(str(train.expn_med_fitted_scaled.iloc[i]) + '\n')	
						# else, don't write to output
	else:
		# write file for positives and negatives according to threshold
		outfile_train = open(train_prefix + '_train_genome_split_classification.txt', 'w')
		for i in range(len(train)):
			 x = train.start.iloc[i]
			 y = train.end.iloc[i]
			 # value = train.expn_med_fitted_scaled.iloc[i]
			 active_flag = train.active.iloc[i]
			 name = train.name.iloc[i]
			 seq = train.variant.iloc[i]

			 for j in range(len(splits)):
			 	if in_range(splits[j], x, y):
					if split_lookup[splits[j]] == 'train':
						# if value < args.neg_threshold:
						if active_flag == 'inactive':
							label = '0'
							outfile_train.write(seq + '\t' + label + '\n')
						# elif value >= args.pos_threshold:
						elif active_flag == 'active':
							label = '1'
							outfile_train.write(seq + '\t' + label + '\n')
		outfile_train.close()

	# For test, only sequences that fall into test splits will be written
	if not args.classification:
		outfile_test_name = test_prefix + '_test_genome_split.txt'
		with open(outfile_test_name, 'w') as outfile_test:
			for i in range(len(test)):
				
				x = test.start.iloc[i]
				y = test.end.iloc[i]
				
				for j in range(len(splits)):
					if in_range(splits[j], x, y):
						if split_lookup[splits[j]] == 'test':
							outfile_test.write(test.variant.iloc[i] + '\t')
							outfile_test.write(str(test.expn_med_fitted_scaled.iloc[i]) + '\n')
						# else, don't write to output
	else:	
		outfile_test = open(test_prefix + '_test_genome_split_classification.txt', 'w')
		for i in range(len(test)):
			
			x = test.start.iloc[i]
			y = test.end.iloc[i]
			value = test.expn_med_fitted_scaled.iloc[i]
			active_flag = test.active.iloc[i]
			name = test.name.iloc[i]
			seq = test.variant.iloc[i]
			
			for j in range(len(splits)):
				if in_range(splits[j], x, y):
					if split_lookup[splits[j]] == 'test':
						
						# if value < args.neg_threshold:
						if active_flag == 'inactive':
							label = '0'
							outfile_test.write(seq + '\t' + label + '\n')
						elif active_flag == 'active':
						# elif value >= args.pos_threshold:
							label = '1'
							outfile_test.write(seq + '\t' + label + '\n')
		outfile_test.close()
