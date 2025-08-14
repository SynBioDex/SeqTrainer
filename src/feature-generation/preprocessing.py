import sbol2
import os
from rdflib import Graph
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from collections import Counter 
import subprocess
import tempfile
from rdflib.query import ResultRow
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
import random
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv, GCNConv, GATConv, global_mean_pool
import sys 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv
import zipfile
import numpy as np
# import xgboost as xgb


import pickle
python_executable = sys.executable
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', '..', 'data')
sbol_path = os.path.join(current_dir, '..', '..', 'sbol_data')
downloaded_sbol_path = os.path.join(current_dir, '..', '..', 'downloaded_sbol')
original_data_path = os.path.join(data_path, 'original_data')
nt_path = os.path.join(current_dir, '..', '..','nt_data')
model_data_path = os.path.join(data_path, 'processed_data', 'replicated_models')


def get_sequences(file_name):

    g = Graph()
    print(os.path.join(sbol_path, file_name))
    g.parse(os.path.join(sbol_path, file_name), format="xml")

    sparql_query ='''PREFIX sbol: <http://sbols.org/v2#>

    SELECT ?sequence
    WHERE {
    ?s sbol:elements ?sequence .
    }
    '''
    query_result = g.query(sparql_query)

    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                return row.sequence
            else:
                print(row)
    else:
        print("No sequence found.")

# print(get_sequences('sample_design_0.xml')) 

# User defines this
om = "http://www.ontology-of-units-of-measure.org/resource/om-2/"

def get_y_label(file_name, base_uri):
    g = Graph()
    g.parse(os.path.join(sbol_path, file_name), format="xml")

    sparql_query = f'''
    PREFIX om: <{base_uri}>
    SELECT ?numericalValue
    WHERE {{
    ?s om:hasNumericalValue ?numericalValue .
    }}
    '''
    query_result = g.query(sparql_query)

    # Process the results
    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                return float(row.numericalValue) 
                
            else:
                print(row)
    else:
        print("No numerical values found.")
    
# print(get_y_label('sample_design_0.xml', om))
type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
def get_all_nodes(file_name, base_uri, filter_uri):
    g = Graph()
    g.parse(os.path.join(sbol_path, file_name), format="xml")

    sparql_query = f'''
    PREFIX ws: <{base_uri}>
    SELECT DISTINCT ?value
    WHERE {{
    ?s ws:type ?value .
    FILTER(?value != <{filter_uri}>)
    }}
    '''

    query_result = g.query(sparql_query)

    vals = []

    # Process the results
    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                vals.append(str(row.value))
                
            else:
                print(row)
    else:
        print("No numerical values found.")

    return vals

measure_uri = "http://www.ontology-of-units-of-measure.org/resource/om-2/Measure"
    
# all_node_uris = get_all_nodes('sample_design_0.xml', type, measure_uri)
# print(all_node_uris)

def get_all_edges(file_name, base_uri, sbol_types):
    
    g = Graph()
    g.parse(os.path.join(sbol_path, file_name), format="xml")
    formatted_types = ",\n    ".join(f"<{uri}>" for uri in sbol_types)

    sparql_query = f"""
    PREFIX rdf: <{base_uri}>

    SELECT DISTINCT ?stype ?prop ?vtype
    WHERE {{
    ?s ?prop ?value .

    ?s rdf:type ?stype .
    FILTER(?stype IN (
        {formatted_types}
    ))

    ?value rdf:type ?vtype .
    FILTER(?vtype IN (
        {formatted_types}
    ))
    }}
    """
    edge = {"node1":[],"uritype":[], "node2":[], }

    query_result = g.query(sparql_query)

    # Process the results
    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                edge['node1'].append(row.stype)
                edge['uritype'].append(row.prop)
                edge['node2'].append(row.vtype)

                
            else:
                print(row)
    else:
        print("No numerical values found.")

    return pd.DataFrame(edge)


# df = get_all_edges('sample_design_0.xml', type, all_node_uris)
# df.to_csv('sample_design_0.csv', index=False)

def one_hot_encode_modern(sequences):
    sequences_array = np.array([[char for char in seq] for seq in sequences])
    print(sequences_array)
    sequence_length = sequences_array.shape[1]
    num_samples = sequences_array.shape[0]

    defined_categories = ['A', 'C', 'G', 'T', 'N']
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

def process_seqs(df, seq_length, seq_col_name):
	padded_seqs = [pad_sequence(x, seq_length) for x in df[seq_col_name]]
	X = one_hot_encode_modern(np.array(padded_seqs))
	return X

df3 = pd.read_csv(os.path.join(original_data_path, "fLP3_Endo2_lb_expression_formatted.txt"), delimiter=" ")
df3 = df3[['variant', 'expn_med']]

def calc_gc(sequences):
    gc_all = []
    for seq in sequences:
        seq = seq.upper()  # ensure consistent casing
        seq_length = len(seq)
        num_gc = seq.count('G') + seq.count('C')
        gc_content = num_gc / seq_length if seq_length > 0 else 0
        gc_all.append(gc_content)
    return np.array(gc_all)

def generate_kmer_counts(sequences, k):
    # all_kmers = [''.join(x) for x in itertools.product(['A', 'C', 'G', 'T'], repeat=k)]
    print("Counting k-mers of length", k, "in all sequences...")
    kmer_counts = [dict(Counter([seq[i:i+k] for i in range(len(seq)-k+1)])) for seq in sequences]
    # ensures all columns are present, even if some keys not present in all dictionaries
    print("Creating features...")
    kmer_df = pd.DataFrame.from_records(kmer_counts)
    kmer_df.fillna(0, inplace=True)
    return kmer_df

kmer = generate_kmer_counts(df3['variant'], 3)
gc = calc_gc(df3['variant'])

