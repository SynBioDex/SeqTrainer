import sbol2
import os
from rdflib import Graph
from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
import torch
from torch import Tensor
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

import pickle
python_executable = sys.executable
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'data')
attachments_path = os.path.join(current_dir, '..', 'attachments')
pulled_attachments_path = os.path.join(current_dir, '..', 'pulled_attachments')
sbol_path = os.path.join(current_dir, '..', 'sbol_data')
downloaded_sbol_path = os.path.join(current_dir, '..', 'downloaded_sbol')
original_data_path = os.path.join(data_path, 'original_data')
nt_path = os.path.join(current_dir, '..', 'nt_data')
scripts_path = os.path.join(current_dir, 'scripts')
model_data_path = os.path.join(data_path, 'processed_data', 'replicated_models')
model_output_path = os.path.join('..', 'model_outputs')


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
    
all_node_uris = get_all_nodes('sample_design_0.xml', type, measure_uri)
print(all_node_uris)

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


df = get_all_edges('sample_design_0.xml', type, all_node_uris)
df.to_csv('sample_design_0.csv', index=False)
hop2 = df.merge(df, left_on="node2", right_on="node1", suffixes=("_1", "_2"))
hop2 = hop2[["node1_1", "uritype_1", "node2_1", "uritype_2", "node2_2"]]
hop2.to_csv('merged.csv', index=False)