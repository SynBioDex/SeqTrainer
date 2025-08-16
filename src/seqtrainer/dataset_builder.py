import sbol2
import os
from rdflib import Graph
import pandas as pd
import numpy as np
from rdflib.query import ResultRow
import sys 
import configparser

def get_sequence_from_sbol(file_path):
    g = Graph()
    g.parse(file_path, format="xml")

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


def get_y_label(file_path, uri="<http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue>"):
    g = Graph()
    g.parse(file_path, format="xml")

    sparql_query = f'''

    SELECT ?numericalValue
    WHERE {{
    ?s {uri} ?numericalValue .
    }}
    '''
    query_result = g.query(sparql_query)

    if query_result:
        for row in query_result:
            if isinstance(row, ResultRow):
                return float(row.numericalValue) 
                
            else:
                print(row)
    else:
        print("No numerical values found.")

def get_sequences_from_sbol(file_paths):
    all_sequences = []
    for file_path in file_paths:
        sequence = get_sequence_from_sbol(file_path)
        all_sequences.append(sequence)
    return all_sequences

def get_y_labels_from_sbol(file_paths, uri):
    y_labels = []
    for file_path in file_paths:
        y_label = get_y_label(file_path, uri)
        y_labels.append(y_label)
    return y_labels

def build_dataset(file_paths, y_uri):
    y_labels = get_y_labels_from_sbol(file_paths, y_uri)
    sequences = get_sequences_from_sbol(file_paths)
    df = pd.DataFrame({"sequence": sequences, "target": y_labels})
    return df

# current_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = os.path.join(current_dir, '..', 'data')
# sbol_path = os.path.join(data_path, 'sbol_data')
# paths = [os.path.join(sbol_path, file_name) for file_name in os.listdir(sbol_path)][:100]


# df = build_dataset(paths, "<http://www.ontology-of-units-of-measure.org/resource/om-2/hasNumericalValue>")
# df.to_csv("dataset.csv", index=False)
    