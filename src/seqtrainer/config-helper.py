import sbol2
import os
from rdflib import Graph
import pandas as pd
import numpy as np
from rdflib.query import ResultRow
import sys 
import configparser

python_executable = sys.executable
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', '..', 'data')
sbol_path = os.path.join(current_dir, '..', '..', 'sbol_data')
downloaded_sbol_path = os.path.join(current_dir, '..', '..', 'downloaded_sbol')
original_data_path = os.path.join(data_path, 'original_data')
nt_path = os.path.join(current_dir, '..', '..','nt_data')
model_data_path = os.path.join(data_path, 'processed_data', 'replicated_models')
    
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


print(get_sequences('sample_design_0.xml')) 

# User defines this
om = "http://www.ontology-of-units-of-measure.org/resource/om-2/"
measure_uri = "http://www.ontology-of-units-of-measure.org/resource/om-2/Measure"  
type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
all_node_uris = get_all_nodes('sample_design_0.xml', type, measure_uri)
edges = get_all_edges('sample_design_0.xml', type, all_node_uris)

 
def build_config(input_path, save_path_numeric, save_path, nld_class, all_node_uris, edges):

    config = configparser.ConfigParser()
    classes_dict = {uri.split("#")[1]: uri for uri in all_node_uris}
    classes_dict["classes"] = ", ".join(list(classes_dict.keys()))
    edges_dict = {"edge_names": []}
    for _, row in edges.iterrows():
        node1_uri = row["node1"].split("#")[1]
        node2_uri = row["node2"].split("#")[1]
        edge_name = f"{node1_uri}_{node2_uri}"
        edges_dict["edge_names"].append(edge_name)
        edges_dict[edge_name + "_start_node"] = row["node1"].split("#")[1]
        edges_dict[edge_name + "_properties"] = str(row["uritype"])
        edges_dict[edge_name + "_end_node"] = row["node2"].split("#")[1]

    config["InputPath"] = {
        "input_path": input_path    
    }

    config["SavePath"] = {
        "save_path_numeric_graph": save_path_numeric,
        "save_path_mapping": save_path
    }

    config["NLD"] = {
        "nld_class": nld_class
    }

    config["EMBEDDING"] = {
        "embedding_model": "allenai/scibert_scivocab_uncased"
    }

    config["Nodes"] = classes_dict

    config["SimpleEdges"] = edges_dict

    config["N-HopEdges"] = {
    }

    config["N-ArayEdges"] = {

    }

    config["N-ArayFeaturePath"] = {
    }

    config["N-ArayFeatureValue"] = {
    }

    with open("config.ini", "w") as configfile:
        config.write(configfile)