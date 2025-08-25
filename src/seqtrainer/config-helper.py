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
sbol_path = os.path.join(data_path, 'sbol_data')
    
def get_all_nodes(file_name, filter_uri, ws_uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
    g = Graph()
    g.parse(os.path.join(sbol_path, file_name), format="xml")

    sparql_query = f'''
    PREFIX ws: <{ws_uri}>
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

def get_all_edges(file_name, sbol_types, base_uri="http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
    
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
        print("No edges found.")

    return pd.DataFrame(edge)
 
def build_content_config(input_path, save_path_numeric, save_path, nld_class, all_node_uris, edges):
    
    config = configparser.ConfigParser()
    config.optionxform = str
    classes_dict = {"classes": []}
    for uri in all_node_uris:
        classes_dict[uri.split("#")[1]] = uri
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
    edges_dict["edge_names"] = ", ".join(edges_dict["edge_names"])


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

    with open("config.txt", "w") as configfile:
        config.write(configfile)

def build_content_config(input_path, save_path_numeric, save_path, nld_class, all_node_uris, edges):
    
    config = configparser.ConfigParser()
    config.optionxform = str
    classes_dict = {"classes": []}
    for uri in all_node_uris:
        classes_dict[uri.split("#")[1]] = uri
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
    edges_dict["edge_names"] = ", ".join(edges_dict["edge_names"])


    config["InputPath"] = {
        "input_path": input_path    
    }

    config["SavePath"] = {
        "save_path_numeric_graph": save_path_numeric,
        "save_path_mapping": save_path
    }

    config["MODEL"] = {
        "kge_model": "dismult"
    }

    config["EMBEDDING"] = {
        "embedding_model": "allenai/scibert_scivocab_uncased"
    }

    config["Nodes"] = classes_dict

    config["SimpleEdges"] = edges_dict

    config["N-HopEdges"] = {
    }

    config["EmbeddingClasses"] = {
        "class_list": classes_dict["classes"]
    }   

    config["EmbeddingPredicates"] = {
        "pred_list": ", ".join(edges['uritype'])
    }

    with open("config.txt", "w") as configfile:
        config.write(configfile)


    return edges_dict["edge_names"]

# n hop edges will be a list of ordered lists
n_hop = [["ModuleDefinition_FunctionalComponent", "FunctionalComponent_ComponentDefinition"], ["ComponentDefinition_Component", "Component_ComponentDefinition"],
         ["ComponentDefinition_SequenceAnnotation", "SequenceAnnotation_Range"], ["ModuleDefinition_Module", "Module_ModuleDefinition"]]
def create_n_hop_edges(config_path, n_hop_edges):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(config_path)
    n_hop_dict = {}
    connections = []
    n_hop_dict["edge_names"] = ", ".join([f"{properties[0].split('_')[0]}_{properties[-1].split('_')[1]}"for properties in n_hop_edges])
    for n_hop in n_hop_edges:
        final_connection = f"{n_hop[0].split('_')[0]}_{n_hop[-1].split('_')[1]}"
        for i, connection in enumerate(n_hop):
            start = cfg['SimpleEdges'].get(f"{connection}_start_node", None)
            end = cfg['SimpleEdges'].get(f"{connection}_end_node", None)
            if (start and end):
                properties = cfg['SimpleEdges'].get(f"{connection}_properties")
                if (i == 0): n_hop_dict[f"{final_connection}_start_node"] = start
                n_hop_dict[f"{final_connection}_hop{i+1}_properties"] = properties
                if (i == len(n_hop) - 1): n_hop_dict[f"{final_connection}_end_node"] = end
            else:
                raise ValueError(f"The edge {connection} is not defined in the config file.")
            connections.append(connection)
            
            for suffix in ["_start_node", "_end_node", "_properties"]:
                cfg.remove_option("SimpleEdges", f"{connection}{suffix}")
    
    cfg["SimpleEdges"]["edge_names"] = ", ".join([edge for edge in cfg["SimpleEdges"]["edge_names"].split(", ") if edge not in connections])
    cfg["N-HopEdges"] = n_hop_dict

    cfg.write(".")

    print("Config file created successfully.")



s = create_n_hop_edges("config.txt", n_hop)

# print(get_sequences('sample_design_0.xml')) 
# # 
# [ModuleDefinition_Module, Module_ModuleDefinition]

# User defines this
om = "http://www.ontology-of-units-of-measure.org/resource/om-2/"
measure_uri = "http://www.ontology-of-units-of-measure.org/resource/om-2/Measure"  
# all_node_uris = get_all_nodes('sample_design_0.xml', type, measure_uri)
# edges = get_all_edges('sample_design_0.xml', type, all_node_uris)


nodes = get_all_nodes(os.path.join(sbol_path,'sample_design_0.xml'), measure_uri)
edges = get_all_edges(os.path.join(sbol_path,'sample_design_0.xml'), nodes)

# ComponentDefinition_Range, ModuleDefinition_ComponentDefinition,
# ModuleDefinition_ModuleDefinition, ComponentDefinition_ComponentDefinition

# ask user to define which n hop to have etc. (ModuleDefinition_FunctionalComponent -> FunctionalComponent_ComponentDefinition)
# validate that the user defined n hops are valid
# then create the n hop edges
# then remove from simple edges

# print(edges)
# c = build_content_config(os.path.join(sbol_path,'sample_design_0.xml'), ".", ".", "ModuleDefinition", nodes, edges)
# print(c)

