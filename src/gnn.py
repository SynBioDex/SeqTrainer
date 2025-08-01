import pandas as pd
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
python_executable = sys.executable

current_dir = os.path.abspath('')
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



def xml_to_nt_and_get_y_measure(file_name):
    g = Graph()
    g.parse(os.path.join(sbol_path, file_name), format="xml")
    g.serialize(destination=os.path.join(nt_path, file_name.replace(".xml", ".nt")), format="nt")

    sparql_query ='''PREFIX om: <http://www.ontology-of-units-of-measure.org/resource/om-2/>

    SELECT ?numericalValue
    WHERE {
    ?s om:hasNumericalValue ?numericalValue .
    }
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
    

def convert_all_xml_to_nt_and_get_y_measures():
    y_measures = []
    for file_name in os.listdir(os.path.join(sbol_path)):
        y = xml_to_nt_and_get_y_measure(file_name)
        y_measures.append(y)
    return y_measures

def return_heterograph_for_one_nt(nt_file_name, node_names, edge_names):
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path_numeric = os.path.join(temp_dir, "save_path_numeric")
        path = os.path.join(temp_dir, "path")
        os.makedirs(save_path_numeric, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        content_config_string = f'''
        [InputPath]
        input_path = {os.path.join(nt_path, nt_file_name)}

        [SavePath]
        save_path_numeric_graph = {save_path_numeric}
        save_path_mapping = {path}

        [NLD]
        nld_class = ModuleDefinition

        [EMBEDDING]
        embedding_model = allenai/scibert_scivocab_uncased

        [Nodes]
        classes = ComponentDefinition, Sequence, ModuleDefinition, Module, FunctionalComponent, Component, SequenceAnnotation, Range

        ComponentDefinition = http://sbols.org/v2#ComponentDefinition
        Sequence = http://sbols.org/v2#Sequence
        ModuleDefinition = http://sbols.org/v2#ModuleDefinition
        Module = http://sbols.org/v2#Module
        FunctionalComponent = http://sbols.org/v2#FunctionalComponent
        Component = http://sbols.org/v2#Component
        SequenceAnnotation = http://sbols.org/v2#SequenceAnnotation
        Range = http://sbols.org/v2#Range

        [SimpleEdges]
        edge_names = ComponentDefinition_Sequence, ComponentDefinition_SequenceAnnotation
        ComponentDefinition_Sequence_start_node = ComponentDefinition
        ComponentDefinition_Sequence_properties = http://sbols.org/v2#sequence
        ComponentDefinition_Sequence_end_node = Sequence
        ComponentDefinition_SequenceAnnotation_start_node = ComponentDefinition
        ComponentDefinition_SequenceAnnotation_properties = http://sbols.org/v2#sequenceAnnotation
        ComponentDefinition_SequenceAnnotation_end_node = SequenceAnnotation

        [N-HopEdges]
        edge_names = ComponentDefinition_Range, ModuleDefinition_ComponentDefinition, ModuleDefinition_ModuleDefinition, ComponentDefinition_ComponentDefinition
        ComponentDefinition_Range_start_node = ComponentDefinition
        ComponentDefinition_Range_hop1_properties = http://sbols.org/v2#sequenceAnnotation
        ComponentDefinition_Range_hop2_properties = http://sbols.org/v2#location
        ComponentDefinition_Range_end_node = Range
        ModuleDefinition_ComponentDefinition_start_node = ModuleDefinition
        ModuleDefinition_ComponentDefinition_hop1_properties = http://sbols.org/v2#functionalComponent
        ModuleDefinition_ComponentDefinition_hop2_properties = http://sbols.org/v2#definition
        ModuleDefinition_ComponentDefinition_end_node = ComponentDefinition
        ModuleDefinition_ModuleDefinition_start_node = ModuleDefinition
        ModuleDefinition_ModuleDefinition_hop1_properties = http://sbols.org/v2#module
        ModuleDefinition_ModuleDefinition_hop2_properties = http://sbols.org/v2#definition
        ModuleDefinition_ModuleDefinition_end_node = ModuleDefinition
        ComponentDefinition_ComponentDefinition_start_node = ComponentDefinition
        ComponentDefinition_ComponentDefinition_hop1_properties = http://sbols.org/v2#component
        ComponentDefinition_ComponentDefinition_hop2_properties = http://sbols.org/v2#definition
        ComponentDefinition_ComponentDefinition_end_node = ComponentDefinition

        [N-ArayEdges]
        edge_names = ComponentDefinition_Range
        ComponentDefinition_Range_start_node = ComponentDefinition
        ComponentDefinition_Range_properties = http://sbols.org/v2#sequenceAnnotation, http://sbols.org/v2#location
        ComponentDefinition_Range_end_node = Range

        [N-ArayFeaturePath]
        ComponentDefinition_Range_feature_path = http://sbols.org/v2#sequenceAnnotation, http://sbols.org/v2#location

        [N-ArayFeatureValue]
        ComponentDefinition_Range_feature_value = http://sbols.org/v2#start, http://sbols.org/v2#end
        '''

        topo_config_string = f'''
        [InputPath]
        input_path = {os.path.join(nt_path, nt_file_name)}

        [SavePath]
        save_path_numeric_graph = {save_path_numeric}
        save_path_mapping = {path}

        [MODEL] ;required, options = transe / complex / distmult / rotate
        kge_model = distmult

        [Nodes]
        classes = ComponentDefinition, Sequence, ModuleDefinition, Module, FunctionalComponent, Component, SequenceAnnotation, Range

        ComponentDefinition = http://sbols.org/v2#ComponentDefinition
        Sequence = http://sbols.org/v2#Sequence
        ModuleDefinition = http://sbols.org/v2#ModuleDefinition
        Module = http://sbols.org/v2#Module
        FunctionalComponent = http://sbols.org/v2#FunctionalComponent
        Component = http://sbols.org/v2#Component
        SequenceAnnotation = http://sbols.org/v2#SequenceAnnotation
        Range = http://sbols.org/v2#Range

        [SimpleEdges]
        edge_names = ComponentDefinition_Sequence, ComponentDefinition_SequenceAnnotation
        ComponentDefinition_Sequence_start_node = ComponentDefinition
        ComponentDefinition_Sequence_properties = http://sbols.org/v2#sequence
        ComponentDefinition_Sequence_end_node = Sequence
        ComponentDefinition_SequenceAnnotation_start_node = ComponentDefinition
        ComponentDefinition_SequenceAnnotation_properties = http://sbols.org/v2#sequenceAnnotation
        ComponentDefinition_SequenceAnnotation_end_node = SequenceAnnotation

        [N-HopEdges]
        edge_names = ComponentDefinition_Range, ModuleDefinition_ComponentDefinition, ModuleDefinition_ModuleDefinition, ComponentDefinition_ComponentDefinition
        ComponentDefinition_Range_start_node = ComponentDefinition
        ComponentDefinition_Range_hop1_properties = http://sbols.org/v2#sequenceAnnotation
        ComponentDefinition_Range_hop2_properties = http://sbols.org/v2#location
        ComponentDefinition_Range_end_node = Range
        ModuleDefinition_ComponentDefinition_start_node = ModuleDefinition
        ModuleDefinition_ComponentDefinition_hop1_properties = http://sbols.org/v2#functionalComponent
        ModuleDefinition_ComponentDefinition_hop2_properties = http://sbols.org/v2#definition
        ModuleDefinition_ComponentDefinition_end_node = ComponentDefinition
        ModuleDefinition_ModuleDefinition_start_node = ModuleDefinition
        ModuleDefinition_ModuleDefinition_hop1_properties = http://sbols.org/v2#module
        ModuleDefinition_ModuleDefinition_hop2_properties = http://sbols.org/v2#definition
        ModuleDefinition_ModuleDefinition_end_node = ModuleDefinition
        ComponentDefinition_ComponentDefinition_start_node = ComponentDefinition
        ComponentDefinition_ComponentDefinition_hop1_properties = http://sbols.org/v2#component
        ComponentDefinition_ComponentDefinition_hop2_properties = http://sbols.org/v2#definition
        ComponentDefinition_ComponentDefinition_end_node = ComponentDefinition

        [N-ArayEdges]
        edge_names = ComponentDefinition_Range
        ComponentDefinition_Range_start_node = ComponentDefinition
        ComponentDefinition_Range_properties = http://sbols.org/v2#sequenceAnnotation, http://sbols.org/v2#location
        ComponentDefinition_Range_end_node = Range

        [EmbeddingClasses] 
        class_list = http://sbols.org/v2#ComponentDefinition, http://sbols.org/v2#Sequence, http://sbols.org/v2#ModuleDefinition, http://sbols.org/v2#Module, http://sbols.org/v2#FunctionalComponent, http://sbols.org/v2#Component, http://sbols.org/v2#SequenceAnnotation, http://sbols.org/v2#Range

        [EmbeddingPredicates] 
        pred_list = http://sbols.org/v2#location, http://sbols.org/v2#sequenceAnnotation, http://sbols.org/v2#functionalComponent, http://sbols.org/v2#definition, http://sbols.org/v2#module, http://sbols.org/v2#component, https://sbols.org/v2#sequence, http://www.w3.org/1999/02/22-rdf-syntax-ns#type
        '''

        with open(os.path.join(temp_dir,'config.ini'), 'w') as file:
            file.write(content_config_string)

        with open(os.path.join(temp_dir,'topo-config.ini'), 'w') as file:
            file.write(topo_config_string)

        content_result = subprocess.run([python_executable, "autordf2gml.py", "--config_path", os.path.join(temp_dir,"config.ini")], shell=False, capture_output=True, text=True)
        topo_result = subprocess.run([python_executable, "autordf2gml-tb.py", "--config_path", os.path.join(temp_dir,"topo-config.ini")], shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace')

        if (topo_result.returncode != 0):
            print(f"topology-based conversion failed with return code {topo_result.returncode}.")

        
        data = HeteroData()
        local_indices_map = {}            

        for node_name in node_names:
            node_features_df = pd.read_csv(os.path.join(save_path_numeric, f'pivoted_df_{node_name}.csv'), header=None).astype(float)
            node_tensor = torch.tensor(node_features_df.values, dtype=torch.float)
            id_mapping_df = pd.read_csv(os.path.join(path, f'pivoted_df_{node_name}.csv'))
            subject_mapping_dict = id_mapping_df.set_index('subject')['mapping'].to_dict()

            topo_df = pd.read_csv(os.path.join(save_path_numeric, f"uri_list_{node_name}.csv"))
            topo_df['mapped_col'] = topo_df['subject'].map(subject_mapping_dict)
            topo_df = topo_df.sort_values(by='mapped_col')
            topo_df = topo_df.drop(['subject', 'mapped_col'], axis=1).astype(float)
            node_tensor_topo = torch.tensor(topo_df.values, dtype=torch.float)


            node_tensor = torch.concat([node_tensor, node_tensor_topo], dim=1)
            
            data[node_name].node_id = torch.arange(len(id_mapping_df))
            data[node_name].x = node_tensor
            local_indices_map = subject_mapping_dict | local_indices_map
                
        for edge in edge_names:
            df = pd.read_csv( os.path.join(save_path_numeric, f"edge_list_{edge}.csv"), header=None)
            
            src = df[0].values
            dst = df[1].values

            src = torch.tensor([local_indices_map[src[i]] for i in range(len(src))], dtype=torch.long)
            dst = torch.tensor([local_indices_map[dst[i]] for i in range(len(dst))], dtype=torch.long)
            data[edge.split("_")[0], f'has_{edge.split("_")[1]}', edge.split("_")[1]].edge_index = torch.stack([src, dst], dim=0)
        
        return data
    
class HeteroGNN_GraphLevel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)

        has_seq = ('ComponentDefinition', 'has_Sequence', 'Sequence')
        has_seq_anno = ('ComponentDefinition', 'has_SequenceAnnotation', 'SequenceAnnotation')
        has_range = ('ComponentDefinition', 'has_Range', 'Range')
        has_comp = ('ModuleDefinition', 'has_ComponentDefinition', 'ComponentDefinition')
        comp_has_comp = ('ComponentDefinition', 'has_ComponentDefinition', 'ComponentDefinition')
        has_mod = ('ModuleDefinition', 'has_ModuleDefinition', 'ModuleDefinition')

        edges = [has_seq, has_seq_anno, has_range, has_comp, comp_has_comp, has_mod]

        unique_nodes = set()

        for edge_tuple in edges:
            unique_nodes.add(edge_tuple[0])
            unique_nodes.add(edge_tuple[2])


        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                has_seq:SAGEConv((-1, -1), hidden_channels),
                has_seq_anno: SAGEConv((-1, -1), hidden_channels),
                has_comp: GATConv((-1, -1), hidden_channels, add_self_loops=False),
                has_mod: GCNConv(-1, hidden_channels),
                comp_has_comp: GCNConv(-1, hidden_channels),
                has_range: GATConv((-1, -1), hidden_channels, add_self_loops=False),
            }, aggr='sum')
            self.convs.append(conv)
      
        total_hidden = len(unique_nodes) * hidden_channels
        self.lin = torch.nn.Sequential(
            Linear(total_hidden, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.7),
            Linear(hidden_channels, 1) 
        )

    def forward(self, x_dict, edge_index_dict, batch_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}


        # Pool all node types, then concatenate
        pooled = [
            global_mean_pool(x_dict[node_type], batch_dict[node_type])
            for node_type in x_dict
        ]        
        graph_embeddings = torch.cat(pooled, dim=-1)  # shape: [batch_size, total_hidden]
        return self.lin(graph_embeddings).view(-1)     # shape: [batch_size]

# def return_all_graphs(node_classes, y_measures, all_edges_formatted):
#     all_data = []
#     files = os.listdir(nt_path)
    
#     for i, filename in enumerate(tqdm(files, desc="Generating graphs")):
#         data = return_heterograph_for_one_nt(filename, node_classes, all_edges_formatted)
#         data.y = y_measures[i]
#         all_data.append(data)
    
#     return all_data
    
import concurrent.futures

def process_one(args):
    i, filename, node_classes, all_edges_formatted, y_measures = args
    data = return_heterograph_for_one_nt(filename, node_classes, all_edges_formatted)
    data.y = y_measures[i]
    return data

def return_all_graphs(node_classes, y_measures, all_edges_formatted):
    files = os.listdir(nt_path)

    tasks = [
        (i, filename, node_classes, all_edges_formatted, y_measures)
        for i, filename in enumerate(files)
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        all_data = list(tqdm(
            executor.map(process_one, tasks),
            total=len(files),
            desc="Generating graphs"
        ))

    return all_data



def train_GNN(all_data):

    random.shuffle(all_data)
    train_size = int(0.8 * len(all_data))
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=10, shuffle=False) 

    model = HeteroGNN_GraphLevel(metadata=all_data[0].metadata(), hidden_channels=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(1, 200):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            target = torch.tensor(batch.y, dtype=torch.float32).view(-1)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * batch.num_graphs

        # Validation Phase
        model.eval() 
        total_val_loss = 0
        with torch.no_grad(): 
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                target = torch.tensor(batch.y, dtype=torch.float32).view(-1)
                loss = F.mse_loss(out, target)
                total_val_loss += loss.item() * batch.num_graphs

        print(f"Epoch {epoch:03d} | Train Loss: {total_train_loss / len(train_loader.dataset):.4f} | Val Loss: {total_val_loss / len(val_loader.dataset):.4f}")

node_classes = [
    "ComponentDefinition",
    "Sequence",
    "ModuleDefinition",
    "Module",
    "FunctionalComponent",
    "Component",
    "SequenceAnnotation",
    "Range"
]

all_edges_formatted = [
    "ComponentDefinition_Sequence",
    "ComponentDefinition_SequenceAnnotation",
    "ComponentDefinition_Range",
    "ModuleDefinition_ComponentDefinition",
    "ModuleDefinition_ModuleDefinition",
    "ComponentDefinition_ComponentDefinition",
]

def return_standardized_labels(y_measures):
    all_y_values_np = np.array(y_measures).reshape(-1, 1)
    scaler_y = StandardScaler()
    scaler_y.fit(all_y_values_np)
    return all_y_values_np

import pickle

if __name__ == "__main__":
    with open("numbers.pkl", "rb") as f:
        y_measures = pickle.load(f)

    y = return_standardized_labels(y_measures)

    # Generate graphs
    all_graphs = return_all_graphs(node_classes, y, all_edges_formatted)

    # Save to pickle
    with open("graphs_data.pkl", "wb") as f:
        pickle.dump(all_graphs, f)