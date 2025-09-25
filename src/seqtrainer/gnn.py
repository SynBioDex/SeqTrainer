
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
import configparser
import pickle
from dataset_builder import get_y_label
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

def xml_to_nt(input_path, output_path):
    # Convert XML to NTriples
    g = Graph()
    g.parse(input_path, format="xml")
    g.serialize(destination=output_path, format="nt")

#os.path.join(nt_path, file_name.replace(".xml", ".nt"))

# def get_y_label(input_path, base_uri="http://www.ontology-of-units-of-measure.org/resource/om-2/"):

#     # Load xml file
#     g = Graph()
#     g.parse(input_path, format="xml")

#     sparql_query = f'''
#     PREFIX om: <{base_uri}>
#     SELECT ?numericalValue
#     WHERE {{
#     ?s om:hasNumericalValue ?numericalValue .
#     }}
#     '''
#     query_result = g.query(sparql_query)

#     if query_result:
#         for row in query_result:
#             if isinstance(row, ResultRow):
#                 return float(row.numericalValue) 
                
#             else:
#                 print(row)
#     else:
#         print("No numerical values found.")
    
def convert_all_xml_to_nt_and_get_y_measures(sbol_data_path, output_data_path, y_uri):
    y_measures = []

    for file_name in os.listdir(sbol_data_path):
        xml_to_nt(os.path.join(sbol_data_path, file_name), os.path.join(output_data_path, file_name.replace(".xml", ".nt")))
        y = get_y_label(os.path.join(sbol_data_path, file_name), y_uri)
        y_measures.append(y)

    return y_measures

def return_heterograph_for_one_nt(nt_path, node_names, content_config_dir, topo_config_dir, edge_names):
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path_numeric = os.path.join(temp_dir, "save_path_numeric")
        path = os.path.join(temp_dir, "path")
        os.makedirs(save_path_numeric, exist_ok=True)
        os.makedirs(path, exist_ok=True)

        cfg = configparser.ConfigParser()
        cfg.optionxform = str

        cfg.read(content_config_dir)
        cfg["InputPath"]["input_path"] = nt_path
        cfg.write(content_config_dir)

        cfg.read(topo_config_dir)
        cfg["InputPath"]["input_path"] = nt_path
        cfg.write(topo_config_dir)

        content_result = subprocess.run([python_executable, "autordf2gml.py", "--config_path", content_config_dir], shell=False, capture_output=True, text=True)
        topo_result = subprocess.run([python_executable, "autordf2gml-tb.py", "--config_path", topo_config_dir], shell=False, capture_output=True, text=True, encoding='utf-8', errors='replace')

        print(topo_result.stderr)
        if (topo_result.returncode == 0):
            print("topology-based conversion done!")
        else:
            print(f"topology-based conversion failed with return code {topo_result.returncode}.")

        # Creating the HeteroData 
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
            

            # For each node, assign its numerical vector and mappped ID to the HeteroData
            data[node_name].node_id = torch.arange(len(id_mapping_df))
            data[node_name].x = node_tensor
            local_indices_map = subject_mapping_dict | local_indices_map

                
        for edge in edge_names:

            df = pd.read_csv( os.path.join(save_path_numeric, f"edge_list_{edge}.csv"), header=None)
            src = df[0].values
            dst = df[1].values
            src = torch.tensor([local_indices_map[src[i]] for i in range(len(src))], dtype=torch.long)
            dst = torch.tensor([local_indices_map[dst[i]] for i in range(len(dst))], dtype=torch.long)

            # Add the edge mappings (src_id, dst_id) to the HeteroData
            data[edge.split("_")[0], f'has_{edge.split("_")[1]}', edge.split("_")[1]].edge_index = torch.stack([src, dst], dim=0)
    

        return data
    
class HeteroGNN_GraphLevel(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, num_layers):
        super().__init__()
        self.metadata = metadata  # (node_types, edge_types)
        print("Metadata: ", self.metadata[1])

        unique_nodes = set()
        
        for edge_tuple in self.metadata[1]:
            unique_nodes.add(edge_tuple[0])
            unique_nodes.add(edge_tuple[2])

        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):  # number of layers
            
            conv_dict = {}
            for edge_type in self.metadata[1]: 
                src, _, dst = edge_type
                conv_layer = GATConv((-1, -1), hidden_channels) if src == dst else SAGEConv((-1, -1), hidden_channels)
                conv_dict[edge_type] = conv_layer
        
            hetero_conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(hetero_conv)            

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


def return_all_graphs(node_classes, y_measures, all_edges_formatted):
    all_data = []
    files = os.listdir(nt_path)
    
    for i, filename in enumerate(tqdm(files, desc="Generating graphs")):
        data = return_heterograph_for_one_nt(filename, node_classes, all_edges_formatted)
        data.y = y_measures[i]
        all_data.append(data)
    
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

if __name__ == "__main__":
    with open("numbers.pkl", "rb") as f:
        y_measures = pickle.load(f)

    y = return_standardized_labels(y_measures)

    all_graphs = return_all_graphs(node_classes, y, all_edges_formatted)

    with open("graphs_data.pkl", "wb") as f:
        pickle.dump(all_graphs, f)