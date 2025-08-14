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
from sklearn.preprocessing import OneHotEncoder
from dotenv import load_dotenv
import zipfile
import numpy as np
# import xgboost as xgb

class DNAConvolution(torch.nn.Module):
    def __init__(self):
        super().__init__()
      
