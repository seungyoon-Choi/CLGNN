import os.path as osp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
device = f'cuda:{6}' if torch.cuda.is_available() else 'cpu'
# load the Cora dataset_lp
dataset_lp = 'Cora'
path = osp.join('../','data',dataset_lp)
dataset_lp = Planetoid(path, dataset_lp, transform=T.NormalizeFeatures())
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset_lp.num_features, 128)
        self.conv2 = GCNConv(128, 64)
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)
    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits
    def decode_all(self, z):
        prob_adj = z @ z.t()
        prob_adj = torch.sigmoid(prob_adj)
        return (prob_adj>0.9).nonzero(as_tuple=False).t()