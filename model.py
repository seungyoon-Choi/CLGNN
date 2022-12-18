import torch
import torch.nn as nn 
import torch.nn.functional as F 
from layer import GraphConvolutionLayer, GraphAttentionLayer
#from super_layer import GraphConvolutionLayer, GraphAttentionLayer

class GCN(nn.Module):
    def __init__(self, F, H, C, dropout):
        super(GCN, self).__init__()
        self.layer1 = GraphConvolutionLayer(F, H)
        self.layer2 = GraphConvolutionLayer(H, C)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, adj):
        # X : a tensor with size [N, F]
        
        x = self.dropout(F.relu(self.layer1(x, adj))) # [N, H]
        return self.layer2(x, adj) # [N, C]

    def get_embeds(self, x, adj):
        x = self.dropout(F.relu(self.layer1(x, adj)))
        return x
    
class GAT(nn.Module):
    def __init__(self, F, H, C, N, dropout, alpha, K):
        super(GAT, self).__init__()
        self.layer1 = GraphAttentionLayer(F, H, K, alpha)
        self.layer2 = GraphAttentionLayer(K * H, C, 1, alpha, concat = False)
        #self.layer2 = GraphAttentionLayer(H, C, 1, alpha, concat = False)
        self.dropout = nn.Dropout(p = dropout)
        self.embed = None

    def forward(self, x, adj):
        # x : a tensor with size [N, F]

        x = self.dropout(F.relu(self.layer1(x, adj))) # [N, KH]
        self.embed = x
        return self.layer2(x, adj) # [N, C]

    def get_embeds(self, x, adj):
        return self.embed
        

    def get_att_scores(self, x, adj):
        att1 = self.layer1.att_score(x)
        att2 = self.layer2.att_score(self.embed)
        return att1, att2


class Autoencoder(nn.Module):
    def __init__(self, F, H, D):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(F, H*4),
            nn.ReLU(),
            nn.Linear(H*4,H*2),
            nn.ReLU(),
            nn.Linear(H*2,H),
            nn.ReLU(),
            nn.Linear(H,D),
        )
        self.decoder = nn.Sequential(
            nn.Linear(D,H),
            nn.ReLU(),
            nn.Linear(H,H*2),
            nn.ReLU(),
            nn.Linear(H*2,H*4),
            nn.ReLU(),
            nn.Linear(H*4, F),
            nn.Sigmoid(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded