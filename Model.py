import torch
from Layers import Basic_GCN_Layers
import torch.nn as nn
import torch.nn.functional as F
class GCN_Net(torch.nn.Module):
    def __init__(self, gcn_filters, node_feature_num = 2, hidden = 40):
        super(GCN_Net, self).__init__()
        self.GCN = Basic_GCN_Layers(gcn_filters, node_feature_num).cuda()
        self.fc1 = nn.Linear(in_features= 2 * gcn_filters[-1], out_features=hidden).cuda()
        self.fc2_adv = nn.Linear(in_features=hidden, out_features = 2).cuda()
        self.relu = nn.ReLU()
    def forward(self, feature_torch ,edge_torch, i, j):
        output = self.GCN(feature_torch, edge_torch)
        embd = torch.cat((output[i], output[j]))
        result = self.relu(self.fc1(embd))
        result = self.fc2_adv(result)
        return result