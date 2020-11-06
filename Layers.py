import torch
from torch_geometric.nn.conv import GCNConv
import torch.nn.functional as F
import numpy as np
import scipy
import torch.nn as nn
from torch.autograd import Variable


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for _ in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)

class Basic_GCN_Layers(torch.nn.Module):
    def __init__(self, gcn_filters, node_feature_num):
        super(Basic_GCN_Layers, self).__init__()
        self.layer_list = [GCNConv(node_feature_num, gcn_filters[0]).cuda()]
        for i in range(len(gcn_filters)-1):
            self.layer_list.append(GCNConv(gcn_filters[i], gcn_filters[i+1]))
        self.layer_list = ListModule(*self.layer_list)

    def forward(self, features, edges):
        features = features.cuda()
        edges = edges.cuda()
        for layer in self.layer_list:
            features = torch.nn.functional.relu(layer(features, edges))
        return features