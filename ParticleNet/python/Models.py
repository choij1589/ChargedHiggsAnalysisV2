import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, ELU, AlphaDroupout
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__(aggr="mean")
        self.mlp = Sequential(
                Linear(2*in_channels, out_channels), ReLU(), BatchNorm1d(out_channels), Dropout(dropout_p),
                Linear(out_channels, out_channels), ReLU(), BatchNorm1d(out_channels), Dropout(dropout_p),
                Linear(out_channels, out_channels), ReLU(), BatchNorm1d(out_channels), Dropout(dropout_p)
        )

    def forward(self, x, edge_index, batch=None):
        return self.propagate(edge_index, x=x, batch=batch)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)

class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, dropout_p, k=4):
        super().__init__(in_channels, out_channels, dropout_p=dropout_p)
        self.shortcut = Sequential(Linear(in_channels, out_channels), BatchNorm1d(out_channels), Dropout(dropout_p))
        self.k = k

    def forward(self, x edge_index=None, batch=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)
        out = super().forward(x, edge_index, batch=batch)
        out += self.shortcut(x)
        return out


class ParticleNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_nodes, dropout_p):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(num_features)
        self.conv1 = DynamicEdgeConv(num_features, num_nodes, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_nodes, num_nodes, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_nodes, num_nodes, dropout_p, k=4)
        self.dense1 = Linear(num_nodes*3, num_nodes)
        self.bn1 = BatchNorm1d(num_nodes)
        self.dense2 = Linear(num_nodes, num_nodes)
        self.bn2 = BatchNorm1d(num_nodes)
        self.output = Linear(num_nodes, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index, batch=None):
        # Convolution layers
        x = self.gn0(x, batch=batch)
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        x = torch.cat([conv1, conv2, conv3], dim=2)

        # readout layers
        x = global_mean_pool(x, batch=batch)

        # dense layers
        x = F.relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)

        return F.softmax(x, dim=1)
