import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, ELU, LayerNorm
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dropout_edge

class GCN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCN, self).__init__()
        self.gn0 = GraphNorm(num_features)
        self.conv1 = GCNConv(num_features, num_hidden)
        self.gn1 = GraphNorm(num_hidden)
        self.conv2 = GCNConv(num_hidden, 64)
        self.gn2 = GraphNorm(num_hidden)
        self.conv3 = GCNConv(num_hidden, 64)
        self.dense = Linear(num_hidden, 64)
        self.output = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch=None):
        # Convolution layers
        x = self.gn0(x, batch)
        x = F.relu(self.conv1(x, edge_index))
        x = self.gn1(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = self.gn2(x, batch)
        x = F.relu(self.conv3(x, edge_index))

        # readout layers
        x = global_mean_pool(x, batch)

        # dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1)


class GNN(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GNN, self).__init__()
        self.gn0 = GraphNorm(num_features)
        self.conv1 = GraphConv(num_features, num_hidden)
        self.gn1 = GraphNorm(num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.gn2 = GraphNorm(num_hidden)
        self.conv3 = GraphConv(num_hidden, num_hidden)
        self.dense = Linear(num_hidden, num_hidden)
        self.output = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch=None):
        # Convolution layers
        x = self.gn0(x, batch)
        x = F.relu(self.conv1(x, edge_index))
        x = self.gn1(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        x = self.gn2(x, batch)
        x = F.relu(self.conv3(x, edge_index))

        # readout layers
        x = global_mean_pool(x, batch)

        # dense layers
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.dense(x))
        x = self.output(x)

        return F.softmax(x, dim=1)


class GAT(nn.Module):
    def __init__(self, num_features, num_graph_features, num_hidden, num_classes, dropout_p):
        super(GAT, self).__init__()
        self.gn0 = GraphNorm(num_features)
        self.bn0 = BatchNorm1d(num_hidden+num_graph_features)
        self.conv1 = GATConv(in_channels=num_features, out_channels=num_hidden, heads=4, dropout=droupout_p)
        self.conv2 = GATConv(in_channels=num_hidden*4, out_channels=num_hidden, heads=4, dropout=dropout_p)
        self.conv3 = GATConv(in_channels=num_hidden*4, out_channels=num_hidden, dropout=droupout_p)
        self.dense1 = Linear(num_hidden+num_graph_features, num_hidden)
        self.dense2 = Linear(num_hidden, num_hidden)
        self.output = Linear(num_hidden, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index, graph_input,  batch=None):
        x = self.gn0(x, batch)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.conv3(x, edge_index)

        # readout layers
        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # dense layers
        x = F.alpha_dropout(x, p=self.dropout_p)
        x = F.selu(self.dense1(x))
        x = F.alpha_dropout(x, p=self.dropout_p)
        x = F.selu(self.dense2(x))
        x = self.output(x)

        return F.softmax(x, dim=1)


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
        self.layer_norm = LayerNorm(out_channels)
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = super().forward(x, edge_index, batch=batch)
        out += self.shortcut(x)
        out = self.layer_norm(out)
        return out


class TransformerEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels, dropout_p, heads=4, k=4):
        super().__init__(aggr="mean")
        self.conv = TransformerConv(in_channels, out_channels // heads, heads=heads, dropout=dropout_p)
        self.skip = Linear(in_channels, out_channels)
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = self.conv(x, edge_index)
        out += self.skip(x)
        return out


class GATEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, heads=4, k=4):
        super(GATEdgeConv, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout_p)
        self.skip = Linear(in_channels, out_channels)
        self.layer_norm = LayerNorm(out_channels)
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = self.gat_conv(x, edge_index)
        out += self.skip(x)
        out = self.layer_norm(out)
        return out



class ParticleNet(torch.nn.Module):
    def __init__(self, num_node_features, num_graph_features, num_classes, num_hidden, dropout_p):
        super(ParticleNet, self).__init__()
        self.gn0 = GraphNorm(num_node_features)
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        #self.linear = Linear(3*num_hidden, num_hidden)
        
        self.bn0 = BatchNorm1d(num_hidden*3+num_graph_features)
        self.dense1 = Linear(num_hidden*3+num_graph_features, num_hidden)
        self.bn1 = BatchNorm1d(num_hidden)
        self.dense2 = Linear(num_hidden, num_hidden)
        self.bn2 = BatchNorm1d(num_hidden)
        self.output = Linear(num_hidden, num_classes)
        self.dropout_p = dropout_p

    def forward(self, x, edge_index, graph_input, batch=None):
        # Convolution layers
        x = self.gn0(x, batch=batch)
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        #x = self.linear(torch.cat([conv1, conv2, conv3], dim=1))
        
        ## Use Attention Mechanism for concatenation
        #weights = F.softmax(self.attention_weights, dim=0)
        #x = weights[0]*conv1 + weights[1]*conv2 + weights[2]*conv3
        x = torch.cat([conv1, conv2, conv3], dim=1)

        # readout layers
        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # dense layers
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)

        return F.softmax(x, dim=1)
