import numpy as np
import torch
from torch import nn
from models.simple import ReluFCN

class StateGraph():
    def __init__(self, n_vertices, t_time, node_data, edge_weights):
        '''
        Represents the state of a population of `n_vertices` people over `t_time` timesteps.

        Note: for fixed time t, the ith row corresponds to contacts of vertex i
        :param n_vertices: number of vertices
        :param t_time: number of timesteps (ie. size of window used for classification)
        :param node_data: a matrix of shape (T, N, d_h) of d_h-dimensional state vectors per vertex
        :param edge_weights: a coo sparse tensor C of shape (T, N, N) representing adjacencies in the contact graph. If
            persons (i, j) are in contact at time t then C_{t, i, j} > 0.
        '''
        self.n_vertices = n_vertices
        self.t_time = t_time
        self.state_dim = node_data.size()[-1]
        self._node_data = node_data
        self._edge_weights = edge_weights.coalesce()

    def local_state(self, t, n):
        '''
        Return the local graph of node (t, n), consisting of the node, its neighbors, edge weights, and all nodes' state information
        :param n: spatial index of requested node
        :param t: time index of requested node
        :return: (present_node_data, past_node_data, [node_data for each neighbor], [edge weight for each neighbor])
        '''
        present_node_data = self._node_data[t, n]
        if(t > 0):
            past_node_data = self._node_data[t-1, n]
            neighbors = self._edge_weights[t-1, n].coalesce()
            neighboring_data = self._node_data[t - 1, neighbors.indices().flatten()]
            neighboring_edge_weights = neighbors.values()
        else:
            past_node_data = np.zeros_like(present_node_data)
            neighboring_data = np.array([])
            neighboring_edge_weights = np.array([])
        return (present_node_data, past_node_data, neighboring_data, neighboring_edge_weights)

    def increment(self, node_data, contacts):
        '''
        Suppose this state graph contains data for times [t-k, t]. Given data for time t+1, return
             a new state graph containing data for times [t-k+1, t+1].
        :param node_data:
        :param contacts: a csr sparse matrix of size (N, N). If person i and j are in contact at time t, then C_{ij} = C_{ji} > 0.
            Internally, these contacts are represented by two edges (v_{t+1, i}, v_{t, j}) and vice versa.
        :return:
        '''

        node_data = np.stack([self._node_data[1:], node_data], axis=0)
        edge_weights = np.stack([self._edge_weights[1:], contacts], axis=0)

        return StateGraph(self.n_vertices, self.t_time, node_data, edge_weights)

    def normalized_adjacencies(self):
        row_sums = self._edge_weights.sum(axis=2)
        return self._edge_weights * (1/row_sums).reshape((1, -1, 1))

    def node_data(self):
        return torch.clone(self._node_data)

    def edge_weights(self):
        return torch.clone(self._edge_weights)

class SpikeEncoder(nn.Module):
    def __init__(self, n_vertices, t_timesteps, state_dim):
        super(SpikeEncoder, self).__init__()
        self.n_vertices = n_vertices
        self.t_timesteps = t_timesteps
        self.state_dim = state_dim
        self.pos_test_spike = nn.Parameter(torch.FloatTensor(np.random.normal(size=(state_dim,), scale=1/np.sqrt(state_dim))))
        self.neg_test_spike = nn.Parameter(torch.FloatTensor(np.random.normal(size=(state_dim,), scale=1/np.sqrt(state_dim))))

    def forward(self, state_graph, observations):
        spikes = torch.zeros((self.t_timesteps, self.n_vertices, self.state_dim))
        spikes[observations == -1, ...] += self.neg_test_spike
        spikes[observations == 1, ...] += self.pos_test_spike
        new_node_data = state_graph.node_data() + spikes
        return StateGraph(self.n_vertices, self.t_timesteps, new_node_data, state_graph.edge_weights())

class MLPEncoder(nn.Module):
    def __init__(self, n_vertices, t_timesteps, layers):
        super(MLPEncoder, self).__init__()
        self.n_vertices = n_vertices
        self.t_timesteps = t_timesteps
        self.pos_mlp = ReluFCN(layers=layers)
        self.neg_mlp = ReluFCN(layers=layers)

    def forward(self, state_graph, observations):
        new_node_data = state_graph.node_data()
        new_node_data[observations == -1, ...] = self.neg_mlp(new_node_data[observations == -1, ...])
        new_node_data[observations == 1, ...] = self.pos_mlp(new_node_data[observations == 1, ...])
        return StateGraph(self.n_vertices, self.t_timesteps, new_node_data, state_graph.edge_weights)

class MLPDecoder(nn.Module):
    def __init__(self, n_vertices, t_timesteps, layers, state_dim):
        super(MLPDecoder, self).__init__()
        assert layers[0] == t_timesteps * state_dim, "First layer must have input dimension T*state_dim"
        assert layers[-1] == state_dim, "Last layer must have output dimension state_dim"
        self.n_vertices = n_vertices
        self.t_timesteps = t_timesteps
        self.decoder = ReluFCN(layers=layers)

    def forward(self, state_graph):
        node_data = state_graph.node_data()
        T, N, dh = node_data.size()
        node_data = node_data.transpose(1, 0).reshape((N, T * dh))
        return self.decoder(node_data)
