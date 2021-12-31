import torch
import numpy as np
import torch.nn as nn
from models.graphs import StateGraph

class GraphConvLayer(torch.nn.Module):
    def __init__(self, state_dim, layer_depth=1, n_layers=1):
        super(GraphConvLayer, self).__init__()
        self.state_dim = state_dim
        self.fcn = ReluFCN(layers=[3 * state_dim] + layer_depth * [state_dim], batchnorm=True)
        self.linear_output = torch.nn.Linear(state_dim, state_dim)
        self.n_layers = n_layers

    def local_update(self, present_state, past_state, neighborhood_states, neighborhood_edge_weights):
        '''
        Update the state vector of a single node.

        :param present_state: state vector node i at time t
        :param past_state: state vector node i at time t-1
        :param neighborhood_states: a matrix [ # of neighbors, state_dim ] containing state vectors for each neighbor
        :param neighborhood_edge_weights: a vector [ # of neighbors, ] containing edge weights
        :return: updated state vector of v_i
        '''
        partition = torch.sum(neighborhood_edge_weights)
        local_avg = torch.matmul(neighborhood_edge_weights.view([1, -1])/partition, neighborhood_states)
        inp = torch.cat([present_state, past_state, local_avg], dim=0)
        return self.linear_output(self.fcn(inp))

    def forward(self, state_graph):
        '''
        Given a state graph at time t, compute states for all vertices at time t+1 and incremement the state graph.
        :param state_graph:
        :return:
        '''

        T, N, dh = state_graph.t_time, state_graph.n_vertices, state_graph.state_dim
        ew = state_graph.edge_weights()
        sum_neighbor_state = torch.bmm(ew, state_graph.node_data())
        Z = torch.sparse.sum(ew, dim=2).to_dense().view((T, N, 1))
        Z[Z==0] = 1
        avg_neighbor_state = sum_neighbor_state / Z

        prev_state = torch.cat([torch.zeros((1, N, dh)), state_graph.node_data()[1:]], dim=0)

        gnn_inputs = torch.cat([state_graph.node_data(), prev_state, avg_neighbor_state], dim=2).reshape((T * N, 3*dh))
        new_state = self.fcn(gnn_inputs).reshape((T, N, dh))
        return StateGraph(N, T, new_state, state_graph.edge_weights())

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
        return self.decoder(node_data).reshape((1, N, dh))


# This guy will do classification of h into [s, i, r] states, it is the P(h) function.
class SoftmaxReluClassifier(torch.nn.Module):
    def __init__(self, layers, batchnorm=False):
        '''
        A fully connected ReLU network followed by softmax classification.

        - Input is d_in dimensional vector
        - For l=1...k, compute layer l activation z_l as
            z_l = ReLU(W_l z_{l-1} + b_l)
        - Layer dimensions are specified by `layers` argument, W_l has dimension (d_l x d_{l-1}) and b_l has dimension dl.
        - The output is y_k = Softmax(z_k)

        :param layers: integers [d0=d_in, d1, d2, ..., dk=dout]
        :param batchnorm: if true, include batchnorm, ie. layers are z_l = ReLU(BatchNorm(Wz + b))
        '''
        super().__init__()

        self.linears = torch.nn.ModuleList([nn.Linear(layers[l], layers[l + 1]) for l in range(len(layers) - 1)])
        if (batchnorm):
            self.batchnorm_layers = torch.nn.ModuleList([nn.BatchNorm1d(num_features=l) for l in layers])
        self.batchnorm = batchnorm
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        Map input x to output y_l.

        :param x: batch of inputs with shape [N, d_in]
        :return: batch of outputs with shape [N, y]
        '''

        for i in range(len(self.linears)):
            if (self.batchnorm):
                x = self.batchnorm_layers[i](x)
            x = self.relu(self.linears[i](x))

        return self.out(x)


class ReluFCN(torch.nn.Module):
    def __init__(self, layers, batchnorm=False):
        '''
        A fully connected ReLU network.

        - Input is d_in dimensional vector
        - For l=1...k, compute layer l activation z_l as
            z_l = ReLU(W_l z_{l-1} + b_l)
        - Layer dimensions are specified by `layers` argument, W_l has dimension (d_l x d_{l-1}) and b_l has dimension dl.

        :param layers: integers [d0=d_in, d1, d2, ..., dk=dout]
        :param batchnorm: if true, include batchnorm, ie. layers are z_l = ReLU(BatchNorm(Wz + b))
        '''
        super().__init__()
        self.linears = torch.nn.ModuleList([nn.Linear(layers[l], layers[l + 1]) for l in range(len(layers) - 1)])
        if (batchnorm):
            self.batchnorm_layers = torch.nn.ModuleList([nn.BatchNorm1d(num_features=l) for l in layers])
        self.batchnorm = batchnorm
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        '''
        Map input x to output y_l.

        :param x: batch of inputs with shape [N, d_in]
        :return: batch of outputs with shape [N, y]
        '''

        for i in range(len(self.linears)):
            if (self.batchnorm):
                x = self.batchnorm_layers[i](x)
            x = self.relu(self.linears[i](x))
        return x


# Example of embedding function V(h)
class ClassConditionalBias(torch.nn.Module):
    def __init__(self, n_classes, dim):
        '''
        Given input x and a discrete class z, add a class-specific vector to x and output y = x + b_z

        :param n_classes: number of classes. Each class is indexed by an integer 0 <= z < n_classes.
        :param dim: dimension of x
        '''

        self.n_classes = len(self.classes)
        self.biases = torch.nn.Parameter(torch.zeros(n_classes, dim))

    def forward(self, x, classes):
        '''
        Apply class conditional bias to x.
        :param x: tensor with dimension [N, d_in]
        :param classes: tensor with dimension [N,] containing integerz 0 <= z < n_classes.
        :return: y = x+bz
        '''
        return x + torch.take(self.biases, classes)

