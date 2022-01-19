import torch
import numpy as np

class GNNRanker():
    def __init__(self, encoder, propagator, decoder, predictor):
        '''
        :param encoder: graph encoder network, StateGraph -> StateGraph
        :param propagator: GNN based hidden update function,
        :param decoder: neural network mapping to perform SIR 'classification' on each node output by the propagator
        :param predictor: map state vectors to SIR predictions
        '''
        self.encoder = encoder
        self.propagator = propagator
        self.decoder = decoder
        self.predictor = predictor

    def state_update(self, state_graph, observations):
        '''
        Given an input state graph, encode its information into a graph and propagate hidden data, returning the
        new hidden data.

        :param state_graph: a StateGraph
        :param observations: a [T by N] sparse matrix of observations
        :return: an [N, d_h] matrix of new hidden information for each node
        '''
        enc_state_graph = self.encoder(state_graph, observations)
        prp_state_graph = self.propagator(enc_state_graph)
        dec_state_vectors = self.decoder(prp_state_graph)
        return dec_state_vectors

    def predict(self, state_vectors):
        return self.predictor(state_vectors)

    def rank(self, state_vectors):
        '''
        See rank() in rankers/Ranker.py, match the format that openabm expects
        '''
        pass

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

    def detach(self):
        '''
        Detach this state graph from the computation graph, preventing its data from accumulating gradients from existing
            function evaluations.
        :return: a detached copy of this state graph.
        '''
        return StateGraph(self.n_vertices, self.t_time, self._node_data.detach(), self._edge_weights.detach())

    def increment(self, node_data, contacts):
        '''
        Suppose this state graph contains data for times [t-k, t]. Given data for time t+1, return
             a new state graph containing data for times [t-k+1, t+1].
        :param node_data:
        :param contacts: a csr sparse matrix of size (Tk, N, N) which replaces current contacts. If person i and j are in contact at time t, then C_{ij} = C_{ji} > 0.
            Internally, these contacts are represented by two edges (v_{t+1, i}, v_{t, j}) and vice versa.
        :return:
        '''

        node_data = torch.cat([self._node_data[1:], node_data], dim=0)
        return StateGraph(self.n_vertices, self.t_time, node_data, contacts)

    def normalized_adjacencies(self):
        row_sums = self._edge_weights.sum(axis=2)
        return self._edge_weights * (1/row_sums).reshape((1, -1, 1))

    def node_data(self):
        return torch.clone(self._node_data)

    def edge_weights(self):
        return torch.clone(self._edge_weights)
