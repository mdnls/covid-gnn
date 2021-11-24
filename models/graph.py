import numpy as np

class StateGraph():
    def __init__(self, n_vertices, t_time, node_data, edge_weights):
        '''
        Represents the state of a population of `n_vertices` people over `t_time` timesteps.

        :param n_vertices: number of vertices
        :param t_time: number of timesteps
        :param node_data: a matrix of shape (T, N, d_h) of d_h-dimensional state vectors per vertex
        :param edge_weights: a csr sparse matrix C of shape (T, N, N) representing adjacencies in the contact graph. If
            persons (i, j) are in contact at time t then C_{t, i, j} > 0.
        '''
        self.n_vertices = n_vertices
        self.t_time = t_time
        self.node_data = node_data
        self.edge_weights = edge_weights

    def local_state(self, t, n):
        '''
        Return the local graph of node (t, n), consisting of the node, its neighbors, edge weights, and all nodes' state information
        :param n: spatial index of requested node
        :param t: time index of requested node
        :return: (present_node_data, past_node_data, [node_data for each neighbor], [edge weight for each neighbor])
        '''
        present_node_data = self.node_data[t, n]
        if(t > 0):
            past_node_data = self.node_data[t - 1, n]
            neighbors = self.edge_weights[t - 1, n].nonzero()
            neighboring_data = self.node_data[t - 1, neighbors]
            neighboring_edge_weights = self.edge_weights[t - 1, n, neighbors]
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

        node_data = np.stack([self.node_data[1:], node_data], axis=0)
        edge_weights = np.stack([self.edge_weights[1:], contacts], axis=0)

        return StateGraph(self.n_vertices, self.t_time, node_data, edge_weights)