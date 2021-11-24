import torch
import numpy as np
import networkx as nx

'''
Functionality for the graph 
- Given a node index, return a short list of neighboring nodes 
- Return a sparse adjacency matrix (ie. implicitly an adjacency list 
- can it be accomplished with networkx? 

Design idea:
- Use networkx to track nodes and adjacencies 
- immutability wrt number of nodes 
- Interactions: either (1) given id, return a local graph for the node with the given id or (2) return the global graph 
    in the form of sparse adjacency 
    
TODO: what is a good global ordering for the (t, n) indexed nodes? 
'''

class StateGraph():
    def __init__(self, n_vertices, t_time, node_data, edge_weights):
        '''
        Represents the state of a population of `n_vertices` people over `t_time` timesteps.

        :param n_vertices: number of vertices
        :param t_time: number of timesteps
        :param node_data: a matrix of shape (t, n, d_h) of d_h-dimensional state vectors per vertex
        :param edge_weights: a csr sparse matrix of shape (t*n, t*n)
        '''
        self.n_vertices = n_vertices
        self.t_time = t_time
        self.node_data = node_data
        self.nodes = np.array([(t, n) for t in range(t_time) for n in range(n_vertices)])
        self.node_to_flat = np.arange(t_time * n_vertices).reshape((t_time, n_vertices))
        self.edge_weights = edge_weights

    def local_state(self, t, n):
        '''
        Return the local graph of node (t, n), consisting of the node, its neighbors, edge weights, and all nodes' state information
        :param n: spatial index of requested node
        :param t: time index of requested node
        :return: (node_data, [node_data for each neighbor], [edge weight for each neighbor])
        '''
        idx = self.node_to_flat[t, n]
        node_data = self.node_data[idx]
        neighbors = self.edge_weights[idx].nonzero()
        neighboring_data = self.node_data[neighbors]
        neighboring_edge_weights = self.edge_weights[idx]
        return (node_data, neighboring_data, neighboring_edge_weights)


    def increment(self, node_data, edge_weights):
        '''
        Suppose this state graph incorporates data for times [t-k, t]. Given data for time t+1, return
             a new state graph incorporating data for times [t-k+1, t+1].
        :param node_data:
        :param edge_weights:
        :return:
        '''


