import torch
from models.simple import ReluFCN
from models.graph import StateGraph

# This will do the propagation step, details TBD
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

