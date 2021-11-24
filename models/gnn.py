import torch


# This guy will do the propagation step, details TBD
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()

    def forward(self, v, neighborhood):
        '''
        Update the state vector of a single node.

        :param v: state vector v_i of a given node i
        :param neighborhood: list of tuples [(e_{ij}, v_j) for neighbors v_j of v]. Each tuple contains a real positive
            valued edge weight e_{ij} and a neighboring state vector v_j.
        :return: updated state vector of v_i
        '''
