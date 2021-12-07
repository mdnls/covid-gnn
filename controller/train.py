from models.simple import ClassConditionalBias, SoftmaxReluClassifier
from models.graph import StateGraph, MLPDecoder, MLPEncoder, SpikeEncoder
from models.gnn import GraphConvLayer
from models.gnn_ranker import GNNRanker
import numpy as np
import torch


def online_mle(self, cnf):
    to_th = lambda x: torch.FloatTensor(x)

    Tk = cnf.setting.time_window
    N = cnf.setting.n_vertices
    dh = cnf.model.propagate.properties.state_dim
    encoder = SpikeEncoder(n_vertices=N,
                           t_timesteps=Tk,
                           state_dim=dh)
    propagator = GraphConvLayer(state_dim=dh)
    decoder = MLPDecoder(n_vertices=N,
                         t_timesteps=Tk,
                         layers=[Tk*dh, dh, dh],
                         state_dim=dh)
    predictor = SoftmaxReluClassifier(layers=[dh, 3], batchnorm=False)
    ranker = GNNRanker(encoder=encoder, propagator=propagator, decoder=decoder, predictor=predictor)

    fake_node_data = torch.FloatTensor(np.random.normal(size=(Tk, N, dh)))
    n_random_contacts = int(0.05 * Tk*N*N)
    random_contact_indices = torch.stack([to_th(np.random.choice(Tk, size=(n_random_contacts,))),
                              to_th(np.random.choice(N, size=(n_random_contacts,))),
                              to_th(np.random.choice(N, size=(n_random_contacts)))])
    random_contact_duration = np.random.exponential(size=(n_random_contacts))
    fake_state_graph = StateGraph(N, Tk, fake_node_data, torch.sparse_coo_tensor(random_contact_indices, random_contact_duration, size=(Tk, N, N), dtype=torch.float32))
    local_state = fake_state_graph.local_state(1, 1)

    fake_observations = np.zeros((Tk, N))
    fake_observations[0, 0] = 1
    fake_observations[1, 1] = -1
    F = ranker.state_update(fake_state_graph, fake_observations)
    print(F)




