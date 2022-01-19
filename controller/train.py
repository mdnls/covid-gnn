from models.graphs import StateGraph, GNNRanker
from models.nets import MLPEncoder, MLPDecoder, SpikeEncoder, GraphConvLayer
from controller.initializers import init_nets, init_optimizer
import numpy as np
import torch



from rankers import Ranker
from data.openABM.loop_abm import loop_abm
from data.openABM import abm_utils



import scipy.sparse
from tqdm import tqdm, trange

def batch(items, items_per_batch):
    # given items=[1, 2, 3, 4] and items_per_batch=2, returns [(1, 2), (2, 3), (3, 4)]
    return [torch.stack([items[i-j-1] for j in range(items_per_batch)], dim=0) for i in range(items_per_batch, len(items))]

def cur_test_proba(cur_obs):
    # given cur_obs in the form of a [Tk, N] matrix of -1, 0, +1 observation values,
    # extract the index Tk=-1, return a list of indices of nonzero elements, and return a [N, 2] list of observation outcome probabilities
    Tk, N = cur_obs.size()
    obs = torch.zeros((N, 2))
    obs[cur_obs[-1, :] == 1, 0] = 1
    obs[cur_obs[-1, :] == -1, 1] = 1
    idxs = torch.nonzero(cur_obs[-1])
    return obs, idxs

def train(cnf, observations, contacts):
    Tk = cnf.setting.time_window
    N = cnf.setting.n_vertices
    dh = cnf.model.encode.state_dim

    assert len(observations) == len(contacts), "Observations and contacts shape differ in first dimension (time index). They should cover the same amount of time."
    assert Tk < len(observations), "Observations and contacts must extend over more timesteps than Tk, the simulation window."
    Enc, Prop, Dec, Pred = init_nets(cnf)
    ranker = GNNRanker(Enc, Prop, Dec, Pred)

    params = list(Enc.parameters()) + list(Prop.parameters()) + list(Dec.parameters()) + list(Pred.parameters())
    opt = init_optimizer(params, cnf)

    # randomly generate initial state vectors


    obs_batches = batch(observations, Tk)
    contact_batches = batch(contacts, Tk)

    init_contacts = contact_batches[0]
    init_node_data = torch.FloatTensor(np.random.normal(size=(Tk, N, dh)))

    state = StateGraph(n_vertices=N, t_time=Tk, node_data=init_node_data, edge_weights=init_contacts)
    loss = torch.nn.BCELoss()
    for epoch in range(cnf.training.epochs):
        sum_loss = 0
        for iter in trange(len(obs_batches) - 1):
            current_obs = obs_batches[iter]
            new_obs = obs_batches[iter+1]
            new_contacts = contact_batches[iter+1]

            new_state_vecs = ranker.state_update(state, current_obs)
            state = state.increment(new_state_vecs, new_contacts)

            new_pred = ranker.predict(new_state_vecs) # Each row is 3-dim probabilities of S, I, R
            new_pred_as_binary = (new_pred @ torch.FloatTensor([[1, 0], [0, 1], [1, 0]])).reshape((N, 2)) # Convert S, I, R to +, -

            new_obs_proba, observed_indices = cur_test_proba(new_obs)
            opt.zero_grad()
            L = loss(new_pred_as_binary[observed_indices], new_obs_proba[observed_indices])
            L.backward(retain_graph=True)
            opt.step()

            sum_loss += L.item()
            print(f"\rLoss: {round(L.item(), 5)}", end="")
        avg_loss = sum_loss / (len(obs_batches) - 1)
        print(f"=== EPOCH {epoch} ===")
        print(f"Average loss: {round(avg_loss, 5)}")

#def simulate_without_intervention(self, cnf):
#    pass

def simulate(cnf):
    '''
    For each time step t=1...T
        use openabm to simulate one step of epidemic
        call a generic method predict(state_graph) to have P in R[N by 3], probabilities of each person's state
        given these probabilities, rank people and decide who to test
        ---> next openabm step, *with* tests computed at this step which decide who to quarantine 
  
    '''
    params_model = {
        "rng_seed" : 0, #cnf.setting.seed,
        "end_time" : cnf.setting.time_total,
        "n_total"  : cnf.setting.n_vertices,
        "days_of_interactions" : cnf.setting.time_total,
        "n_seed_infection" : cnf.openabm.n_seed_infection,
    }

    data={}

    loop_abm(
        params = params_model,
        inference_algo = Ranker.DotdRanker(),
        #logger = logging.getLogger(f"iteration"),
        data = data,
        #callback = callback,
        initial_steps = cnf.openabm.initial_steps,
        num_test_random = cnf.openabm.num_test_random,
        num_test_algo = cnf.openabm.num_test_algo,
        fraction_SM_obs = cnf.openabm.fraction_SM_obs,
        fraction_SS_obs = cnf.openabm.fraction_SS_obs,
        quarantine_HH = cnf.openabm.quarantine_HH,
        test_HH = cnf.openabm.test_HH,
        adoption_fraction = cnf.openabm.adoption_fraction,
        fp_rate = cnf.openabm.fp_rate,
        fn_rate = cnf.openabm.fn_rate,
        name_file_res = f"res_N_{cnf.setting.n_vertices}_T_{cnf.setting.time_total}_obs_{cnf.openabm.num_test_algo}_SM_obs_{cnf.openabm.fraction_SM_obs}"
    )

    contact_indices = [ np.transpose(np.array([ ot[o][:2] for o in range(len(ot)) ])) for ot in data["contacts"] ]
    
    return data["observations"], contact_indices


def test_init(cnf):
    to_th = lambda x: torch.FloatTensor(x)

    T = cnf.setting.time_total
    Tk = cnf.setting.time_window
    N = cnf.setting.n_vertices
    dh = cnf.model.propagate.state_dim

    '''
    encoder = SpikeEncoder(n_vertices=N,
                           t_timesteps=Tk,
                           state_dim=dh)
    propagator = GraphConvLayer(state_dim=dh)
    decoder = MLPDecoder(n_vertices=N,
                         t_timesteps=Tk,
                         layers=[Tk*dh, dh, dh],
                         state_dim=dh)
    predictor = SoftmaxReluClassifier(layers=[dh, 3], batchnorm=False)
    '''

    fake_node_data = torch.FloatTensor(np.random.normal(size=(Tk, N, dh)))

    n_random_contacts = int(0.05 * N*N)
    random_contact_indices =[torch.vstack( (to_th(np.random.choice(N, size=(n_random_contacts,))),
                                            to_th(np.random.choice(N, size=(n_random_contacts,)))))
                             for _ in range(T)]
    random_contact_duration = [np.random.exponential(size=(n_random_contacts,)) for _ in range(T)]
    '''
    fake_state_graph = StateGraph(N, Tk, fake_node_data,
                                  torch.sparse_coo_tensor(random_contact_indices, random_contact_duration, size=(Tk, N, N), dtype=torch.float32))

    local_state = fake_state_graph.local_state(1, 1)
    '''
    random_contacts = [torch.sparse_coo_tensor(random_contact_indices[i], random_contact_duration[i], size=(N, N), dtype=torch.float32) for i in range(T)]
    n_random_observations = int(0.05 * T * N)
    fake_observations = np.zeros((T, N))
    fake_observations[np.random.choice(T, size=(n_random_observations)), np.random.choice(N, size=(n_random_observations,))] = 1
    fake_observations[np.random.choice(T, size=(n_random_observations)), np.random.choice(N, size=(n_random_observations,))] = -1
    fake_observations = torch.FloatTensor(fake_observations)

    train(cnf, fake_observations, random_contacts)
    print("F")

def test_init_openabm(cnf):
    to_th = lambda x: torch.FloatTensor(x)

    T = cnf.setting.time_total
    #Tk = cnf.setting.time_window
    N = cnf.setting.n_vertices
    #dh = cnf.model.propagate.state_dim

    obs_matrix, contact_indices_matrix = simulate(cnf)

    contact_indices =[torch.vstack( (to_th(contact_indices_matrix[i][0]),
                                            to_th(contact_indices_matrix[i][1])))
                             for i in range(T)]
    n_contacts = [len(c[0]) for c in contact_indices_matrix ]
    contact_duration = [ np.random.exponential(size=(n_contacts[i],)) for i in range(T)]
    contacts = [torch.sparse_coo_tensor(contact_indices[i], contact_duration[i], size=(N, N), dtype=torch.float32) for i in range(T)]
    observations = torch.FloatTensor(obs_matrix)
    print("I")
    train(cnf, observations, contacts)

    print("F")



