from models.graphs import StateGraph, GNNRanker
from models.nets import MLPEncoder, MLPDecoder, SpikeEncoder, GraphConvLayer
from controller.initializers import init_nets, init_optimizer
import numpy as np
import torch

#from rankers import Ranker
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
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(cnf.training.epochs):
        sum_loss = 0
        for iter in trange(len(obs_batches) - 1):
            current_obs = obs_batches[iter]
            new_obs = obs_batches[iter+1]
            new_contacts = contact_batches[iter+1]

            new_state_vecs = ranker.state_update(state, current_obs)
            state = state.increment(new_state_vecs, new_contacts).detach()

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

def simulate_without_intervention(self, cnf):
    pass

'''
def simulate(self, cnf):
    For each time step t=1...T
        use openabm to simulate one step of epidemic
        call a generic method predict(state_graph) to have P in R[N by 3], probabilities of each person's state
        given these probabilities, rank people and decide who to test
        ---> next openabm step, *with* tests computed at this step which decide who to quarantine 

    import loop_abm, abm_utils

    params_model = {
        "rng_seed" : cnf.setting.seed,
        "end_time" : cnf.setting.time_total,
        "n_total"  : cnf.setting.n_vertices,
        "days_of_interactions" : cnf.setting.time_total,
        "n_seed_infection" : cnf.data.openabm.n_seed_infection,
    }

    import imp
    imp.reload(loop_abm)

    import predict

    import logging

    data={}

    loop_abm.loop_abm(
        params_model,
        Ranker,
        logger = logging.getLogger(f"iteration"),
        data = data,
        #callback = callback,
        initial_steps = cnf.data.openabm.initial_steps,
        num_test_random = cnf.data.openabm.num_test_random,
        num_test_algo = cnf.data.openabm.num_test_algo,
        fraction_SM_obs = cnf.data.openabm.fraction_SM_obs,
        fraction_SS_obs = cnf.data.openabm.fraction_SS_obs,
        quarantine_HH = cnf.data.openabm.quarantine_HH,
        test_HH = cnf.data.openabm.test_HH,
        adoption_fraction = cnf.data.openabm.adoption_fraction,
        fp_rate = cnf.data.openabm.fp_rate,
        fn_rate = cnf.data.openabm.fn_rate,
        name_file_res = s + f"_N_{cnf.setting.n_vertices}_T_{cnf.setting.time_total}_obs_{cnf.data.openabm.num_test_algo}_SM_obs_{cnf.data.openabm.fraction_SM_obs}_seed_{cnf.setting.seed}"
    )


def loop_abm(params,
             inference_algo,
             logger = dummy_logger(),
             input_parameter_file = "./abm_params/baseline_parameters.csv",
             household_demographics_file = "./abm_params/baseline_household_demographics.csv",
             parameter_line_number = 1,
             initial_steps = 10,
             num_test_random = 0,
             num_test_algo = 200,
             fraction_SM_obs = 0.5,
             fraction_SS_obs = 1,
             quarantine_HH = True,             
             test_HH = True,
             name_file_res = "res",
             output_dir = "./output/",
             save_every_iter = 5,
             stop_zero_I = True,
             adoption_fraction = 1.0,
             fp_rate = 0.0,
             fn_rate = 0.0,
             smartphone_users_abm = False, # if True use app users fraction from OpenABM model
             callback = lambda x : None,
             data = {}
            ):
    Simulate interventions strategy on the openABM epidemic simulation.

    input
    -----
    params: Dict
            Dictonary with openABM to set
    inference_algo: Class (rank_template)
            Class for order the nodes according to the prob to be infected
            logger = logger for printing intermediate steps
    results:
        print on file true configurations and transmission

    import covid19
    from COVID19.model import AgeGroupEnum, EVENT_TYPES, TransmissionTypeEnum
    from COVID19.model import Model, Parameters, ModelParameterException
    import COVID19.simulation as simulation


    params_model = Parameters(input_parameter_file,
                          parameter_line_number,
                          output_dir,
                          household_demographics_file)
    
    ### create output_dir if missing
    fold_out = Path(output_dir)
    if not fold_out.exists():
        fold_out.mkdir(parents=True)

    ### initialize a separate random stream
    rng = np.random.RandomState()
    rng.seed(cnf.setting.seed)
    
    ### initialize ABM model
    for k, val in params.items():
        params_model.set_param(k, val)
    model = Model(params_model)
    model = simulation.COVID19IBM(model=model)

    
    T = params_model.get_param("end_time")
    N = params_model.get_param("n_total")
    sim = simulation.Simulation(env=model, end_time=T, verbose=False)
    house = covid19.get_house(model.model.c_model)
    housedict = listofhouses(house)
    has_app = covid19.get_app_users(model.model.c_model) if smartphone_users_abm else np.ones(N,dtype = int)
    has_app &= (rng.random(N) <= adoption_fraction)    

    ### init data and data_states
    data_states = {}
    data_states["true_conf"] = np.zeros((T,N))
    data_states["statuses"] = np.zeros((T,N))
    data_states["tested_algo"] = []
    data_states["tested_random"] = []
    data_states["tested_SS"] = []
    data_states["tested_SM"] = []
    for name in ["num_quarantined", "q_SS", "q_SM", "q_algo", "q_random", "q_all", "infected_free", "S", "I", "R", "IR", "aurI", "prec1%", "prec5%", "test_+", "test_-", "test_f+", "test_f-"]:
        data[name] = np.full(T,np.nan)
    data["logger"] = logger

    
    ### init inference algo
    #nference_algo.init(N, T)
    
    ### running variables
    indices = np.arange(N, dtype=int)
    excluded = np.zeros(N, dtype=bool)
    daily_obs = []
    all_obs = []
    all_quarantined = []
    freebirds = 0
    num_quarantined = 0
    fp_num = 0
    fn_num = 0
    p_num = 0
    n_num = 0
    
    noise_SM = rng.random(N)
    nfree = params_model.get_param("n_seed_infection")
    for t in range(T):
        ### advance one time step
        sim.steps(1)
        status = np.array(covid19.get_state(model.model.c_model))
        state = status_to_state(status)
        data_states["true_conf"][t] = state
        nS, nI, nR = (state == 0).sum(), (state == 1).sum(), (state == 2).sum()
        if nI == 0 and stop_zero_I:
            logger.info("stopping simulation as there are no more infected individuals")
            break
        if t == initial_steps:
            logger.info("\nobservation-based inference algorithm starts now\n")
        logger.info(f'time:{t}')

        ### extract contacts
        daily_contacts = covid19.get_contacts_daily(model.model.c_model, t)
        logger.info(f"number of unique contacts: {len(daily_contacts)}")

        
        
        ### compute potential test results for all
        if fp_rate or fn_rate:
            noise = rng.random(N)
            f_state = (state==1)*(noise > fn_rate) + (state==0)*(noise < fp_rate) + 2*(state==2)
        else:
            f_state = state
        
        to_quarantine = []
        all_test = []
        excluded_now = excluded.copy()
        fp_num_today = 0
        fn_num_today = 0
        p_num_today = 0
        n_num_today = 0
        def test_and_quarantine(rank, num):
            nonlocal to_quarantine, excluded_now, all_test, fp_num_today, fn_num_today, p_num_today, n_num_today
            test_rank = []
            for i in rank:
                if len(test_rank) == num:
                    break;
                if excluded_now[i]:
                    continue
                test_rank += [i]
                if f_state[i] == 1:
                    p_num_today += 1
                    if state[i] != 1:
                        fp_num_today += 1
                    q = housedict[house[i]] if quarantine_HH else [i]
                    excluded_now[q] = True
                    to_quarantine += q
                    excluded[q] = True
                    if test_HH:
                        all_test += q
                    else:
                        all_test += [i]
                else:
                    n_num_today += 1
                    if state[i] == 1:
                        fn_num_today += 1
                    excluded_now[i] = True
                    all_test += [i]
            return test_rank
        
        ### compute rank from algorithm
        num_test_algo_today = num_test_algo
        if t < initial_steps:
            daily_obs = []
            num_test_algo_today = 0            
        
        weighted_contacts = [(c[0], c[1], c[2], 2.0 if c[3] == 0 else 1.0) for c in daily_contacts if (has_app[c[0]] and has_app[c[1]])]
        if nfree == 0 and quarantine_HH:
            print("faster end")
            rank_algo = np.zeros((N,2))
            rank_algo[:, 0]=np.arange(N)
            rank_algo[:, 1]=np.random.rand(N)
        else:
            rank_algo = inference_algo.rank(t, weighted_contacts, daily_obs, data) 
        rank = np.array(sorted(rank_algo, key= lambda tup: tup[1], reverse=True))
        rank = [int(tup[0]) for tup in rank]
        
        ### test num_test_algo_today individuals
        test_algo = test_and_quarantine(rank, num_test_algo_today)

        ### compute roc now, only excluding past tests
        #eventsI = events_list(t, [(i,1,t) for (i,tf) in enumerate(excluded) if tf], data_states["true_conf"], check_fn = check_fn_I)
        #xI, yI, aurI, sortlI = roc_curve(dict(rank_algo), eventsI, lambda x: x)
        
        ### test all SS
        SS = test_and_quarantine(indices[status == 4], N)
        
        ### test a fraction of SM
        SM = indices[(status == 5) & (noise_SM < fraction_SM_obs)]
        SM = test_and_quarantine(SM, len(SM))

        ### do num_test_random extra random tests
        test_random = test_and_quarantine(rng.permutation(N), num_test_random)

        ### quarantine infected individuals
        num_quarantined += len(to_quarantine)
        covid19.intervention_quarantine_list(model.model.c_model, to_quarantine, T+1)
            
        ### update observations
        daily_obs = [(int(i), int(f_state[i]), int(t)) for i in all_test]
        all_obs += daily_obs

        ### exclude forever nodes that are observed recovered
        rec = [i[0] for i in daily_obs if f_state[i[0]] == 2]
        excluded[rec] = True

        ### update data 
        data_states["tested_algo"].append(test_algo)
        data_states["tested_random"].append(test_random)
        data_states["tested_SS"].append(SS)
        data_states["tested_SM"].append(SM)
        data_states["statuses"][t] = status
        data["S"][t] = nS
        data["I"][t] = nI
        data["R"][t] = nR
        data["IR"][t] = nR+nI
        data["aurI"][t] = aurI
        prec = lambda f: yI[int(f/100*len(yI))]/int(f/100*len(yI)) if len(yI) else np.nan
        ninfq = sum(state[to_quarantine]>0)
        nfree = int(nI - sum(excluded[state == 1]))
        data["aurI"][t] = aurI
        data["prec1%"][t] = prec(1)
        data["prec5%"][t] = prec(5)
        data["num_quarantined"][t] = num_quarantined
        data["test_+"][t] = p_num
        data["test_-"][t] = n_num
        data["test_f+"][t] = fp_num
        data["test_f-"][t] = fn_num
        data["q_SS"][t] = len(SS)
        data["q_SM"][t] = len(SM)
        sus_test_algo = sum(state[test_algo]==0)
        inf_test_algo = sum(state[test_algo]==1)
        rec_test_algo = sum(state[test_algo]==2)
        inf_test_random = sum(state[test_random]==1)
        data["q_algo"][t] = inf_test_algo
        data["q_random"][t] = sum(state[test_random]==1)
        data["infected_free"][t] = nfree
        asbirds = 'a bird' if nfree == 1 else 'birds'

        fp_num += fp_num_today
        fn_num += fn_num_today
        n_num += n_num_today
        p_num += p_num_today
        
        ### show output
        logger.info(f"True  : (S,I,R): ({nS:.1f}, {nI:.1f}, {nR:.1f})")
        logger.info(f"AUR_I : {aurI:.3f}, prec(1% of {len(yI)}): {prec(1):.2f}, prec5%: {prec(5):.2f}")
        logger.info(f"SS: {len(SS)}, SM: {len(SM)}, results test algo (S,I,R): ({sus_test_algo},{inf_test_algo},{rec_test_algo}), infected test random: {inf_test_random}/{num_test_random}")
        logger.info(f"false+: {fp_num} (+{fp_num_today}), false-: {fn_num} (+{fn_num_today})")
        logger.info(f"...quarantining {len(to_quarantine)} guys -> got {ninfq} infected, {nfree} free as {asbirds} ({nfree-freebirds:+d})")
        freebirds = nfree

        ### callback
        #callback(data)

        if t % save_every_iter == 0:
            df_save = pd.DataFrame.from_records(data, exclude=["logger"])
            df_save.to_csv(output_dir + name_file_res + "_res.gz")

    # save files
    df_save = pd.DataFrame.from_records(data, exclude=["logger"])
    df_save.to_csv(output_dir + name_file_res + "_res.gz")
    with open(output_dir + name_file_res + "_states.pkl", mode="wb") as f_states:
        pickle.dump(data_states, f_states)
    sim.env.model.write_individual_file()
    df_indiv = pd.read_csv(output_dir+"individual_file_Run1.csv", skipinitialspace = True)
    df_indiv.to_csv(output_dir+name_file_res+"_individuals.gz")
    sim.env.model.write_transmissions()
    df_trans = pd.read_csv(output_dir+"transmission_Run1.csv")
    df_trans.to_csv(output_dir + name_file_res+"_transmissions.gz")
    return df_save
'''


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




