abm_folder = "../OpenABM-Covid19/src" #Open ABM path

sys.path.insert(0,abm_folder)

import loop_abm, abm_utils




params_model = {
    "rng_seed" : seed,
    "end_time" : T,
    "n_total"  : N,
    "days_of_interactions" : T,
    "n_seed_infection" : n_seed_infection,
}

'''
fraction_SM_obs = 0.5 #fraction of Symptomatic Mild tested positive
fraction_SS_obs = 1 #fraction of Symptomatic Severe tested positive
initial_steps = 12 #starting time of intervention
quarantine_HH = True #Households quarantine
test_HH = True #Tests the households when quarantined
adoption_fraction = 1 #app adoption (fraction)
num_test_random = 0 #number of random tests per day
num_test_algo = 200 #number of tests using by the ranker per day
fp_rate = 0.0 #test false-positive rate
fn_rate = 0.0 #test false-negative rate


prob_seed = 1/N
prob_sus = 0.55
pseed = prob_seed / (2 - prob_seed)
psus = prob_sus * (1 - pseed)
pautoinf = 1/N
'''

import imp
imp.reload(loop_abm)
'''
loop_abm.loop_abm(
    params_model,
    rankers[s],
    seed=new_seed,
    logger = logging.getLogger(f"iteration.{s}"),
    data = data,
    callback = callback,
    initial_steps = initial_steps,
    num_test_random = num_test_random,
    num_test_algo = num_test_algo,
    fraction_SM_obs = fraction_SM_obs,
    fraction_SS_obs = fraction_SS_obs,
    quarantine_HH = quarantine_HH,
    test_HH = test_HH,
    adoption_fraction = adoption_fraction,
    fp_rate = fp_rate,
    fn_rate = fn_rate,
    name_file_res = s + f"_N_{N}_T_{T}_obs_{num_test_algo}_SM_obs_{fraction_SM_obs}_seed_{seed}"
)
'''
loop_abm.free_abm(
    params_model,
    #seed=new_seed,
    logger = logging.getLogger(f"iteration.{s}"),
    data = data,
    callback = callback,
    name_file_res = s + f"_N_{N}_T_{T}_obs_{num_test_algo}_SM_obs_{fraction_SM_obs}_seed_{seed}"
)


