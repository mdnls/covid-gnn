from .template_rank import AbstractRanker
from models.graphs import StateGraph
import numpy as np

'''
class Ranker(AbstractRanker):

    def __init__(self, delay=0):
        self.delay = delay
        self.rng = np.random.RandomState(np.random.randint(1000))

    def init(self, cnf):
        self.transmissions = []
        self.observations = []
        self.T = cnt.setting.time_total
        self.N = cnt.setting.n_vertices
        self.mfIs = np.full(cnt.setting.time_total, np.nan)

        return True

    def predict(state_graph):
    ''

    :param state_graph: a StateGraph for time t
    :return: a matrix [N by 3] whose rows sum to 1. Predicted probabilities of [S, I, R] for each person.
    ''
    return ...

    def rank(self, state_graph):
        ''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        ''
        pred_prob = predict(state_graph)
        inf_prob = [[i, pred_prob[i,1]] for i in range(self.N)]
            
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank
'''
class RandomRanker(AbstractRanker):

    def __init__(self, delay=0):
        self.delay = delay
        self.rng = np.random.RandomState(np.random.randint(1000))

    def init(self, cnf):
        self.transmissions = []
        self.observations = []
        self.T = cnt.setting.time_total
        self.N = cnt.setting.n_vertices
        self.mfIs = np.full(cnt.setting.time_total, np.nan)

        return True

    def predict(self):
        '''
        :return: a matrix [N by 3] whose rows sum to 1. Predicted probabilities of [S, I, R] for each person.
        '''
        inf_prob = np.random.uniform(0,0.9,size=self.N)
        rec_prob = np.full(self.N,0.1)
        sus_prob = 0.9 - inf_prob

        return np.array([sus_prob,inf_prob,rec_prob]).reshape(self.N,3)

    def rank(self, state_graph):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        pred_prob = predict()
        inf_prob = [[i, pred_prob[i,1]] for i in range(self.N)]
            
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank

class DotdRanker(AbstractRanker):

    def __init__(self):
        self.description = "class for random tests of openABM loop"
        self.rng = np.random.RandomState(1)
    
    def init(self, N, T):
        self.T = T
        self.N = N

        return True

    def rank(self, t_day, daily_contacts, daily_obs, data):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        rank = [[i, self.rng.random()] for i in range(self.N)]

        return rank


