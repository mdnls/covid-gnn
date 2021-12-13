from .template_rank import AbstractRanker
from models.graph import StateGraph

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
    '''

    :param state_graph: a StateGraph for time t
    :return: a matrix [N by 3] whose rows sum to 1. Predicted probabilities of [S, I, R] for each person.
    '''
    return ...

    def rank(self, state_graph):
        '''
        computing rank of infected individuals
        return: list -- [(index, value), ...]
        '''
        pred_prob = predict(state_graph)
        inf_prob = [[i, predict[i,1]] for i in range(self.N)]
            
        rank = list(sorted(inf_prob, key=lambda tup: tup[1], reverse=True))
        return rank
