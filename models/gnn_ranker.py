

class GNNRanker():
    def __init__(self, encoder, propagator, decoder, predictor):
        '''
        :param encoder: graph encoder network, StateGraph -> StateGraph
        :param propagator: GNN based hidden update function,
        :param decoder: neural network mapping to perform SIR 'classification' on each node output by the propagator
        :param predictor: map state vectors to SIR predictions
        '''
        self.encoder = encoder
        self.propagator = propagator
        self.decoder = decoder
        self.predictor = predictor

    def state_update(self, state_graph, observations):
        '''
        Given an input state graph, encode its information into a graph and propagate hidden data, returning the
        new hidden data.

        :param state_graph: a StateGraph
        :param observations: a [T by N] sparse matrix of observations
        :return: an [N, d_h] matrix of new hidden information for each node
        '''
        enc_state_graph = self.encoder(state_graph, observations)
        prp_state_graph = self.propagator(enc_state_graph)
        dec_state_vectors = self.decoder(prp_state_graph)
        return dec_state_vectors

    def predict(self, state_vectors):
        return self.predictor(state_vectors)

