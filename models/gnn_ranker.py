

class GNNRanker():
    def __init__(self, encoder, propagator, decoder):
        '''

        :param encoder: encoding function,
            input: [N, d_h] batch of hidden vectors. [N,] vector of classes, each coordinate is 0 (no test) or 1 (negative test) or 2 (positive test).
            output: [N, d_{h'}] encoded hidden vectors
        :param propagator: GNN based hidden update function,
            input: StateGraph
            output: StateGraph
        :param decoder: neural network mapping to perform SIR 'classification' on each node output by the propagator
            input: StateGraph
            output: [N, 3] vector, each row sums to 1, representing
        '''
        self.encoder = encoder
        self.propagator = propagator
        self.decoder = decoder

    def encode(self, hidden_state):
        return self.encoder.forward(hidden_state)

    def propagate(self, hidden_state):
        return self.propagate(hidden_state)

    def decode(self, hidden_state):
        return self.decode(hidden_state)