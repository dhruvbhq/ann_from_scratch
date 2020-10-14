# The neuron class file

# The necessary imports
import numpy as np

class neuron:
    def __init__(self, addr, input_dim):
        self.addr = addr
        self.input_dim = input_dim
        self.input_vec = np.zeros([input_dim])
        self.edge_weights = np.zeros([input_dim])
        self.bias = 0
        self.activation = 0
        self.output = 0

    def init_weights(self):
        self.edge_weights = np.random([input_dim])
        self.bias = np.random()

    def calc_activation(self):
        self.activation = np.dot(np.transpose(self.edge_weight), self.input_vec) + self.bias

    def apply_activation_function(self):
        self.activation = np.tanh(self.activation)

    def forward_pass(self, input_vec):
        self.init_weights()
        self.input_vec = input_vec
        self.calc_activation()
        self.apply_activation_function()
        self.output = self.activation

