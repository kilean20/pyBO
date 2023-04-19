import numpy as np

class Particle:
    def __init__(self, A, Q, W_input, initial_phase=0.0):
        self.A = A
        self.Q = Q
        self.input_energy_in_eV_u = W_input
        self.initial_phase = initial_phase
        self.initial_Q = Q
        self.stripped_Q = Q
        self.zp = None
        self.taup = None
        self.Wp = None
        self.W = None
        self.z = None
        self.tau = None

    @property
    def beta(self):
        gamma = 1.0 + self.W / self.rest_energy
        beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
        return beta

    @property
    def input_energy_in_eV(self):
        return self.input_energy_in_eV_u * self.A

    @property
    def rest_energy(self):
        return 931.5e6 * self.A
