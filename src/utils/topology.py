

class ESAMTopology:

    def __init__(self, f, m, s_m, f_h_sparsity, h, h_f_sparsity_e, h_f_sparsity_i):
        self.neurons = f * m * h

        # Connection Count
        max_conns = f * h * m
        self.conns_f_h = max_conns * s_m * f_h_sparsity
        self.conns_h_f_e = max_conns * s_m * h_f_sparsity_e
        self.conns_h_f_i = max_conns * (1-s_m) * h_f_sparsity_i
        self.conns_h_f = self.conns_h_f_e + self.conns_h_f_i
        self.conns_total = self.conns_f_h + self.conns_h_f

        # Fan-in per neuron type
        self.fan_in_per_h = f * s_m * f_h_sparsity
        self.fan_in_per_f = self.conns_h_f / f


