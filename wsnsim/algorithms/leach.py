"""LEACH (Heinzelman et al., 2000): distributed probabilistic CH rotation.

T(n) = p / (1 - p * (r mod 1/p))  for nodes not yet CH in the current epoch (set G),
0 otherwise. Every node is CH exactly once per epoch of 1/p rounds.
"""
import numpy as np
from .base import ClusteringAlgorithm


class LEACH(ClusteringAlgorithm):
    name = "LEACH"
    centralized = False

    def reset(self, sim):
        self.was_ch = np.zeros(sim.N, bool)

    def select(self, sim, r, setup_idx):
        p = sim.cfg.p_ch
        epoch = int(round(1 / p))
        s = setup_idx % epoch
        if s == 0:
            self.was_ch[:] = False
        T = p / (1 - p * s)
        cand = sim.alive & ~self.was_ch
        chs = np.nonzero(cand & (sim.rng_alg.random(sim.N) < T))[0]
        self.was_ch[chs] = True
        return chs
