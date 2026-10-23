"""Algorithm interface: select(sim, r, setup_idx) -> array of CH node ids."""
import numpy as np


class ClusteringAlgorithm:
    name = "base"
    centralized = False     # True -> BS computes CHs and broadcasts them

    def reset(self, sim):
        pass

    def select(self, sim, r, setup_idx):
        raise NotImplementedError

    @staticmethod
    def k_opt(sim):
        """Number of CHs: p * (alive nodes), at least 1."""
        return max(1, int(round(sim.cfg.p_ch * sim.alive.sum())))


class Direct(ClusteringAlgorithm):
    """No clustering: every node transmits straight to the BS (sanity baseline)."""
    name = "Direct"

    def select(self, sim, r, setup_idx):
        return np.array([], int)
