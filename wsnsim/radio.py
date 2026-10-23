"""First-order radio energy model (Heinzelman et al., 2000/2002)."""
import numpy as np


def tx_energy(cfg, bits, d):
    d = np.asarray(d, dtype=float)
    amp = np.where(d < cfg.d0, cfg.eps_fs * d ** 2, cfg.eps_mp * d ** 4)
    return bits * (cfg.e_elec + amp)


def rx_energy(cfg, bits):
    return bits * cfg.e_elec


def agg_energy(cfg, bits, n_signals):
    return cfg.e_da * bits * n_signals
