"""Synthetic, spatially correlated sensing environment.

Reading of node i at round r:
    v_i(r) = b_i(r) + n_i(r) + sum_{e active} A * exp(-||p_i - c_e||^2 / (2 s^2))
with b_i an AR(1) background drift, n_i white sensor noise and events e arriving
as a Poisson process with random centre c_e and exponential duration.

The whole event schedule is drawn up-front from its own RNG stream, so every
algorithm run with the same seed sees exactly the same environment.
"""
import numpy as np


class Environment:
    def __init__(self, cfg, x, y, rng):
        self.cfg = cfg
        self.rng = rng
        self.n = len(x)
        T = cfg.max_rounds
        n_ev = rng.poisson(cfg.event_rate * T)
        self.ev_start = np.sort(rng.integers(0, T, n_ev))
        self.ev_dur = np.maximum(1, np.round(rng.exponential(cfg.event_mean_dur, n_ev))).astype(int)
        self.ev_end = self.ev_start + self.ev_dur            # exclusive
        ex = rng.uniform(0, cfg.area_x, n_ev)
        ey = rng.uniform(0, cfg.area_y, n_ev)
        d2 = (x[None, :] - ex[:, None]) ** 2 + (y[None, :] - ey[:, None]) ** 2
        self.ev_infl = cfg.event_amp * np.exp(-d2 / (2 * cfg.event_sigma ** 2))   # (n_ev, N)
        # a node can "see" an event if the event alone exceeds the base threshold there
        self.detectable = self.ev_infl > cfg.theta0
        self.drift = np.zeros(self.n)

    def readings(self, r):
        c = self.cfg
        self.drift = c.drift_rho * self.drift + self.rng.normal(0, c.drift_std, self.n)
        noise = self.rng.normal(0, c.noise_std, self.n)
        active = np.nonzero((self.ev_start <= r) & (self.ev_end > r))[0]
        v = self.drift + noise
        if active.size:
            v = v + self.ev_infl[active].sum(axis=0)
        return v, active
