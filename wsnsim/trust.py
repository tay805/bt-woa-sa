"""Base-station-side Beta reputation for cluster heads.

Members stamp data packets with sequence numbers, so the BS learns, with a
one-round lag, how many member packets a CH received (S + F) and how many it
actually forwarded (S). Trust:  T_i = (a_i + 1) / (a_i + b_i + 2)
with forgetting  a_i <- rho * a_i + S_i,  b_i <- rho * b_i + F_i  on observation.
A node never observed keeps the uninformative prior T = 0.5.
"""
import numpy as np


class TrustManager:
    def __init__(self, n, cfg):
        self.cfg = cfg
        self.a = np.zeros(n)
        self.b = np.zeros(n)
        self.pending = None

    def submit(self, S, F):
        """Observations from the current round; applied at the next round."""
        self.pending = (S.copy(), F.copy())

    def apply_pending(self):
        if self.pending is None:
            return
        S, F = self.pending
        obs = (S + F) > 0
        rho = self.cfg.trust_forget
        self.a[obs] = rho * self.a[obs] + S[obs]
        self.b[obs] = rho * self.b[obs] + F[obs]
        self.pending = None

    @property
    def value(self):
        return (self.a + 1.0) / (self.a + self.b + 2.0)

    @property
    def observed(self):
        return (self.a + self.b) > 0

    def detection_stats(self, malicious):
        """TPR / FPR of 'trust < threshold' among nodes the BS has observed."""
        obs = self.observed
        pred = self.value < self.cfg.trust_threshold
        mal, ben = obs & malicious, obs & ~malicious
        tpr = (pred & mal).sum() / mal.sum() if mal.sum() else np.nan
        fpr = (pred & ben).sum() / ben.sum() if ben.sum() else np.nan
        return dict(trust_tpr=float(tpr), trust_fpr=float(fpr),
                    n_observed=int(obs.sum()))
