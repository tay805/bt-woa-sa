"""Proposed method BT-WOA-SA: Bayesian trust-aware WOA-SA clustering, executed at the BS.

Default configuration (the one evaluated in the paper):
  1. Bayesian suspicion gate. For every alive node, z_i = P(phi_i < susp_p | evidence) with
     phi_i ~ Beta(a_i + a0, b_i + b0), where (a_i, b_i) are the discounted forwarding successes
     and failures the BS observed while node i was CH (wsnsim/trust.py) and (a0, b0) = susp_prior
     = (9, 1) is an optimistic prior. Nodes with z_i > susp_max (0.5) are excluded from CH
     candidacy. If every node is excluded, all alive nodes are used.
  2. Candidate set: trusted nodes with residual energy >= the mean of the trusted pool
     (the LEACH-C / PSO-C rule); the K highest-energy trusted nodes if fewer than K remain.
  3. Cost (minimised, returned negated): beta * f1 + (1 - beta) * f2, beta = 0.5
     (Latiff et al., PIMRC 2007), f1 = largest mean member-to-CH distance over clusters,
     f2 = total alive energy / total CH energy.
  4. Search: WOA (Mirjalili and Lewis, 2016) over continuous CH coordinates, each whale decoded
     to the nearest distinct candidates; the best set is refined by simulated annealing (swap one
     CH for a candidate within sa_radius, Metropolis acceptance).
  Re-clustering uses the simulator's fixed interval (cfg.recluster_every).

Options used for ablations and comparisons:
  use_trust=False             no gate (equals WOA-SA on the PSO-C cost)
  use_woa=False / use_sa=False   SA only / WOA only
  use_trigger=True            energy-triggered re-clustering instead of the fixed interval
  gate="mean"                 CTRF-style gate: trust (posterior mean) >= average over alive nodes
  susp_prior, susp_max        prior and threshold sensitivity (experiment E7)
  fitness_mode="v1" / "v2"    earlier fitness designs kept for reference (not used in the paper)
"""
import numpy as np

from .base import ClusteringAlgorithm
from ..radio import tx_energy, rx_energy, agg_energy


class EWOASA(ClusteringAlgorithm):
    """Proposed method (default configuration = BT-WOA-SA, Bayesian Trust-aware WOA-SA):
    fitness_mode="v3" (energy-distance cost of Latiff et al. 2007 on the trusted candidate pool),
    suspicious nodes excluded, WOA search + SA refinement, fixed-interval re-clustering.
    fitness_mode="v2"/"v1" and use_trigger=True reproduce the earlier designs for ablation."""
    centralized = True

    def __init__(self, label=None, use_trust=True, use_sa=True, use_rotation=True,
                 use_trigger=False, use_woa=True,
                 weights=None, fitness_mode="v3",
                 n_whales=20, n_iter=30, b_spiral=1.0,
                 sa_iter=100, sa_t0=0.02, sa_alpha=0.97, sa_radius=20.0,
                 e_frac=0.5, z_lcb=1.0, tau=20.0,
                 delta=0.02, max_interval=50,
                 susp_prior=(9.0, 1.0), susp_p=0.85, susp_max=0.5, gate="bayes"):
        self.name = label or "BT-WOA-SA"
        self.use_trust, self.use_sa, self.use_rotation = use_trust, use_sa, use_rotation
        self.use_trigger, self.use_woa = use_trigger, use_woa
        self.mode = fitness_mode
        self.beta = 0.5
        if fitness_mode == "v3":
            weights = weights or (0.0, 0.0, 0.0, 0.0)
        if weights is None:
            # v1: (E, T, S, B, R)       v2: (L, S, T, R)
            weights = (0.25, 0.20, 0.30, 0.15, 0.10) if fitness_mode == "v1" else (0.45, 0.30, 0.15, 0.10)
        w = np.array(weights, float)
        iT, iR = (1, 4) if fitness_mode == "v1" else (2, 3)
        if not use_trust:
            w[iT] = 0.0
        if not use_rotation:
            w[iR] = 0.0
        self.w = w
        self.n_whales, self.n_iter, self.b = n_whales, n_iter, b_spiral
        self.sa_iter, self.sa_t0, self.sa_alpha, self.sa_radius = sa_iter, sa_t0, sa_alpha, sa_radius
        self.e_frac, self.z, self.tau = e_frac, z_lcb, tau
        self.delta, self.max_interval = delta, max_interval
        self.susp_prior, self.susp_p, self.susp_max = susp_prior, susp_p, susp_max
        self.gate = gate                  # "bayes" (proposed) | "mean" (CTRF-style: trust >= network mean)

    # lifecycle
    def reset(self, sim):
        k = sim.cfg.data_bits
        self.Etx = tx_energy(sim.cfg, k, sim.D)            # member -> CH cost matrix (N x N)
        self.Etx_bs = tx_energy(sim.cfg, k, sim.d_bs)
        self.last_setup = np.full(sim.N, -10 ** 9, dtype=float)
        self.E_at_setup = np.zeros(sim.N)
        self.setup_round = 0
        self.fitness_trace = []          # best fitness per WOA iteration (last set-up)
        self.gate_log = []               # (round, honest excluded frac, malicious excluded frac)
        self.first_excluded = np.full(sim.N, -1)
        self.n_setups = 0

    def _log_gate(self, sim, alive, ok):
        mal = sim.malicious[alive]
        excl = ~ok
        self.gate_log.append((sim.rounds_run,
                              excl[~mal].mean() if (~mal).any() else np.nan,
                              excl[mal].mean() if mal.any() else np.nan))
        newly = alive[excl & mal & (self.first_excluded[alive] < 0)]
        self.first_excluded[newly] = sim.rounds_run

    def gate_stats(self, sim):
        if not self.gate_log:
            return {}
        g = np.array(self.gate_log, float)
        hnd = None
        alive_hist = np.asarray(sim.hist["alive"])
        if (alive_hist <= sim.N / 2).any():
            hnd = int(np.argmax(alive_hist <= sim.N / 2))
        pre = g[:, 0] < (hnd if hnd is not None else np.inf)
        mal_ch = sim.malicious & (sim.ch_count > 0)
        det = self.first_excluded[mal_ch]
        return dict(gate_honest_excluded=float(np.nanmean(g[pre, 1])) if pre.any() else np.nan,
                    gate_malicious_excluded=float(np.nanmean(g[pre, 2])) if pre.any() else np.nan,
                    gate_detected_frac=float((det >= 0).mean()) if det.size else np.nan,
                    gate_detect_round=float(det[det >= 0].mean()) if (det >= 0).any() else np.nan)

    def need_recluster(self, sim, r):
        if not self.use_trigger:
            return None                  # simulator falls back to the fixed interval
        chs = sim.chs
        if chs.size == 0:
            return True
        spent = self.E_at_setup[chs] - sim.E[chs]
        return bool((spent >= self.delta * sim.cfg.e_init).any() or
                    (r - self.setup_round) >= self.max_interval)

    def on_setup_done(self, sim):
        self.E_at_setup[:] = sim.E       # after control packets are charged

    # fitness
    def _prepare(self, sim, s):
        cfg = sim.cfg
        alive = np.nonzero(sim.alive)[0]
        self.alive_idx = alive
        E = sim.E
        Ea = E[alive]
        tm = sim.trust.value
        n = sim.trust.a + sim.trust.b
        var = tm * (1 - tm) / (n + 3)
        self.T_lcb = np.round(np.clip(tm - self.z * np.sqrt(var), 0, 1), 9)
        # suspicion = P(forwarding rate < susp_p | evidence), "innocent until proven guilty" prior
        from scipy.stats import beta as _beta
        a0, b0 = self.susp_prior
        self.susp = np.round(_beta.cdf(self.susp_p, sim.trust.a + a0, sim.trust.b + b0), 9)
        self.Emax = Ea.max()
        self.rec = np.round(np.exp(-(s - self.last_setup) / self.tau), 9)
        k = cfg.data_bits
        self.e_rx = rx_energy(cfg, k)
        self.J_direct = self.Etx_bs[alive].sum()
        self.EA = self.Etx[alive]                          # rows: alive nodes
        self.DA = sim.D[alive]                             # rounded distances: decide membership
        self.E_alive = E[alive]
        self.pos = np.full(sim.N, -1)
        self.pos[alive] = np.arange(alive.size)

        # trusted pool: never trade trust for energy; suspicious nodes only if nobody else is left
        if self.use_trust:
            if self.mode == "v1":
                ok = tm[alive] >= cfg.trust_threshold
            elif self.gate == "mean":
                ok = tm[alive] >= tm[alive].mean()          # CTRF-style gate (Hosseinzadeh et al. 2023)
            else:
                ok = self.susp[alive] <= self.susp_max      # proposed Bayesian suspicion gate
            trusted = alive[ok] if ok.any() else alive
            self._log_gate(sim, alive, ok)
        else:
            trusted = alive
        Et = E[trusted]
        frac = 1.0 if self.mode == "v3" else self.e_frac       # v3: PSO-C/LEACH-C rule E >= mean
        cand = trusted[Et >= frac * Et.mean()]
        K = self.k_opt(sim)
        if cand.size < K:                 # relax the energy filter, within the trusted pool
            cand = trusted[np.argsort(-Et)][:K]
        self.cand = cand
        self.cand_xy = np.c_[sim.x[cand], sim.y[cand]]
        self.K = min(K, cand.size)

    def fitness_batch(self, sim, Cs):
        """Fitness of W candidate CH sets at once. Cs: (W, K) node ids, distinct per row."""
        cfg, k = sim.cfg, sim.cfg.data_bits
        Cs = np.atleast_2d(Cs)
        W, K = Cs.shape
        sub = self.EA[:, Cs]                               # (nA, W, K) member uplink energy
        j = np.argmin(self.DA[:, Cs], axis=2)              # nearest CH per alive node (same rule as simulator)
        m = np.take_along_axis(sub, j[..., None], axis=2)[..., 0]          # (nA, W)
        pc = self.pos[Cs]                                  # CH rows are not members
        J_m = m.sum(0) - m[pc, np.arange(W)[:, None]].sum(1)
        n_c = (j[..., None] == np.arange(K)).sum(0) - 1    # (W, K); each CH row picked itself
        J_c = n_c * self.e_rx + cfg.e_da * k * (n_c + 1) + self.Etx_bs[Cs]
        S = np.clip(1 - (J_m + J_c.sum(1)) / self.J_direct, -1, 1)
        Ec = sim.E[Cs]
        if self.mode == "v3":
            jd = np.argmin(self.DA[:, Cs], axis=2)
            dd = np.take_along_axis(self.DA[:, Cs], jd[..., None], axis=2)[..., 0]
            onehot = (jd[..., None] == np.arange(K))
            sums = (onehot * dd[..., None]).sum(0)
            counts = onehot.sum(0) - 1
            f1 = (sums / np.maximum(counts, 1)).max(1)
            f2 = self.E_alive.sum() / Ec.sum(1)
            return np.round(-(self.beta * f1 + (1 - self.beta) * f2), 9)
        R = self.rec[Cs].mean(1)
        if self.mode == "v1":
            t = Ec / J_c
            B = t.min(1) / t.mean(1)
            return np.round(self.w[0] * Ec.mean(1) / self.Emax + self.w[1] * self.T_lcb[Cs].mean(1)
                            + self.w[2] * S + self.w[3] * B - self.w[4] * R, 9)
        # v2: bottleneck lifetime ratio. Per-node cost this round: members pay their uplink,
        # CHs pay reception + aggregation + uplink to the BS.
        J = m.copy()                                        # (nA, W)
        J[pc, np.arange(W)[:, None]] = J_c
        life = self.E_alive[:, None] / J                    # rounds each node could sustain
        L = life.min(0) / (self.E_alive.sum() / J.sum(0))   # bottleneck / network-average, in (0, 1]
        Z = self.susp[Cs].mean(1)                           # suspicion of the chosen CHs
        return np.round(self.w[0] * L + self.w[1] * S - self.w[2] * Z - self.w[3] * R, 9)

    def fitness(self, sim, C):
        return float(self.fitness_batch(sim, np.asarray(C)[None, :])[0])

    # search
    def _decode_batch(self, Xs):
        """(W, 2K) coordinates -> (W, K) distinct candidate ids, nearest-first greedy."""
        W = Xs.shape[0]
        P = Xs.reshape(W, self.K, 2)
        d = np.round(np.hypot(P[..., 0, None] - self.cand_xy[None, None, :, 0],
                              P[..., 1, None] - self.cand_xy[None, None, :, 1]), 9)   # (W, K, M)
        order = np.argsort(d.min(axis=2), axis=1)                            # closest points first
        used = np.zeros((W, self.cand.size), bool)
        out = np.empty((W, self.K), int)
        rows = np.arange(W)
        for step in range(self.K):
            kk = order[:, step]
            dd = np.where(used, np.inf, d[rows, kk])
            jj = np.argmin(dd, axis=1)
            used[rows, jj] = True
            out[rows, kk] = jj
        return self.cand[out]

    def _woa(self, sim, rng):
        cfg = sim.cfg
        dim, nW = 2 * self.K, self.n_whales
        lo = np.zeros(dim)
        hi = np.tile([max(cfg.area_x, sim.x.max()), max(cfg.area_y, sim.y.max())], self.K)
        X = np.stack([self.cand_xy[rng.choice(self.cand.size, self.K, replace=False)].ravel()
                      for _ in range(nW)])
        sets = self._decode_batch(X)
        fit = self.fitness_batch(sim, sets)
        b = int(np.argmax(fit))
        best_x, best_set, best_f = X[b].copy(), sets[b].copy(), fit[b]
        trace = [best_f]
        for t in range(self.n_iter):
            a = 2.0 - 2.0 * t / self.n_iter
            A = 2 * a * rng.random((nW, dim)) - a
            Cc = 2 * rng.random((nW, dim))
            p = rng.random(nW)
            l = rng.uniform(-1, 1, (nW, dim))
            explo = np.all(np.abs(A) < 1, axis=1)
            Xr = X[rng.integers(nW, size=nW)]
            enc = best_x - A * np.abs(Cc * best_x - X)                       # encircle best
            srch = Xr - A * np.abs(Cc * Xr - X)                              # search around random whale
            spir = np.abs(best_x - X) * np.exp(self.b * l) * np.cos(2 * np.pi * l) + best_x
            X = np.where((p < 0.5)[:, None], np.where(explo[:, None], enc, srch), spir)
            X = np.round(np.clip(X, lo, hi), 9)
            sets = self._decode_batch(X)
            fit = self.fitness_batch(sim, sets)
            b = int(np.argmax(fit))
            if fit[b] > best_f:
                best_x, best_set, best_f = X[b].copy(), sets[b].copy(), fit[b]
            trace.append(best_f)
        self.fitness_trace = trace
        return best_set, best_f

    def _sa(self, sim, rng, C, f):
        cur, fcur = C.copy(), f
        best, fbest = cur.copy(), fcur
        T = self.sa_t0
        for _ in range(self.sa_iter):
            j = rng.integers(cur.size)
            near = self.cand[(sim.D[cur[j], self.cand] <= self.sa_radius) & ~np.isin(self.cand, cur)]
            pool = near if near.size else self.cand[~np.isin(self.cand, cur)]
            if pool.size == 0:
                break
            new = cur.copy()
            new[j] = rng.choice(pool)
            fn = self.fitness(sim, new)
            if fn > fcur or rng.random() < np.exp((fn - fcur) / max(T, 1e-12)):
                cur, fcur = new, fn
                if fcur > fbest:
                    best, fbest = cur.copy(), fcur
            T *= self.sa_alpha
        return best, fbest

    def select(self, sim, r, setup_idx):
        rng = sim.rng_alg
        self._prepare(sim, setup_idx)
        if self.K == 0:
            return np.array([], int)
        if self.use_woa:
            C, f = self._woa(sim, rng)
        else:                              # random-restart baseline for ablation
            C = rng.choice(self.cand, self.K, replace=False)
            f = self.fitness(sim, C)
        if self.use_sa:
            C, f = self._sa(sim, rng, C, f)
        self.last_setup[C] = setup_idx
        self.setup_round = r
        self.n_setups += 1
        return C
