"""Classic clustering baselines, implemented from their original papers.

LEACH-C  Heinzelman, Chandrakasan & Balakrishnan, IEEE Trans. Wireless Commun. 1(4), 2002.
         BS picks K = p*N CHs among nodes with E >= mean energy, minimising the total
         squared distance of nodes to their nearest CH, via simulated annealing.
DEEC     Qing, Zhu & Wang, Computer Communications 29(12), 2006.
         Distributed; p_i = p_opt * E_i / E_avg(r), epoch n_i = 1/p_i, threshold
         T = p_i / (1 - p_i (r mod n_i)) for nodes not CH in the last n_i rounds;
         E_avg(r) = (E_total / N)(1 - r / R), R = E_total / E_round estimated from the radio model.
PSO-C    Latiff, Tsimenidis & Sharif, IEEE PIMRC 2007.
         BS runs PSO over K CH positions (decoded to nearest eligible nodes, E >= mean),
         cost = beta * f1 + (1 - beta) * f2,
         f1 = max_k mean distance of cluster k's members to its CH,
         f2 = sum_i E_i / sum_k E_CH_k.
WOA-C    The same energy-distance cost optimised with the standard WOA (Mirjalili & Lewis,
         2016): isolates what the proposed fitness, trust and trigger add beyond WOA itself.

IWOA-SA  WOA-SA with an Iwendi-style fitness (our reimplementation of Iwendi et al.,
         Software: Practice and Experience 51(12), 2021). The paper gives the fitness only as a
         weighted sum of temperature, load, energy, distance and delay with unspecified weights,
         no SA placement or schedule, and no network size; the gaps are filled as follows
         (deviation table in the paper): temperature dropped (needs the authors' platform data);
         equal weights 0.25; all terms normalised to [0, 1]:
           F_E  = mean CH residual energy / max alive residual energy          (maximise)
           F_D  = 0.5 mean member-CH distance / D + 0.5 mean CH-BS distance / D,
                  D = max(field diagonal, max node-BS distance)                 (minimise)
           F_T  = largest cluster size / alive nodes   (TDMA delay proxy)       (minimise)
           F_L  = coefficient of variation of cluster sizes / sqrt(K - 1)       (minimise)
           FF   = 0.25 F_E + 0.25 (1 - F_D) + 0.25 (1 - F_T) + 0.25 (1 - F_L)
         Search: WOA (20 whales, 30 iterations) + 4 SA swap steps on the best whale after every
         iteration (740 fitness evaluations, matching BT-WOA-SA's 720).

None of these use trust. All re-cluster on the simulator's fixed interval (cfg.recluster_every).
"""
import numpy as np

from .base import ClusteringAlgorithm
from .search import Context, woa, pso, sa, woa_sa


class _Centralized(ClusteringAlgorithm):
    centralized = True

    def _context(self, sim):
        alive = np.nonzero(sim.alive)[0]
        Ea = sim.E[alive]
        cand = alive[Ea >= Ea.mean()]                     # energy-eligible nodes (LEACH-C / PSO-C rule)
        K = self.k_opt(sim)
        if cand.size < K:
            cand = alive[np.argsort(-Ea, kind="stable")][:K]
        ctx = Context(sim, cand)
        return ctx, min(K, cand.size)

    @staticmethod
    def energy_distance_cost(ctx, beta):
        """Latiff et al. (2007) cost, returned negated for maximisation."""
        E_alive_sum = ctx.E[ctx.alive].sum()

        def obj(Cs):
            Cs = np.atleast_2d(Cs)
            W, K = Cs.shape
            j, d = ctx.assign(Cs)
            onehot = (j[..., None] == np.arange(K))                    # (nA, W, K)
            sums = (onehot * d[..., None]).sum(0)                      # CH rows add 0
            counts = onehot.sum(0) - 1                                 # exclude the CH itself
            f1 = (sums / np.maximum(counts, 1)).max(1)
            f2 = E_alive_sum / ctx.E[Cs].sum(1)
            return -(beta * f1 + (1 - beta) * f2)
        return obj


class LEACHC(_Centralized):
    name = "LEACH-C"

    def __init__(self, label=None, sa_iter=500, sa_alpha=0.99):
        self.name = label or "LEACH-C"
        self.sa_iter, self.sa_alpha = sa_iter, sa_alpha

    def select(self, sim, r, setup_idx):
        ctx, K = self._context(sim)
        if K == 0:
            return np.array([], int)

        def obj(Cs):
            j, d = ctx.assign(np.atleast_2d(Cs))
            return -(d ** 2).sum(0)

        rng = sim.rng_alg
        C0 = rng.choice(ctx.cand, K, replace=False)
        C, _ = sa(ctx, obj, C0, rng, n_iter=self.sa_iter, alpha=self.sa_alpha)
        return C


class PSOC(_Centralized):
    name = "PSO-C"

    def __init__(self, label=None, beta=0.5, n_agents=20, n_iter=30):
        self.name = label or "PSO-C"
        self.beta, self.n_agents, self.n_iter = beta, n_agents, n_iter

    def select(self, sim, r, setup_idx):
        ctx, K = self._context(sim)
        if K == 0:
            return np.array([], int)
        C, _ = pso(ctx, self.energy_distance_cost(ctx, self.beta), K, sim.rng_alg,
                   n_agents=self.n_agents, n_iter=self.n_iter)
        return C


class WOAC(_Centralized):
    name = "WOA-C"

    def __init__(self, label=None, beta=0.5, n_agents=20, n_iter=30):
        self.name = label or "WOA-C"
        self.beta, self.n_agents, self.n_iter = beta, n_agents, n_iter

    def select(self, sim, r, setup_idx):
        ctx, K = self._context(sim)
        if K == 0:
            return np.array([], int)
        C, _ = woa(ctx, self.energy_distance_cost(ctx, self.beta), K, sim.rng_alg,
                   n_agents=self.n_agents, n_iter=self.n_iter)
        return C


class DEEC(ClusteringAlgorithm):
    name = "DEEC"
    centralized = False

    def __init__(self, label=None, true_average=False):
        self.name = label or "DEEC"
        self.true_average = true_average      # paper estimates E_avg; option to use the true value

    def reset(self, sim):
        cfg = sim.cfg
        N = sim.N
        self.last_ch = np.full(N, -10 ** 9)
        self.E_total = N * cfg.e_init
        k = max(1, int(round(cfg.p_ch * N)))
        M = np.sqrt(cfg.area_x * cfg.area_y)
        d_ch = M / np.sqrt(2 * np.pi * k)
        d_bs = float(np.mean(sim.d_bs))
        L = cfg.data_bits
        self.E_round = L * (2 * N * cfg.e_elec + N * cfg.e_da + k * cfg.eps_mp * d_bs ** 4
                            + N * cfg.eps_fs * d_ch ** 2)
        self.R = self.E_total / self.E_round

    def select(self, sim, r, setup_idx):
        cfg = sim.cfg
        alive = sim.alive
        if self.true_average:
            E_avg = sim.E[alive].mean() if alive.any() else 0.0
        else:
            E_avg = (self.E_total / sim.N) * max(1 - r / self.R, 1e-3)
        p = np.clip(cfg.p_ch * sim.E / E_avg, 1e-6, 1.0)
        n_i = np.maximum(np.round(1 / p), 1).astype(int)
        eligible = alive & ((setup_idx - self.last_ch) >= n_i)
        T = p / (1 - p * (setup_idx % n_i))
        T = np.where(T > 0, T, 1.0)                       # guard r mod n_i edge case
        chs = np.nonzero(eligible & (sim.rng_alg.random(sim.N) < T))[0]
        self.last_ch[chs] = setup_idx
        return chs


class IWOASA(_Centralized):
    name = "IWOA-SA"

    def __init__(self, label=None, weights=(0.25, 0.25, 0.25, 0.25), n_agents=20, n_iter=30, sa_steps=4):
        self.name = label or "IWOA-SA"
        self.w = weights
        self.n_agents, self.n_iter, self.sa_steps = n_agents, n_iter, sa_steps

    def iwendi_fitness(self, ctx, sim):
        wE, wD, wT, wL = self.w
        Emax = max(float(ctx.E[ctx.alive].max()), 1e-12)
        nA = ctx.alive.size
        cfg = sim.cfg
        Dn = max(float(np.hypot(cfg.area_x, cfg.area_y)), float(sim.d_bs[ctx.alive].max()))
        d_bs = sim.d_bs

        def obj(Cs):
            Cs = np.atleast_2d(Cs)
            W, K = Cs.shape
            j, d = ctx.assign(Cs)
            onehot = (j[..., None] == np.arange(K))                    # (nA, W, K)
            sizes = onehot.sum(0).astype(float)                        # incl. CH, (W, K)
            fE = ctx.E[Cs].mean(1) / Emax
            fD = 0.5 * d.sum(0) / max(nA - K, 1) / Dn + 0.5 * d_bs[Cs].mean(1) / Dn
            fT = sizes.max(1) / nA
            cv = sizes.std(1) / np.maximum(sizes.mean(1), 1e-12)
            fL = cv / np.sqrt(max(K - 1, 1))
            return wE * fE + wD * (1 - fD) + wT * (1 - fT) + wL * (1 - np.minimum(fL, 1.0))
        return obj

    def select(self, sim, r, setup_idx):
        ctx, K = self._context(sim)
        if K == 0:
            return np.array([], int)
        C, _ = woa_sa(ctx, self.iwendi_fitness(ctx, sim), K, sim.rng_alg,
                      n_agents=self.n_agents, n_iter=self.n_iter, sa_steps=self.sa_steps)
        return C
