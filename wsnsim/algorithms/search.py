"""Shared search machinery for centralized baselines (LEACH-C, PSO-C, WOA-C).

Positions are continuous CH coordinates (K x 2 per agent) decoded to the nearest
distinct candidate nodes, the standard approach of PSO-C (Latiff et al., 2007).
All objective values and positions are rounded to 1e-9 so that last-bit
floating-point differences across numpy builds/CPUs cannot change a decision.
Objectives are MAXIMISED; cost functions are passed negated.
"""
import numpy as np


class Context:
    """Per-set-up snapshot the objectives need (alive set, candidates, distances)."""

    def __init__(self, sim, cand):
        self.alive = np.nonzero(sim.alive)[0]
        self.DA = sim.D[self.alive]                         # rounded distances (alive x N)
        self.pos = np.full(sim.N, -1)
        self.pos[self.alive] = np.arange(self.alive.size)
        self.cand = np.asarray(cand, int)
        self.cand_xy = np.c_[sim.x[self.cand], sim.y[self.cand]]
        self.E = sim.E
        self.lo = np.array([min(0.0, sim.x.min()), min(0.0, sim.y.min())])
        self.hi = np.array([max(sim.cfg.area_x, sim.x.max()), max(sim.cfg.area_y, sim.y.max())])

    def assign(self, Cs):
        """Nearest-CH index (nA, W) and distance (nA, W) for W candidate sets."""
        sub = self.DA[:, Cs]                                # (nA, W, K)
        j = np.argmin(sub, axis=2)
        d = np.take_along_axis(sub, j[..., None], axis=2)[..., 0]
        return j, d


def decode_batch(ctx, Xs, K):
    W = Xs.shape[0]
    P = Xs.reshape(W, K, 2)
    d = np.round(np.hypot(P[..., 0, None] - ctx.cand_xy[None, None, :, 0],
                          P[..., 1, None] - ctx.cand_xy[None, None, :, 1]), 9)
    order = np.argsort(d.min(axis=2), axis=1, kind="stable")
    used = np.zeros((W, ctx.cand.size), bool)
    out = np.empty((W, K), int)
    rows = np.arange(W)
    for step in range(K):
        kk = order[:, step]
        jj = np.argmin(np.where(used, np.inf, d[rows, kk]), axis=1)
        used[rows, jj] = True
        out[rows, kk] = jj
    return ctx.cand[out]


def _init_positions(ctx, K, n, rng):
    return np.stack([ctx.cand_xy[rng.choice(ctx.cand.size, K, replace=False)].ravel() for _ in range(n)])


def woa(ctx, obj, K, rng, n_agents=20, n_iter=30, b=1.0):
    """Whale Optimization Algorithm (Mirjalili & Lewis, 2016), synchronous update."""
    dim = 2 * K
    lo, hi = np.tile(ctx.lo, K), np.tile(ctx.hi, K)
    X = _init_positions(ctx, K, n_agents, rng)
    sets = decode_batch(ctx, X, K)
    fit = np.round(obj(sets), 9)
    i = int(np.argmax(fit))
    best_x, best_set, best_f = X[i].copy(), sets[i].copy(), fit[i]
    for t in range(n_iter):
        a = 2.0 - 2.0 * t / n_iter
        A = 2 * a * rng.random((n_agents, dim)) - a
        C = 2 * rng.random((n_agents, dim))
        p = rng.random(n_agents)
        l = rng.uniform(-1, 1, (n_agents, dim))
        exploit = np.all(np.abs(A) < 1, axis=1)
        Xr = X[rng.integers(n_agents, size=n_agents)]
        enc = best_x - A * np.abs(C * best_x - X)
        srch = Xr - A * np.abs(C * Xr - X)
        spir = np.abs(best_x - X) * np.exp(b * l) * np.cos(2 * np.pi * l) + best_x
        X = np.where((p < 0.5)[:, None], np.where(exploit[:, None], enc, srch), spir)
        X = np.round(np.clip(X, lo, hi), 9)
        sets = decode_batch(ctx, X, K)
        fit = np.round(obj(sets), 9)
        i = int(np.argmax(fit))
        if fit[i] > best_f:
            best_x, best_set, best_f = X[i].copy(), sets[i].copy(), fit[i]
    return best_set, best_f


def pso(ctx, obj, K, rng, n_agents=20, n_iter=30, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
    """Global-best PSO with linearly decreasing inertia (Shi & Eberhart, 1998)."""
    dim = 2 * K
    lo, hi = np.tile(ctx.lo, K), np.tile(ctx.hi, K)
    vmax = 0.2 * (hi - lo)
    X = _init_positions(ctx, K, n_agents, rng)
    V = np.zeros_like(X)
    sets = decode_batch(ctx, X, K)
    fit = np.round(obj(sets), 9)
    pbest, pbest_f, pbest_set = X.copy(), fit.copy(), sets.copy()
    g = int(np.argmax(fit))
    gbest, gbest_f, gbest_set = X[g].copy(), fit[g], sets[g].copy()
    for t in range(n_iter):
        w = w_max - (w_max - w_min) * t / max(n_iter - 1, 1)
        r1, r2 = rng.random((n_agents, dim)), rng.random((n_agents, dim))
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X), -vmax, vmax)
        X = np.round(np.clip(X + V, lo, hi), 9)
        sets = decode_batch(ctx, X, K)
        fit = np.round(obj(sets), 9)
        better = fit > pbest_f
        pbest[better], pbest_f[better], pbest_set[better] = X[better], fit[better], sets[better]
        g = int(np.argmax(pbest_f))
        if pbest_f[g] > gbest_f:
            gbest, gbest_f, gbest_set = pbest[g].copy(), pbest_f[g], pbest_set[g].copy()
    return gbest_set, gbest_f


def sa(ctx, obj, C0, rng, n_iter=200, t0=None, alpha=0.97):
    """Simulated annealing over CH sets: swap one CH for a random non-CH candidate."""
    cur = np.asarray(C0, int).copy()
    fcur = float(np.round(obj(cur[None, :])[0], 9))
    best, fbest = cur.copy(), fcur
    T = t0 if t0 is not None else max(abs(fcur) * 0.05, 1e-9)
    for _ in range(n_iter):
        pool = ctx.cand[~np.isin(ctx.cand, cur)]
        if pool.size == 0:
            break
        new = cur.copy()
        new[rng.integers(cur.size)] = rng.choice(pool)
        fn = float(np.round(obj(new[None, :])[0], 9))
        if fn > fcur or rng.random() < np.exp((fn - fcur) / T):
            cur, fcur = new, fn
            if fcur > fbest:
                best, fbest = cur.copy(), fcur
        T *= alpha
    return best, fbest


def woa_sa(ctx, obj, K, rng, n_agents=20, n_iter=30, b=1.0, sa_steps=4, sa_alpha=0.97):
    """WOA with an SA local search applied to the best whale after every iteration
    (the usual WOA-SA hybrid, e.g. Mafarja & Mirjalili 2017; Iwendi et al. 2021 describe SA
    as a local-search operator whose result replaces the solution only if better).
    The SA temperature is carried across iterations and cooled geometrically."""
    dim = 2 * K
    lo, hi = np.tile(ctx.lo, K), np.tile(ctx.hi, K)
    X = _init_positions(ctx, K, n_agents, rng)
    sets = decode_batch(ctx, X, K)
    fit = np.round(obj(sets), 9)
    i = int(np.argmax(fit))
    best_x, best_set, best_f = X[i].copy(), sets[i].copy(), fit[i]
    T = max(abs(float(best_f)) * 0.05, 1e-9)
    cand_index = {int(c): k for k, c in enumerate(ctx.cand)}
    for t in range(n_iter):
        a = 2.0 - 2.0 * t / n_iter
        A = 2 * a * rng.random((n_agents, dim)) - a
        C = 2 * rng.random((n_agents, dim))
        p = rng.random(n_agents)
        l = rng.uniform(-1, 1, (n_agents, dim))
        exploit = np.all(np.abs(A) < 1, axis=1)
        Xr = X[rng.integers(n_agents, size=n_agents)]
        enc = best_x - A * np.abs(C * best_x - X)
        srch = Xr - A * np.abs(C * Xr - X)
        spir = np.abs(best_x - X) * np.exp(b * l) * np.cos(2 * np.pi * l) + best_x
        X = np.where((p < 0.5)[:, None], np.where(exploit[:, None], enc, srch), spir)
        X = np.round(np.clip(X, lo, hi), 9)
        sets = decode_batch(ctx, X, K)
        fit = np.round(obj(sets), 9)
        i = int(np.argmax(fit))
        if fit[i] > best_f:
            best_x, best_set, best_f = X[i].copy(), sets[i].copy(), fit[i]
        # SA local search on the current best; accepted only if it improves the best
        S, fS = sa(ctx, obj, best_set, rng, n_iter=sa_steps, t0=T, alpha=sa_alpha)
        T *= sa_alpha ** sa_steps
        if fS > best_f:
            best_set, best_f = np.asarray(S, int).copy(), fS
            best_x = ctx.cand_xy[[cand_index[int(c)] for c in best_set]].ravel().copy()
    return best_set, best_f
