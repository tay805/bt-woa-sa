"""Round-based clustered-WSN simulator.

Each round:
  1. Apply last round's trust observations.
  2. Set-up phase (every `recluster_every` rounds, or if a CH died):
     algorithm selects CHs -> nodes join nearest alive CH -> control
     energy is charged (ADV broadcast, JOIN, TDMA schedule).
  3. Steady-state phase: environment produces readings, nodes decide to
     report (periodic or event-driven), members transmit to their CH in
     TDMA slots, CH (possibly malicious) forwards or drops, aggregates and
     transmits one packet to the BS. Nodes with no CH send directly to BS.
     Every data hop can fail with probability `per_base`.
  4. Per-round metrics are recorded from the packets actually sent/received.

All randomness comes from independent streams spawned from one seed, so the
deployment, adversary and environment are identical across algorithms.
"""
import numpy as np

from .radio import tx_energy, rx_energy, agg_energy
from .environment import Environment
from .trust import TrustManager

ROUND_KEYS = ["setup", "alive", "energy_total", "energy_var", "energy_cv", "energy_std", "energy_std_honest", "generated", "delivered",
              "dropped_malicious", "lost_channel", "lost_other", "delay_sum", "n_ch",
              "n_mal_ch", "e_control", "e_data", "recon_rmse", "tx_attempts", "coverage"]


class Simulation:
    def __init__(self, cfg, seed, algorithm):
        self.cfg = cfg
        self.seed = seed
        ss = np.random.SeedSequence(seed)
        s_dep, s_env, s_adv, s_net, s_alg = ss.spawn(5)

        if cfg.env == "intel":
            from .intel import load_intel, TraceEnvironment
            tr = load_intel(cfg.intel_dir, cfg.intel_epoch_start, cfg.max_rounds,
                            t_min=cfg.intel_t_min, t_max=cfg.intel_t_max, v_min=cfg.intel_v_min,
                            min_coverage=cfg.intel_min_coverage, max_gap=cfg.intel_max_gap,
                            auto_window=cfg.intel_auto_window)
            self.x, self.y = tr["x"], tr["y"]
            self.env = TraceEnvironment(tr["V"], tr["valid"])
            self.trace_info = {k: tr[k] for k in ("epoch_start", "raw_valid_frac", "n_dropped")}
        else:
            rng_dep = np.random.default_rng(s_dep)
            self.x = rng_dep.uniform(0, cfg.area_x, cfg.n_nodes)
            self.y = rng_dep.uniform(0, cfg.area_y, cfg.n_nodes)
            self.env = Environment(cfg, self.x, self.y, np.random.default_rng(s_env))
        N = len(self.x)
        self.N = N
        # rounded to 1e-9 m so that last-bit differences between numpy builds/CPUs
        # can never change a nearest-CH decision (cross-machine reproducibility)
        self.D = np.round(np.hypot(self.x[:, None] - self.x[None, :], self.y[:, None] - self.y[None, :]), 9)
        self.d_bs = np.round(np.hypot(self.x - cfg.bs_x, self.y - cfg.bs_y), 9)

        rng_adv = np.random.default_rng(s_adv)
        self.malicious = np.zeros(N, bool)
        n_mal = int(round(cfg.malicious_ratio * N))
        if n_mal:
            self.malicious[rng_adv.choice(N, n_mal, replace=False)] = True
        self.drop_prob = np.where(self.malicious, rng_adv.uniform(cfg.drop_low, cfg.drop_high, N), 0.0)
        self.attack_phase = rng_adv.integers(0, max(cfg.attack_period, 1), N)

        self.rng_net = np.random.default_rng(s_net)
        self.rng_alg = np.random.default_rng(s_alg)

        self.E = np.full(N, cfg.e_init, dtype=float)
        self.trust = TrustManager(N, cfg)
        self.ch_count = np.zeros(N, int)
        self.last_ch_round = np.full(N, -10 ** 9)
        self.last_sent_val = np.zeros(N)
        self.last_sent_round = np.full(N, -10 ** 9)
        self.bs_est = None                     # BS's latest known value per node

        self.chs = np.array([], int)
        self.cluster_of = np.full(N, -1)
        self.setup_idx = 0
        self.t_pkt = cfg.data_bits / cfg.data_rate_bps

        self.alg = algorithm
        self.alg.reset(self)

        self.ev_detected_round = np.full(len(self.env.ev_start), -1)
        self.death_round = np.full(N, 10 ** 9)
        # sensing coverage: fraction of a regular grid over the deployment area within
        # sense_radius of at least one alive node
        g = cfg.coverage_grid
        gx = np.linspace(self.x.min(), self.x.max(), g) if cfg.env == "intel" else (np.arange(g) + 0.5) * cfg.area_x / g
        gy = np.linspace(self.y.min(), self.y.max(), g) if cfg.env == "intel" else (np.arange(g) + 0.5) * cfg.area_y / g
        PX, PY = np.meshgrid(gx, gy)
        self.cover = np.hypot(PX.ravel()[:, None] - self.x[None, :], PY.ravel()[:, None] - self.y[None, :]) <= cfg.sense_radius
        reachable = self.cover.any(axis=1)
        self.cover = self.cover[reachable]            # only points coverable at deployment count
        self.hist = {k: [] for k in ROUND_KEYS}
        self.trace = None                      # set to [] to record the schedule for NS-3 replay
        self.rounds_run = 0

    # helpers
    @property
    def alive(self):
        return self.E > 0

    def _spend(self, idx, e):
        """Deduct energy; returns mask of nodes that could afford it (others die)."""
        idx = np.atleast_1d(idx)
        e = np.broadcast_to(np.asarray(e, float), idx.shape)
        ok = self.E[idx] >= e
        self.E[idx] = np.where(ok, self.E[idx] - e, 0.0)
        return ok

    def _attempts(self, n):
        """Unicast with ACK + up to max_retx retransmissions.
        Returns (attempts used, delivered?) per frame; energy is charged per attempt."""
        if self.cfg.per_base <= 0 or n == 0:
            return np.ones(n, int), np.ones(n, bool)
        g = self.rng_net.geometric(1 - self.cfg.per_base, n)
        cap = 1 + self.cfg.max_retx
        return np.minimum(g, cap), g <= cap

    # set-up
    def _setup(self, r):
        cfg = self.cfg
        alive_idx = np.nonzero(self.alive)[0]
        chs = np.asarray(self.alg.select(self, r, self.setup_idx), dtype=int)
        chs = np.unique(chs[self.E[chs] > 0]) if chs.size else chs
        self.setup_idx += 1
        self.chs = chs
        self.cluster_of[:] = -1
        if chs.size == 0:
            return 0.0
        self.ch_count[chs] += 1
        self.last_ch_round[chs] = r
        members = np.setdiff1d(alive_idx, chs)
        if members.size:
            self.cluster_of[members] = chs[np.argmin(self.D[np.ix_(members, chs)], axis=1)]
        self.cluster_of[chs] = chs

        before = self.E.sum()
        if getattr(self.alg, "centralized", False):          # BS broadcasts CH list
            self._spend(alive_idx, rx_energy(cfg, cfg.ctrl_bits))
        for h in chs:                                         # ADV broadcast
            m = np.nonzero(self.cluster_of == h)[0]
            m = m[m != h]
            self._spend([h], tx_energy(cfg, cfg.ctrl_bits, self.D[h, m].max() if m.size else 0.0))
        if members.size:
            self._spend(members, chs.size * rx_energy(cfg, cfg.ctrl_bits))
        for h in chs:                                         # JOIN + TDMA schedule
            m = np.nonzero(self.cluster_of == h)[0]
            m = m[(m != h) & (self.E[m] > 0)]
            if not m.size:
                continue
            self._spend(m, tx_energy(cfg, cfg.ctrl_bits, self.D[h, m]))
            self._spend([h], m.size * rx_energy(cfg, cfg.ctrl_bits))
            self._spend([h], tx_energy(cfg, cfg.ctrl_bits, self.D[h, m].max()))
            self._spend(m, rx_energy(cfg, cfg.ctrl_bits))
        if hasattr(self.alg, "on_setup_done"):
            self.alg.on_setup_done(self)
        return before - self.E.sum()

    # steady state
    def _report_mask(self, r, vals):
        cfg = self.cfg
        alive = self.alive
        if cfg.traffic == "periodic":
            return alive.copy()
        theta = cfg.theta0 * (1 + cfg.theta_lambda * (1 - self.E / cfg.e_init))
        change = np.abs(vals - self.last_sent_val) > theta
        hb = (r - self.last_sent_round) >= cfg.heartbeat
        return alive & (change | hb)

    def _mark_detected(self, r, nodes, active_ev):
        if active_ev.size == 0 or len(nodes) == 0:
            return
        hit = self.env.detectable[np.ix_(active_ev, np.asarray(nodes))].any(axis=1)
        new = active_ev[hit & (self.ev_detected_round[active_ev] < 0)]
        self.ev_detected_round[new] = r

    def step(self, r):
        cfg = self.cfg
        if not self.alive.any():
            return False
        self.trust.apply_pending()

        trig = self.alg.need_recluster(self, r) if hasattr(self.alg, "need_recluster") else None
        periodic = (r % cfg.recluster_every == 0) if trig is None else trig
        need_setup = periodic or self.chs.size == 0 or (self.E[self.chs] <= 0).any()
        e_ctrl = self._setup(r) if need_setup else 0.0

        vals, active_ev = self.env.readings(r)
        if self.bs_est is None:                 # bootstrap: BS knows initial readings
            self.bs_est = vals.copy()
            self.last_sent_val = vals.copy()
            self.last_sent_round[:] = r
        send = self._report_mask(r, vals)
        if self.trace is not None:
            self.trace.append((r, self.chs.copy(), self.cluster_of.copy(), np.nonzero(send)[0]))
        self.last_sent_val[send] = vals[send]
        self.last_sent_round[send] = r

        before = self.E.sum()
        S = np.zeros(self.N)
        F = np.zeros(self.N)
        generated = int(send.sum())
        delivered = dropped_mal = lost_ch = lost_other = 0
        attempts = 0
        delay_sum = 0.0
        delivered_nodes = []
        e_rx = rx_energy(cfg, cfg.data_bits)

        # clustered traffic
        for h in self.chs:
            m_all = np.nonzero(self.cluster_of == h)[0]
            m_all = m_all[m_all != h]                       # TDMA order = node id order
            if self.E[h] <= 0:
                lost_other += int(send[m_all].sum())
                continue
            tx_m = m_all[send[m_all]]
            att, got = self._attempts(tx_m.size)
            ok = self._spend(tx_m, att * tx_energy(cfg, cfg.data_bits, self.D[h, tx_m])) if tx_m.size \
                else np.zeros(0, bool)
            lost_other += int((~ok).sum())
            attempts += int(att[ok].sum())
            sent, att, got = tx_m[ok], att[ok], got[ok]
            cum = np.cumsum(att * e_rx)                     # CH listens to every attempt, TDMA order
            k = int(np.searchsorted(cum, self.E[h], side="right"))
            if k:
                self.E[h] -= cum[k - 1]
            if k < sent.size:
                self.E[h] = 0.0
            lost_ch += int((~got[:k]).sum())
            lost_other += int(got[k:].sum())
            received = sent[:k][got[:k]]
            attacking = self.malicious[h] and (cfg.attack_period <= 0 or
                ((r + self.attack_phase[h]) % cfg.attack_period) < cfg.attack_on_frac * cfg.attack_period)
            if attacking and received.size:
                keep = self.rng_net.random(received.size) >= self.drop_prob[h]
            else:
                keep = np.ones(received.size, bool)
            dropped_mal += int((~keep).sum())
            forwarded = received[keep]
            own = bool(send[h]) and self.E[h] > 0
            n_sig = forwarded.size + int(own)
            ok_bs = False
            if n_sig and self.E[h] > 0:
                att_bs, got_bs = self._attempts(1)
                e = agg_energy(cfg, cfg.data_bits, n_sig) + att_bs[0] * tx_energy(cfg, cfg.data_bits, self.d_bs[h])
                if self._spend([h], e)[0]:
                    attempts += int(att_bs[0])
                    if got_bs[0]:
                        ok_bs = True
                    else:
                        lost_ch += n_sig
                else:
                    lost_other += n_sig
            if ok_bs:
                delivered += n_sig
                delivered_nodes.extend(forwarded.tolist())
                if own:
                    delivered_nodes.append(int(h))
                # delay = BS arrival - round start: CH forwards after its TDMA frame of
                # |members| slots, then one more packet time to the BS
                delay_sum += n_sig * (m_all.size + 1) * self.t_pkt
                S[h] = forwarded.size
            # BS view: member packets transmitted vs. member packets delivered.
            # Channel losses and CH failures count against the CH as well; the
            # BS cannot tell them apart from deliberate drops.
            F[h] = sent.size - S[h]

        # nodes without a CH transmit directly to the BS
        direct = np.nonzero(send & (self.cluster_of == -1))[0]
        if direct.size:
            att, gotd = self._attempts(direct.size)
            ok = self._spend(direct, att * tx_energy(cfg, cfg.data_bits, self.d_bs[direct]))
            attempts += int(att[ok].sum())
            lost_other += int((~ok).sum())
            lost_ch += int((ok & ~gotd).sum())
            got = direct[ok & gotd]
            delivered += got.size
            delivered_nodes.extend(got.tolist())
            delay_sum += got.size * self.t_pkt

        self.trust.submit(S, F)
        self._mark_detected(r, delivered_nodes, active_ev)
        if delivered_nodes:
            dn = np.asarray(delivered_nodes, int)
            self.bs_est[dn] = vals[dn]

        alive = self.alive
        newly_dead = (~alive) & (self.death_round == 10 ** 9)
        self.death_round[newly_dead] = r
        Ea = self.E[alive]
        h = self.hist
        h["setup"].append(int(need_setup))
        h["alive"].append(int(alive.sum()))
        h["energy_total"].append(float(self.E.sum()))
        h["energy_var"].append(float(np.var(Ea)) if Ea.size else 0.0)
        h["energy_cv"].append(float(np.std(Ea) / Ea.mean()) if Ea.size else 0.0)
        h["energy_std"].append(float(np.std(Ea) / cfg.e_init) if Ea.size else 0.0)
        Eh = self.E[alive & ~self.malicious]
        h["energy_std_honest"].append(float(np.std(Eh) / cfg.e_init) if Eh.size else 0.0)
        h["generated"].append(generated)
        h["delivered"].append(delivered)
        h["dropped_malicious"].append(dropped_mal)
        h["lost_channel"].append(lost_ch)
        h["lost_other"].append(lost_other)
        h["delay_sum"].append(delay_sum)
        h["n_ch"].append(int(self.chs.size))
        h["n_mal_ch"].append(int(self.malicious[self.chs].sum()) if self.chs.size else 0)
        h["e_control"].append(float(e_ctrl))
        h["e_data"].append(float(before - self.E.sum()))
        mask = alive & self.env.truth_mask(r) if hasattr(self.env, "truth_mask") else alive
        err = (vals - self.bs_est)[mask]
        h["tx_attempts"].append(attempts)
        h["coverage"].append(float(self.cover[:, alive].any(axis=1).mean()) if self.cover.size else 0.0)
        h["recon_rmse"].append(float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan)
        self.rounds_run = r + 1
        return True

    def run(self):
        for r in range(self.cfg.max_rounds):
            if not self.step(r):
                break
        return self.summary()

    def summary(self):
        from .metrics import summarize
        return summarize(self)
