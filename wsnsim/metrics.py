"""Run-level metrics computed only from what the simulator actually recorded."""
import numpy as np


def lifetime(alive, N):
    """FND/HND/LND as 1-indexed round numbers; None if not reached (censored)."""
    alive = np.asarray(alive)
    def first(mask):
        idx = np.nonzero(mask)[0]
        return int(idx[0]) + 1 if idx.size else None
    return dict(FND=first(alive < N), HND=first(alive <= N / 2), LND=first(alive == 0))


def summarize(sim):
    h = {k: np.asarray(v) for k, v in sim.hist.items()}
    N = sim.N
    out = dict(algorithm=sim.alg.name, seed=sim.seed, traffic=sim.cfg.traffic,
               n_nodes=N, rounds=sim.rounds_run)
    out.update(lifetime(h["alive"], N))
    gen, dlv = h["generated"].sum(), h["delivered"].sum()
    out["packets_generated"] = int(gen)
    out["report_rate"] = float(gen / max(h["alive"].sum(), 1))   # reports per alive node per round
    out["packets_delivered"] = int(dlv)
    out["PDR"] = float(dlv / gen) if gen else np.nan
    out["dropped_malicious"] = int(h["dropped_malicious"].sum())
    out["avg_delay_ms"] = float(h["delay_sum"].sum() / dlv * 1e3) if dlv else np.nan
    out["mean_energy_var"] = float(h["energy_var"].mean())
    out["mean_energy_cv"] = float(h["energy_cv"].mean())
    # energy balance during the stable period (before the first node dies), as a fraction of E0
    fnd_idx = out["FND"] - 1 if out["FND"] else len(h["alive"])
    out["energy_std_stable"] = float(h["energy_std"][:max(fnd_idx, 1)].mean())
    out["energy_std_stable_honest"] = float(h["energy_std_honest"][:max(fnd_idx, 1)].mean())
    out["residual_at_FND"] = float(h["energy_total"][fnd_idx] / (N * sim.cfg.e_init)) if out["FND"] else np.nan
    out["instability"] = (out["LND"] - out["FND"]) if (out["LND"] and out["FND"]) else np.nan
    out["lost_channel"] = int(h["lost_channel"].sum())
    out["recon_rmse"] = float(np.nanmean(h["recon_rmse"])) if len(h["recon_rmse"]) else np.nan
    out["energy_per_packet_mJ"] = float((N * sim.cfg.e_init - sim.E.sum()) / dlv * 1e3) if dlv else np.nan
    out["control_energy_share"] = float(h["e_control"].sum() / max(h["e_control"].sum() + h["e_data"].sum(), 1e-12))
    ch_rounds = h["n_ch"].sum()
    out["malicious_CH_share"] = float(h["n_mal_ch"].sum() / ch_rounds) if ch_rounds else np.nan
    out["avg_CH_per_round"] = float(h["n_ch"].mean())
    hnd_idx = out["HND"] if out["HND"] else len(h["alive"])
    ch_pre = h["n_ch"][:hnd_idx].sum()
    out["malicious_CH_share_pre_HND"] = float(h["n_mal_ch"][:hnd_idx].sum() / ch_pre) if ch_pre else np.nan
    out["setups"] = int(h["setup"].sum())
    cov = h["coverage"]
    for lvl in (90, 70):
        below = np.nonzero(cov < lvl / 100)[0]
        out[f"COV{lvl}"] = int(below[0]) + 1 if below.size else None   # coverage lifetime (rounds)
    rr = h["recon_rmse"][:hnd_idx]
    out["recon_rmse_pre_HND"] = float(np.nanmean(rr)) if rr.size and np.isfinite(rr).any() else np.nan
    # events: only those that started while the network ran and were observable by >=1 node
    env = sim.env
    # observable = at least one node that can sense it is still alive when it starts
    alive_at_start = sim.death_round[None, :] > env.ev_start[:, None]
    elig = (env.ev_start < sim.rounds_run) & (env.detectable & alive_at_start).any(axis=1)
    det = sim.ev_detected_round[elig]
    out["events_eligible"] = int(elig.sum())
    out["missed_event_rate"] = float((det < 0).mean()) if elig.any() else np.nan
    starts = env.ev_start[elig]
    for nm, lim in (("FND", out["FND"]), ("HND", out["HND"])):
        msk = starts < (lim if lim else sim.rounds_run)
        out[f"missed_pre_{nm}"] = float((det[msk] < 0).mean()) if msk.any() else np.nan
    lat = det[det >= 0] - env.ev_start[elig][det >= 0]
    out["event_latency_rounds"] = float(lat.mean()) if lat.size else np.nan
    out.update(sim.trust.detection_stats(sim.malicious))
    out.update(getattr(sim, "trace_info", {}))
    if hasattr(sim.alg, "gate_stats"):
        out.update(sim.alg.gate_stats(sim))
    return out


def timeseries(sim):
    return {k: np.asarray(v) for k, v in sim.hist.items()}
