"""Multi-seed, multi-algorithm experiment runner (parallel over processes)."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .config import Config
from .simulation import Simulation
from . import algorithms


def _one(args):
    cfg_dict, alg_name, alg_kw, seed = args
    cfg_dict = {k: v for k, v in cfg_dict.items() if k != "d0"}
    cfg = Config(**cfg_dict)
    t = time.time()
    sim = Simulation(cfg, seed, algorithms.make(alg_name, **alg_kw))
    s = sim.run()
    s["runtime_s"] = time.time() - t
    s["label"] = alg_kw.get("label", alg_name)
    ts = {k: np.asarray(v) for k, v in sim.hist.items()}
    return s, ts


def run_experiment(cfg, algs, seeds, out_dir, n_jobs=None, tag="exp"):
    """algs: list of (name, kwargs). Saves <tag>_runs.csv, <tag>_ts.npz, <tag>_config.json."""
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(cfg.as_dict(), a, kw, s) for a, kw in algs for s in seeds]
    n_jobs = n_jobs or os.cpu_count()
    rows, series = [], {}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for s, ts in ex.map(_one, jobs):
            rows.append(s)
            series[(s["label"], s["seed"])] = ts
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"{tag}_runs.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, f"{tag}_ts.npz"),
                        **{f"{lab}|{seed}|{k}": v for (lab, seed), ts in series.items() for k, v in ts.items()})
    with open(os.path.join(out_dir, f"{tag}_config.json"), "w") as f:
        json.dump(dict(config=cfg.as_dict(), algorithms=algs, seeds=list(seeds)), f, indent=2)
    return df, series


def aggregate(df, metrics):
    """mean ± std per algorithm label (censored lifetimes reported separately)."""
    g = df.groupby("label")
    out = pd.DataFrame(index=g.size().index)
    out["runs"] = g.size()
    for m in metrics:
        col = pd.to_numeric(df[m], errors="coerce")
        out[f"{m}_mean"] = col.groupby(df["label"]).mean()
        out[f"{m}_std"] = col.groupby(df["label"]).std()
        if m in ("FND", "HND", "LND"):
            out[f"{m}_censored"] = df[m].isna().groupby(df["label"]).sum()
    return out


def mean_curve(series, label, key, length=None):
    curves = [ts[key] for (lab, _), ts in series.items() if lab == label]
    L = length or max(len(c) for c in curves)
    padded = np.zeros((len(curves), L))
    for i, c in enumerate(curves):
        n = min(len(c), L)
        padded[i, :n] = c[:n]
        padded[i, n:] = c[-1] if len(c) else 0   # run ended: hold last value (0 once all dead)
    return padded.mean(0), padded.std(0)
