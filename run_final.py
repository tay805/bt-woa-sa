"""Final experiments: resumable batch runner.

Every (experiment, scenario, method, seed) is one job; each finished job is appended to
results/final/rows/<experiment>.jsonl immediately, so the runner can be stopped and restarted
at any time and continues where it left off.

Usage:
  python run_final.py --budget 1550            # run pending jobs for ~1550 s, then stop
  python run_final.py --status                 # show progress per experiment
  python run_final.py --only E1_periodic       # restrict to one scenario group (prefix match)

Experiments
  E0  re-clustering interval 1/5/10/20/50 rounds, LEACH, event-driven, 30 seeds
  E1  main comparison, N=100, periodic and event-driven, 30 seeds, all methods + ablations
  E2  scalability, N = 200/300/500 in the same 100x100 m field, periodic, 30 seeds
  E3  attacker ratio 0/10/30/40 % (20 % = E1), periodic, 30 seeds
  E4  base station at the field centre (50,50), periodic, 30 seeds
  E5  low-rate and on-off attackers (grey-hole = E1), periodic 30 seeds, event 10 seeds
  E6  Intel Lab trace, far sink, event-driven, 10 seeds (needs INTEL_DIR=<folder with data.txt(.gz)>)
  E7  sensitivity of the Bayesian gate to its prior and threshold, periodic, 30 seeds
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wsnsim import Config, Simulation, algorithms  # noqa: E402

OUT = os.path.join(HERE, "results", "final")

M7 = [("LEACH", {}), ("DEEC", {}), ("LEACH-C", {}), ("PSO-C", {}), ("WOA-C", {}),
      ("BT-WOA-SA", {"label": "CTRF-style gate", "gate": "mean"}),
      ("BT-WOA-SA", {})]
IW = ("IWOA-SA", {})                     # Iwendi-style WOA-SA (added 2026-10-18)
DEEC_TRUE = ("DEEC", {"label": "DEEC (true avg)", "true_average": True})
E1_EXTRA = [("DEEC", {"label": "DEEC (true avg)", "true_average": True}),
            ("BT-WOA-SA", {"label": "BT: without trust gate", "use_trust": False}),
            ("BT-WOA-SA", {"label": "BT: WOA only (no SA)", "use_sa": False}),
            ("BT-WOA-SA", {"label": "BT: SA only (no WOA)", "use_woa": False}),
            ("BT-WOA-SA", {"label": "BT + energy trigger", "use_trigger": True, "max_interval": 10})]
GATES = [("WOA-C", {}), ("BT-WOA-SA", {"label": "CTRF-style gate", "gate": "mean"}), ("BT-WOA-SA", {})]

BASE = dict(recluster_every=10)


def scenarios():
    """(experiment, scenario name, config kwargs, algorithms, seeds)."""
    S = []
    # E0: re-clustering interval (system parameter shared by all methods), LEACH, event-driven
    for rc in (1, 5, 10, 20, 50):
        S.append(("E0", f"E0_recluster{rc}", dict(traffic="event", recluster_every=rc), [("LEACH", {})], range(30)))
    for tr in ("periodic", "event"):
        S.append(("E1", f"E1_{tr}", dict(BASE, traffic=tr), M7 + [IW] + E1_EXTRA, range(30)))
    S.append(("E4", "E4_bs_centre", dict(BASE, traffic="periodic", bs_x=50.0, bs_y=50.0), M7 + [IW], range(30)))
    for rho in (0.0, 0.1, 0.3, 0.4):
        S.append(("E3", f"E3_mal{int(rho * 100)}", dict(BASE, traffic="periodic", malicious_ratio=rho),
                  [("LEACH-C", {}), ("PSO-C", {}), ("BT-WOA-SA", {"label": "CTRF-style gate", "gate": "mean"}),
                   ("BT-WOA-SA", {}), IW], range(30)))
    for tr, seeds in (("periodic", range(30)), ("event", range(10))):
        S.append(("E5", f"E5_lowrate_{tr}", dict(BASE, traffic=tr, drop_low=0.05, drop_high=0.15), GATES, seeds))
        S.append(("E5", f"E5_onoff_{tr}", dict(BASE, traffic=tr, attack_period=40, attack_on_frac=0.5), GATES, seeds))
    for N in (200, 300, 500):
        S.append(("E2", f"E2_N{N}", dict(BASE, traffic="periodic", n_nodes=N),
                  [("LEACH", {}), ("LEACH-C", {}), ("PSO-C", {}), ("WOA-C", {}),
                   ("BT-WOA-SA", {"label": "CTRF-style gate", "gate": "mean"}), ("BT-WOA-SA", {}),
                   ("BT-WOA-SA", {"label": "BT: SA only (no WOA)", "use_woa": False}), IW], range(30)))
    # E7: sensitivity of the Bayesian gate to its prior Beta(a0, 1) and exclusion threshold
    S.append(("E7", "E7_prior_sensitivity", dict(BASE, traffic="periodic"),
              [("BT-WOA-SA", {"label": f"Beta({a:g},1), z_max={t}", "susp_prior": (float(a), 1.0), "susp_max": t})
               for a in (1, 4, 9, 19) for t in (0.5, 0.7)], range(30)))
    # E6: Intel Lab replay (only when the data folder is given), far sink, event-driven, 10 seeds
    intel = os.environ.get("INTEL_DIR")
    if intel:
        S.append(("E6", "E6_intel_far_event", dict(BASE, env="intel", intel_dir=intel, traffic="event",
                                                    bs_x=20.0, bs_y=131.0, theta0=0.3), M7 + [IW, DEEC_TRUE], range(10)))
    # event-driven E1 last: longest jobs
    S.sort(key=lambda s: s[1] == "E1_event")
    return S


def jobs():
    for exp, scen, ckw, algs, seeds in scenarios():
        for seed in seeds:
            for name, kw in algs:
                yield exp, scen, ckw, name, kw, seed


def job_key(scen, name, kw, seed):
    return f"{scen}|{kw.get('label', name)}|{seed}"


def run_job(args):
    exp, scen, ckw, name, kw, seed = args
    t = time.time()
    sim = Simulation(Config(**ckw), seed, algorithms.make(name, **kw))
    s = sim.run()
    s.update(experiment=exp, scenario=scen, label=kw.get("label", name), runtime_s=time.time() - t)
    s = {k: (None if isinstance(v, float) and not np.isfinite(v) else
             (v.item() if hasattr(v, "item") else v)) for k, v in s.items()}
    ts = None
    if exp == "E1":       # keep curves for figures (alive nodes, coverage, residual energy)
        h = sim.hist
        ts = dict(alive=np.asarray(h["alive"], np.int16), coverage=np.asarray(h["coverage"], np.float16),
                  energy=np.asarray(h["energy_total"], np.float32))
    return exp, job_key(scen, name, kw, seed), s, ts


def done_keys():
    keys = set()
    rows = os.path.join(OUT, "rows")
    if os.path.isdir(rows):
        for f in os.listdir(rows):
            with open(os.path.join(rows, f)) as fh:
                for line in fh:
                    keys.add(json.loads(line)["_key"])
    return keys


def status():
    done = done_keys()
    total = {}
    for exp, scen, ckw, name, kw, seed in jobs():
        t = total.setdefault(scen, [0, 0])
        t[1] += 1
        t[0] += job_key(scen, name, kw, seed) in done
    for scen, (d, n) in total.items():
        print(f"  {scen:22s} {d:5d} / {n}")
    print(f"  {'TOTAL':22s} {sum(v[0] for v in total.values()):5d} / {sum(v[1] for v in total.values())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=1550)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--only", default="")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(OUT, "rows"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "ts"), exist_ok=True)
    if a.status:
        status()
        return
    done = done_keys()
    pending = [j for j in jobs() if job_key(j[1], j[3], j[4], j[5]) not in done and j[1].startswith(a.only)]
    print(f"{len(pending)} pending jobs", flush=True)
    t0, finished = time.time(), 0
    it = iter(pending)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        running = set()
        while True:
            while len(running) < a.workers and time.time() - t0 < a.budget:
                j = next(it, None)
                if j is None:
                    break
                running.add(ex.submit(run_job, j))
            if not running:
                break
            fin, running = wait(running, return_when=FIRST_COMPLETED)
            for f in fin:
                exp, key, s, ts = f.result()
                s["_key"] = key
                with open(os.path.join(OUT, "rows", f"{exp}.jsonl"), "a") as fh:
                    fh.write(json.dumps(s) + "\n")
                if ts is not None:
                    np.savez_compressed(os.path.join(OUT, "ts", key.replace("|", "__").replace(" ", "_")
                                                     .replace("(", "").replace(")", "").replace(":", "") + ".npz"), **ts)
                finished += 1
    print(f"finished {finished} jobs in {time.time() - t0:.0f} s", flush=True)
    status()


if __name__ == "__main__":
    main()
