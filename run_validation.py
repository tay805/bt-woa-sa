"""Phase 5: trace-driven NS-3 validation of the Python simulator.

For each method and seed, the Python simulator runs the first R rounds (100-byte data frames,
the 802.15.4 frame-size limit), exports its schedule, and NS-3 replays it over the IEEE 802.15.4
stack. Compared: PDR, share of readings dropped by attackers, transmission attempts per
delivered reading, and the ranking of methods.

Usage: python run_validation.py      env: N_SEEDS (5), ROUNDS (300), NS3_DIR (./ns-3)
"""
import os, sys, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from wsnsim import Config, algorithms
from wsnsim.ns3trace import run_and_export

OUT = os.path.join(HERE, "results", "validation")
NS3 = os.environ.get("NS3_DIR", os.path.join(HERE, "ns-3"))
BIN = os.path.join(NS3, "build", "scratch", "wsn-validate", "ns3.44-wsn-validate-optimized")
SEEDS = range(int(os.environ.get("N_SEEDS", 5)))
ROUNDS = int(os.environ.get("ROUNDS", 300))
METHODS = [("LEACH", {}), ("PSO-C", {}), ("WOA-C", {}), ("BT-WOA-SA", {})]
NS3_SETUPS = {"ns3: own channels": ["--mode=channels"],
              "ns3: own channels + fading": ["--mode=channels", "--fadingM=3"],
              "ns3: shared channel": ["--mode=shared"]}
COLS = ["rounds", "generated", "rx_at_ch", "dropped_mal", "delivered", "tx_attempts", "mac_fail", "delay_ms"]


def ns3(trace, args, seed):
    env = dict(os.environ, LD_LIBRARY_PATH=os.path.join(NS3, "build", "lib"))
    out = subprocess.run([BIN, f"--trace={trace}", f"--seed={seed + 1}"] + args, env=env,
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    return dict(zip(COLS, [float(v) for v in out.split(",")]))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cfg = Config(traffic="periodic", recluster_every=10, data_bits=800)
    rows = []
    for name, kw in METHODS:
        for seed in SEEDS:
            trace = os.path.join(OUT, f"trace_{name}_{seed}.txt")
            py = run_and_export(cfg, seed, algorithms.make(name, **kw), ROUNDS, trace)
            rows.append(dict(method=name, seed=seed, simulator="python", **py))
            for setup, args in NS3_SETUPS.items():
                r = ns3(trace, args, seed)
                rows.append(dict(method=name, seed=seed, simulator=setup, generated=r["generated"],
                                 delivered=r["delivered"], dropped_malicious=r["dropped_mal"],
                                 tx_attempts=r["tx_attempts"], delay_ms=r["delay_ms"], mac_fail=r["mac_fail"]))
            os.remove(trace)
    df = pd.DataFrame(rows)
    df["PDR"] = df.delivered / df.generated
    df["attacker_drop_share"] = df.dropped_malicious / df.generated
    df["attempts_per_delivered"] = df.tx_attempts / df.delivered
    df.to_csv(os.path.join(OUT, "validation_runs.csv"), index=False)
    s = df.groupby(["simulator", "method"])[["PDR", "attacker_drop_share", "attempts_per_delivered", "delay_ms"]].mean()
    s.to_csv(os.path.join(OUT, "validation_summary.csv"))
    pd.set_option("display.width", 200)
    print(s.round(4).to_string())
    print("\nPDR ranking per simulator (best first):")
    for sim_name, g in df.groupby("simulator"):
        order = g.groupby("method").PDR.mean().sort_values(ascending=False)
        print(f"  {sim_name:28s}", " > ".join(f"{m} ({v:.3f})" for m, v in order.items()))
