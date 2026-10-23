"""Export a Python run's schedule so NS-3 can replay it (trace-driven validation).

File format (text, one record per line):
  HEADER N <n> BS <x> <y> DATA_BYTES <b> ROUNDS <R> ATTACK_PERIOD <p> ATTACK_ON <f>
  NODE <id> <x> <y> <malicious 0/1> <drop_prob> <attack_phase>
  ROUND <r>
  CH <id> <id> ...                (cluster heads this round)
  M <member> <ch> ...             (pairs: member id, its CH; direct-to-BS nodes have ch = -1)
  S <id> <id> ...                 (nodes that report this round)
"""
import numpy as np

from .simulation import Simulation


def run_and_export(cfg, seed, algorithm, rounds, path):
    sim = Simulation(cfg, seed, algorithm)
    sim.trace = []
    for r in range(rounds):
        if not sim.step(r):
            break
    with open(path, "w") as f:
        f.write(f"HEADER N {sim.N} BS {cfg.bs_x} {cfg.bs_y} DATA_BYTES {cfg.data_bits // 8} "
                f"ROUNDS {len(sim.trace)} ATTACK_PERIOD {cfg.attack_period} ATTACK_ON {cfg.attack_on_frac}\n")
        for i in range(sim.N):
            f.write(f"NODE {i} {sim.x[i]:.6f} {sim.y[i]:.6f} {int(sim.malicious[i])} "
                    f"{sim.drop_prob[i]:.6f} {int(sim.attack_phase[i])}\n")
        for r, chs, cluster_of, senders in sim.trace:
            f.write(f"ROUND {r}\n")
            f.write("CH " + " ".join(map(str, chs.tolist())) + "\n")
            alive = np.nonzero(sim.death_round > r)[0]
            pairs = [(int(i), int(cluster_of[i])) for i in alive if i not in set(chs.tolist())]
            f.write("M " + " ".join(f"{a} {b}" for a, b in pairs) + "\n")
            f.write("S " + " ".join(map(str, senders.tolist())) + "\n")
    h = {k: np.asarray(v[:len(sim.trace)]) for k, v in sim.hist.items()}
    return dict(rounds=len(sim.trace), generated=int(h["generated"].sum()), delivered=int(h["delivered"].sum()),
                dropped_malicious=int(h["dropped_malicious"].sum()), tx_attempts=int(h["tx_attempts"].sum()),
                delay_ms=float(h["delay_sum"].sum() / max(h["delivered"].sum(), 1) * 1e3))
