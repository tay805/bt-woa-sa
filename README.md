# BT-WOA-SA: Bayesian trust-aware metaheuristic clustering for wireless sensor networks

This repository contains the complete code, raw results and analysis scripts for the paper

> T. A. Teli, "A Bayesian Trust-Aware Metaheuristic Clustering Framework for Secure and
> Energy-Efficient IoT-Enabled Wireless Sensor Networks".

Every number, table and figure in the paper can be regenerated from this repository.

## What the method does

In a clustered sensor network, a cluster head (CH) that silently drops part of its members'
packets (a grey-hole attack) wastes the energy those members spent on sensing and transmitting.
BT-WOA-SA lets the base station (BS) keep a Beta reputation for every node that has served as
CH, using only the sequence numbers of the packets it receives (no extra messages). Before each
re-clustering, a node is excluded from CH candidacy only if the posterior probability that it
forwards less than 85 % of its traffic is above 0.5. An optimistic prior, Beta(9, 1), keeps nodes
without evidence eligible. The CHs are then chosen among the trusted nodes by a whale
optimization algorithm (WOA) refined with simulated annealing (SA), on the energy-distance cost
of Latiff et al. (2007).

## Repository layout

```
wsnsim/                    the simulator (pure Python)
  config.py                every simulation parameter, in one place
  radio.py                 first-order radio energy model
  environment.py           synthetic spatially correlated sensing field
  intel.py                 loader and cleaner for the Intel Berkeley Lab trace
  trust.py                 BS-side Beta reputation from sequence-number evidence
  simulation.py            round-based simulator (set-up, TDMA steady state, ACK and retransmissions)
  metrics.py               lifetime, coverage, delivery, energy, security and fidelity metrics
  ns3trace.py              exports a run's schedule for the ns-3 replay
  runner.py                helper for multi-seed runs
  algorithms/
    base.py                interface; direct transmission
    leach.py               LEACH
    baselines.py           LEACH-C, DEEC, PSO-C, WOA-C, IWOA-SA (Iwendi-style WOA-SA)
    search.py              shared WOA, PSO, SA and WOA-SA search code
    ewoasa.py              BT-WOA-SA (proposed) and its ablations, including the CTRF-style gate
run_final.py               runs all experiments E0 to E7 (resumable)
analyze_final.py           statistics: means, confidence intervals, Wilcoxon + Holm, A12, Friedman
make_tables.py             LaTeX tables used in the paper
make_figures.py            figures used in the paper
run_validation.py          trace-driven ns-3 validation (needs ns-3, see ns3/README.md)
ns3/wsn-validate.cc        the ns-3 replay program (IEEE 802.15.4)
notebooks/                 Google Colab notebook that runs everything
results/final/rows/        raw results, one JSON line per simulation run
results/final/ts/          per-round curves of experiment E1 (used by the alive-nodes figure)
results/final/stats/       statistics written by analyze_final.py
results/final/tables/      LaTeX tables written by make_tables.py
results/final/figures/     figures written by make_figures.py (PDF and PNG)
results/validation/        ns-3 validation results
docs/REPRODUCE.md          which command produces which table and figure of the paper
```

## Installation

Python 3.10 or newer. The results in the paper were produced with Python 3.13 and the package
versions pinned in `requirements.txt`.

```bash
git clone <repository URL>
cd bt-woa-sa
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

Run one simulation of the proposed method and of PSO-C on the same network (same seed, so the
same deployment, attackers and sensed field):

```python
from wsnsim import Config, Simulation, algorithms

cfg = Config(traffic="periodic", recluster_every=10)      # paper defaults, 20 % grey-hole attackers
for name in ("BT-WOA-SA", "PSO-C"):
    s = Simulation(cfg, seed=0, algorithm=algorithms.make(name)).run()
    print(name, "FND", s["FND"], "delivered", s["packets_delivered"], "PDR", round(s["PDR"], 3),
          "attacker CH share", round(s["malicious_CH_share_pre_HND"], 3))
```

Available method names: `LEACH`, `DEEC`, `LEACH-C`, `PSO-C`, `WOA-C`, `IWOA-SA`, `BT-WOA-SA`, `Direct`.
Variants are created with keyword arguments, for example
`algorithms.make("BT-WOA-SA", label="CTRF-style gate", gate="mean")` or
`algorithms.make("DEEC", label="DEEC (true avg)", true_average=True)`.
All parameters are fields of `Config` and can be changed with `Config(n_nodes=200, malicious_ratio=0.3)`.

## Reproducing the paper

```bash
python run_final.py --status            # shows how many runs of each experiment are finished
python run_final.py --budget 36000      # runs the pending experiments for up to 10 hours
python analyze_final.py                 # statistics
python make_tables.py                   # LaTeX tables
python make_figures.py                  # figures
```

`run_final.py` writes every finished run immediately to `results/final/rows/<experiment>.jsonl`
and skips runs that are already there, so it can be stopped and restarted at any time. Use
`--only E3` to run a single experiment and `--workers 4` to set the number of parallel
processes. The repository already contains all results, so the analysis scripts can be run
directly. Rerunning all synthetic experiments takes about 10 CPU hours.

Experiment E6 replays the Intel Berkeley Lab trace. Download `data.txt.gz` and `mote_locs.txt`
from http://db.csail.mit.edu/labdata/labdata.html into one folder and run

```bash
INTEL_DIR=/path/to/that/folder python run_final.py --only E6
```

The ns-3 validation is described in `ns3/README.md`. `docs/REPRODUCE.md` maps every table and
figure of the paper to the command and result file that produce it.

## Experiments

| Id | Purpose | Seeds |
| :- | :- | -: |
| E0 | re-clustering interval 1, 5, 10, 20, 50 rounds (LEACH, event-driven) | 30 |
| E1 | main comparison, N = 100, periodic and event-driven, all methods and ablations | 30 |
| E2 | scalability, N = 200, 300, 500 in the same 100 m x 100 m field | 30 |
| E3 | attacker fraction 0, 10, 30, 40 % (20 % is E1) | 30 |
| E4 | base station at the centre of the field | 30 |
| E5 | low-rate and on-off attackers | 30 (periodic), 10 (event-driven) |
| E6 | Intel Berkeley Lab trace, 43 motes, far sink, event-driven | 10 |
| E7 | sensitivity of the Bayesian gate to its prior and threshold | 30 |

## Reproducibility

* A seed fixes the deployment, the attackers, the sensed field and the channel. They come from
  independent random streams, so every method is evaluated on exactly the same conditions for
  the same seed, and comparisons are paired.
* Distances, costs and optimiser positions are rounded to 1e-9 before any comparison, so results
  are bit-identical across machines. We checked this between NumPy 2.5.3 on an AVX-512 Xeon and
  NumPy 2.1.3 on Google Colab (AVX2): all metrics matched exactly.
* `results/final/rows/*.jsonl` stores every metric of every run, including the run time and the
  seed, so all statistics can be recomputed without rerunning the simulations.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT License, see `LICENSE`. The Intel Lab data is not included and is subject to its own terms.
