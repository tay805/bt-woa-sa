"""Paper figures from the final experiments (results/final) and the NS-3 validation.

Colours: validated categorical palette (colourblind-safe, fixed per method across all
figures); every series also has its own marker/line style so figures read in greyscale.
Output: results/final/figures/*.pdf (vector, for LaTeX) and *.png (preview).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "final", "figures")
STATS = os.path.join(HERE, "results", "final", "stats")

# method -> (colour, marker, linestyle); fixed across every figure
STYLE = {
    "BT-WOA-SA":       ("#2a78d6", "o", "-"),
    "PSO-C":           ("#eb6834", "s", "--"),
    "LEACH-C":         ("#1baf7a", "^", "-."),
    "WOA-C":           ("#eda100", "D", ":"),
    "LEACH":           ("#e87ba4", "v", "--"),
    "DEEC":            ("#008300", "P", "-."),
    "CTRF-style gate": ("#6250d6", "X", ":"),
    "IWOA-SA":         ("#8a5a2b", "*", "--"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def load_rows():
    rows = []
    for f in glob.glob(os.path.join(HERE, "results", "final", "rows", "*.jsonl")):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh]
    return pd.DataFrame(rows)


def ts_files(scen, label):
    tag = f"{scen}__{label}__".replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    return sorted(glob.glob(os.path.join(HERE, "results", "final", "ts", tag + "*.npz")))


def fig_alive():
    """Alive nodes vs round, E1 periodic and event (mean over 30 seeds)."""
    labs = ["LEACH", "DEEC", "LEACH-C", "PSO-C", "CTRF-style gate", "BT-WOA-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, scen, title in zip(axes, ["E1_periodic", "E1_event"], ["(a) Periodic reporting", "(b) Event-driven reporting"]):
        for lab in labs:
            curves = [np.load(f)["alive"].astype(float) for f in ts_files(scen, lab)]
            if not curves:
                continue
            L = max(len(c) for c in curves)
            M = np.array([np.pad(c, (0, L - len(c)), constant_values=c[-1]) for c in curves]).mean(0)
            c, mk, ls = STYLE[lab]
            ax.plot(M, color=c, ls=ls, lw=2.2 if lab == "BT-WOA-SA" else 1.4, label=lab)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Round")
        ax.set_ylim(0, 102)
    axes[0].set_ylabel("Alive nodes")
    axes[1].legend(loc="upper right", ncol=1)
    fig.tight_layout()
    save(fig, "fig_alive_nodes")


# label placement for the trade-off plot (overlapping centralised baselines share one label)
LABEL_POS = {("E1_periodic", "PSO-C"): ("LEACH-C / PSO-C / WOA-C", (-8, -14)),
             ("E1_periodic", "WOA-C"): ("", (0, 0)), ("E1_periodic", "LEACH-C"): ("", (0, 0)),
             ("E1_event", "PSO-C"): ("PSO-C / WOA-C", (-8, 6)), ("E1_event", "WOA-C"): ("", (0, 0)),
             ("E1_event", "LEACH-C"): ("LEACH-C", (-8, -12)),
             ("E1_periodic", "BT-WOA-SA"): ("BT-WOA-SA", (-8, 6)), ("E1_event", "BT-WOA-SA"): ("BT-WOA-SA", (-8, 6)),
             ("E1_periodic", "IWOA-SA"): ("IWOA-SA", (6, -12))}


def fig_tradeoff(df):
    """Security-lifetime trade-off: FND vs delivered readings, mean +/- 95% CI."""
    labs = ["LEACH", "DEEC", "LEACH-C", "PSO-C", "WOA-C", "IWOA-SA", "CTRF-style gate", "BT-WOA-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, scen, title in zip(axes, ["E1_periodic", "E1_event"], ["(a) Periodic reporting", "(b) Event-driven reporting"]):
        g = df[df.scenario == scen]
        for lab in labs:
            v = g[g.label == lab]
            if v.empty:
                continue
            x, y = v.FND.astype(float), v.packets_delivered.astype(float) / 1e3
            ci = lambda s: 1.96 * s.std(ddof=1) / np.sqrt(len(s))
            c, mk, _ = STYLE[lab]
            ax.errorbar(x.mean(), y.mean(), xerr=ci(x), yerr=ci(y), fmt=mk, color=c, ms=7 if lab == "BT-WOA-SA" else (9 if mk == "*" else 6),
                        mec="white", mew=0.8, elinewidth=1, capsize=0, label=lab)
            txt, off = LABEL_POS.get((scen, lab), (lab, (5, 4)))
            if txt:
                ax.annotate(txt, (x.mean(), y.mean()), xytext=off, textcoords="offset points", fontsize=7.5,
                            color=INK, ha="right" if off[0] < 0 else "left")
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("First node dies (rounds)")
    axes[0].set_ylabel("Readings delivered (thousands)")
    fig.tight_layout()
    save(fig, "fig_tradeoff_fnd_vs_delivered")


def fig_attackers(df):
    """Attacker share sweep: delivered readings and PDR."""
    scen = {0: "E3_mal0", 10: "E3_mal10", 20: "E1_periodic", 30: "E3_mal30", 40: "E3_mal40"}
    labs = ["LEACH-C", "PSO-C", "IWOA-SA", "CTRF-style gate", "BT-WOA-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for lab in labs:
        xs, d, p = [], [], []
        for rho, s in scen.items():
            v = df[(df.scenario == s) & (df.label == lab)]
            if v.empty:
                continue
            xs.append(rho); d.append(v.packets_delivered.mean() / 1e3); p.append(v.PDR.mean())
        c, mk, ls = STYLE[lab]
        lw = 2.2 if lab == "BT-WOA-SA" else 1.4
        axes[0].plot(xs, d, color=c, marker=mk, ls=ls, lw=lw, ms=8 if mk == "*" else 5, label=lab)
        axes[1].plot(xs, p, color=c, marker=mk, ls=ls, lw=lw, ms=5, label=lab)
    axes[0].set_title("(a) Readings delivered", loc="left", color=INK)
    axes[0].set_ylabel("Thousands")
    axes[1].set_title("(b) Packet delivery ratio", loc="left", color=INK)
    for ax in axes:
        ax.set_xlabel("Malicious nodes (%)")
        ax.set_xticks([0, 10, 20, 30, 40])
    axes[1].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig_attacker_share")


def fig_scalability(df):
    """Delivered readings relative to PSO-C, and FND, for N = 100..500."""
    scen = {100: "E1_periodic", 200: "E2_N200", 300: "E2_N300", 500: "E2_N500"}
    labs = ["LEACH-C", "WOA-C", "IWOA-SA", "CTRF-style gate", "BT-WOA-SA"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for lab in labs + ["PSO-C"]:
        xs, rel, fnd = [], [], []
        for N, s in scen.items():
            g = df[df.scenario == s]
            v, ref = g[g.label == lab], g[g.label == "PSO-C"]
            if v.empty or ref.empty:
                continue
            xs.append(N)
            rel.append(100 * (v.packets_delivered.mean() / ref.packets_delivered.mean() - 1))
            fnd.append(v.FND.astype(float).mean())
        c, mk, ls = STYLE[lab]
        lw = 2.2 if lab == "BT-WOA-SA" else 1.4
        if lab != "PSO-C":
            axes[0].plot(xs, rel, color=c, marker=mk, ls=ls, lw=lw, ms=5, label=lab)
        axes[1].plot(xs, fnd, color=c, marker=mk, ls=ls, lw=lw, ms=5, label=lab)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].set_title("(a) Readings delivered vs PSO-C", loc="left", color=INK)
    axes[0].set_ylabel("Difference (%)")
    axes[1].set_title("(b) First node dies", loc="left", color=INK)
    axes[1].set_ylabel("Rounds")
    for ax in axes:
        ax.set_xlabel("Number of nodes")
        ax.set_xticks([100, 200, 300, 500])
    axes[1].legend(loc="center right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_scalability")


def fig_ns3():
    """PDR in the Python simulator vs ns-3 (trace-driven replay)."""
    f = os.path.join(HERE, "results", "validation", "validation_summary.csv")
    if not os.path.exists(f):
        return
    s = pd.read_csv(f)
    methods = ["LEACH", "PSO-C", "WOA-C", "BT-WOA-SA"]
    sims = [("python", "Python simulator", "#52514e", None),
            ("ns3: own channels", "ns-3, per-cluster channels", "#b6b5ad", None),
            ("ns3: own channels + fading", "ns-3, + Nakagami fading", "#ffffff", "////")]
    # dot plot (no bars: PDR differences are small and bars would need a zero baseline)
    sims = [("python", "Python simulator", "o", INK, 7),
            ("ns3: own channels", "ns-3, per-cluster channels", "s", MUTED, 6),
            ("ns3: own channels + fading", "ns-3, + Nakagami fading", "^", "#9a998f", 6)]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    for k, (sim, name, mk, col, ms) in enumerate(sims):
        vals = [s[(s.simulator == sim) & (s.method == m)].PDR.mean() for m in methods]
        y = np.arange(len(methods)) + (k - 1) * 0.18
        ax.plot(vals, y, mk, color=col, ms=ms, mfc=col if k == 0 else "white", mew=1.2, ls="none", label=name)
    ax.set_yticks(np.arange(len(methods)), methods)
    ax.set_xlabel("Packet delivery ratio")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_ns3_validation")


def main():
    df = load_rows()
    fig_alive()
    fig_tradeoff(df)
    fig_attackers(df)
    fig_scalability(df)
    fig_ns3()
    print("figures ->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
