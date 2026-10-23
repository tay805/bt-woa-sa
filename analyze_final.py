"""Statistics and tables for the final experiments.

For every scenario:
  * mean, std and 95 % confidence interval per method;
  * paired Wilcoxon signed-rank tests (same seeds = same deployment, attackers and events)
    of BT-WOA-SA against every other method, Holm-corrected within each (scenario, metric);
  * Vargha-Delaney A12 effect size (probability that BT-WOA-SA beats the other method;
    0.5 = no difference, >= 0.71 or <= 0.29 = large);
  * Friedman test across the main methods.

Usage: python analyze_final.py            -> results/final/stats/*.csv, *.tex, summary.md
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "final")
STATS = os.path.join(OUT, "stats")
PROPOSED = "BT-WOA-SA"
MAIN = ["LEACH", "DEEC", "LEACH-C", "PSO-C", "WOA-C", "IWOA-SA", "CTRF-style gate", "BT-WOA-SA"]

# metric -> (pretty name, higher is better?)
METRICS = {
    "FND": ("FND (rounds)", True),
    "HND": ("HND (rounds)", True),
    "COV90": ("90 % coverage lifetime (rounds)", True),
    "packets_delivered": ("Readings delivered", True),
    "PDR": ("PDR", True),
    "energy_per_packet_mJ": ("Energy per delivered reading (mJ)", False),
    "malicious_CH_share_pre_HND": ("Attacker CH share before HND", False),
    "missed_pre_HND": ("Missed events before HND", False),
    "recon_rmse_pre_HND": ("Reconstruction error before HND", False),
    "energy_std_stable_honest": ("Honest energy std (stable period, /E0)", False),
    "avg_delay_ms": ("Mean delay (ms)", False),
}


def load():
    rows = []
    for f in glob.glob(os.path.join(OUT, "rows", "*.jsonl")):
        with open(f) as fh:
            rows += [json.loads(l) for l in fh]
    return pd.DataFrame(rows)


def a12(x, y):
    """Vargha-Delaney A12: P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x), np.asarray(y)
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return gt + 0.5 * eq


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def summarize(df):
    out = []
    for (scen, lab), g in df.groupby(["scenario", "label"]):
        for m in METRICS:
            if m not in g:
                continue
            v = pd.to_numeric(g[m], errors="coerce").dropna()
            if v.empty:
                continue
            ci = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            out.append(dict(scenario=scen, label=lab, metric=m, n=len(v), mean=v.mean(), std=v.std(ddof=1), ci95=ci))
    return pd.DataFrame(out)


def pairwise(df):
    out = []
    for scen, g in df.groupby("scenario"):
        if PROPOSED not in set(g.label):
            continue
        base = g[g.label == PROPOSED].set_index("seed")
        for m, (_, higher) in METRICS.items():
            if m not in g:
                continue
            res = []
            for lab in sorted(set(g.label) - {PROPOSED}):
                o = g[g.label == lab].set_index("seed")
                common = base.index.intersection(o.index)
                x = pd.to_numeric(base.loc[common, m], errors="coerce")
                y = pd.to_numeric(o.loc[common, m], errors="coerce")
                ok = x.notna() & y.notna()
                x, y = x[ok].values, y[ok].values
                if len(x) < 5:
                    continue
                d = x - y
                p = 1.0 if np.allclose(d, 0) else stats.wilcoxon(x, y, zero_method="zsplit").pvalue
                A = a12(x, y) if higher else a12(y, x)        # A > 0.5 means BT-WOA-SA is better
                res.append(dict(scenario=scen, metric=m, versus=lab, n=len(x), bt_mean=x.mean(), other_mean=y.mean(),
                                rel_diff=(x.mean() - y.mean()) / abs(y.mean()) if y.mean() else np.nan,
                                p=p, A12_bt_better=A))
            if res:
                adj = holm([r["p"] for r in res])
                for r, pa in zip(res, adj):
                    r["p_holm"] = pa
                    r["verdict"] = ("better" if r["A12_bt_better"] > 0.5 else "worse") if pa < 0.05 else "n.s."
                out += res
    return pd.DataFrame(out)


def friedman(df):
    out = []
    for scen, g in df.groupby("scenario"):
        labs = [l for l in MAIN if l in set(g.label)]
        if len(labs) < 3:
            continue
        for m in METRICS:
            if m not in g:
                continue
            piv = g[g.label.isin(labs)].pivot_table(index="seed", columns="label", values=m, aggfunc="first")
            piv = piv.apply(pd.to_numeric, errors="coerce").dropna()
            if len(piv) < 5:
                continue
            try:
                with np.errstate(all="ignore"):      # identical columns (e.g. E7 settings) give NaN
                    stat, p = stats.friedmanchisquare(*[piv[c].values for c in piv.columns])
            except ValueError:
                continue
            higher = METRICS[m][1]
            ranks = piv.rank(axis=1, ascending=not higher).mean()
            out.append(dict(scenario=scen, metric=m, n=len(piv), chi2=stat, p=p,
                            best=ranks.idxmin(), mean_ranks=json.dumps(ranks.round(2).to_dict())))
    return pd.DataFrame(out)


def latex_table(summ, scen, metrics, labels, path):
    s = summ[(summ.scenario == scen) & (summ.metric.isin(metrics))]
    lines = ["\\begin{tabular}{l" + "r" * len(metrics) + "}", "\\toprule",
             "Method & " + " & ".join(METRICS[m][0] for m in metrics) + " \\\\", "\\midrule"]
    for lab in labels:
        cells = []
        for m in metrics:
            r = s[(s.label == lab) & (s.metric == m)]
            if r.empty:
                cells.append("--")
                continue
            mu, sd = r["mean"].iloc[0], r["std"].iloc[0]
            fmt = "{:.3f}" if abs(mu) < 10 else "{:.0f}"
            cells.append(f"{fmt.format(mu)} $\\pm$ {fmt.format(sd)}")
        name = f"\\textbf{{{lab}}}" if lab == PROPOSED else lab
        lines.append(name + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(STATS, exist_ok=True)
    df = load()
    if df.empty:
        print("no results yet")
        return
    summ, pw, fr = summarize(df), pairwise(df), friedman(df)
    summ.to_csv(os.path.join(STATS, "summary.csv"), index=False)
    pw.to_csv(os.path.join(STATS, "pairwise_wilcoxon_holm.csv"), index=False)
    fr.to_csv(os.path.join(STATS, "friedman.csv"), index=False)
    key = ["FND", "COV90", "packets_delivered", "PDR", "energy_per_packet_mJ", "malicious_CH_share_pre_HND"]
    for scen in sorted(df.scenario.unique()):
        labs = [l for l in MAIN if l in set(df[df.scenario == scen].label)] + \
               sorted(set(df[df.scenario == scen].label) - set(MAIN))
        latex_table(summ, scen, key, labs, os.path.join(STATS, f"table_{scen}.tex"))
    md = ["# Final experiments: statistical summary", ""]
    for scen in sorted(df.scenario.unique()):
        g = pw[pw.scenario == scen]
        n = df[df.scenario == scen].seed.nunique()
        md += [f"## {scen}  ({n} seeds)", "", "| metric | vs | BT-WOA-SA | other | diff | p (Holm) | A12 | verdict |",
               "|:-|:-|-:|-:|-:|-:|-:|:-|"]
        for _, r in g.iterrows():
            md.append(f"| {r.metric} | {r.versus} | {r.bt_mean:.4g} | {r.other_mean:.4g} | {100 * r.rel_diff:+.1f}% | "
                      f"{r.p_holm:.2g} | {r.A12_bt_better:.2f} | {r.verdict} |")
        md.append("")
    with open(os.path.join(STATS, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print(f"{len(df)} runs, {df.scenario.nunique()} scenarios -> {STATS}")


if __name__ == "__main__":
    main()
