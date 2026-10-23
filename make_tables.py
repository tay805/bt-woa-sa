"""LaTeX tables for the paper, generated directly from the result files.

Reads results/final/stats/summary.csv and pairwise_wilcoxon_holm.csv (written by analyze_final.py)
and results/validation/validation_summary.csv, and writes one .tex snippet per table to
results/final/tables/. Every number printed in the paper's result tables comes from here.

Usage: python make_tables.py
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
STATS = os.path.join(HERE, "results", "final", "stats")
OUT = os.path.join(HERE, "results", "final", "tables")
PROPOSED = "BT-WOA-SA"

summ = pd.read_csv(os.path.join(STATS, "summary.csv"))
pw = pd.read_csv(os.path.join(STATS, "pairwise_wilcoxon_holm.csv"))


def stat(scen, lab, metric, what="mean"):
    r = summ[(summ.scenario == scen) & (summ.label == lab) & (summ.metric == metric)]
    return float(r[what].iloc[0]) if len(r) else np.nan


def mark(scen, lab, metric):
    """Significance mark of BT-WOA-SA versus this method (Holm-corrected Wilcoxon)."""
    if lab == PROPOSED:
        return ""
    r = pw[(pw.scenario == scen) & (pw.versus == lab) & (pw.metric == metric)]
    if r.empty or r.p_holm.iloc[0] >= 0.05:
        return ""
    return "$^{+}$" if r.verdict.iloc[0] == "better" else "$^{-}$"


def fmt(v, metric):
    if not np.isfinite(v):
        return "--"
    if metric in ("PDR",):
        return f"{v:.3f}"
    if metric in ("malicious_CH_share_pre_HND", "missed_pre_HND"):
        return f"{100 * v:.1f}"
    if metric == "energy_per_packet_mJ":
        return f"{v:.3f}"
    if metric == "recon_rmse_pre_HND":
        return f"{v:.3f}"
    if metric == "packets_delivered":
        return f"{v / 1e3:.1f}"
    return f"{v:.0f}"


def cell(scen, lab, metric, sd=False, sig=True):
    v = stat(scen, lab, metric)
    s = fmt(v, metric)
    if sd:
        s += r" $\pm$ " + fmt(stat(scen, lab, metric, "std"), metric)
    if sig:
        s += mark(scen, lab, metric)
    return r"\textbf{" + s + "}" if lab == PROPOSED else s


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


MAIN = ["LEACH", "DEEC", "DEEC (true avg)", "LEACH-C", "PSO-C", "WOA-C", "IWOA-SA", "CTRF-style gate", PROPOSED]
SHOW = {"DEEC (true avg)": "DEEC (true $\\bar{E}$)", "CTRF-style gate": "CTRF-style gate", PROPOSED: PROPOSED}
HEAD = ["FND", "COV90", "packets_delivered", "PDR", "energy_per_packet_mJ", "malicious_CH_share_pre_HND"]
HEADNAMES = ["FND", "COV$_{90}$", "Deliv. (k)", "PDR", "mJ/read.", "Att. CH (\\%)"]


def main_table(scen, name, extra=None, labels=MAIN):
    metrics = HEAD + (extra or [])
    names = HEADNAMES + {"missed_pre_HND": ["Missed events (\\%)"],
                         "recon_rmse_pre_HND": ["RMSE"]}.get((extra or [""])[0], []) if extra else HEADNAMES
    if extra:
        names = HEADNAMES + [{"missed_pre_HND": "Missed (\\%)", "recon_rmse_pre_HND": "RMSE"}[m] for m in extra]
    lines = [r"\begin{tabular}{l" + "r" * len(metrics) + "}", r"\toprule",
             "Method & " + " & ".join(names) + r" \\", r"\midrule"]
    for lab in labels:
        if np.isnan(stat(scen, lab, "FND")):
            continue
        lines.append(SHOW.get(lab, lab) + " & " + " & ".join(cell(scen, lab, m) for m in metrics) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(name, lines)


main_table("E1_periodic", "tab_e1_periodic")
main_table("E1_event", "tab_e1_event", extra=["missed_pre_HND"])
main_table("E4_bs_centre", "tab_e4", labels=[l for l in MAIN if l != "DEEC (true avg)"])
main_table("E6_intel_far_event", "tab_e6", extra=["recon_rmse_pre_HND"])

# ablation (E1, both traffic models)
ABL = [PROPOSED, "BT: without trust gate", "BT: WOA only (no SA)", "BT: SA only (no WOA)", "BT + energy trigger"]
ABLN = {PROPOSED: "BT-WOA-SA (full)", "BT: without trust gate": "without trust gate",
        "BT: WOA only (no SA)": "WOA only (no SA)", "BT: SA only (no WOA)": "SA only (no WOA)",
        "BT + energy trigger": "energy-triggered re-clustering"}
lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{Periodic} & \multicolumn{3}{c}{Event-driven} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Variant & FND & Deliv. (k) & PDR & FND & Deliv. (k) & PDR \\", r"\midrule"]
for lab in ABL:
    c = [cell(s, lab, m) for s in ("E1_periodic", "E1_event") for m in ("FND", "packets_delivered", "PDR")]
    lines.append(ABLN[lab] + " & " + " & ".join(c) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_ablation", lines)

# attacker ratio (E3 + E1 at 20 %)
SC = [("E3_mal0", "0"), ("E3_mal10", "10"), ("E1_periodic", "20"), ("E3_mal30", "30"), ("E3_mal40", "40")]
LAB3 = ["LEACH-C", "PSO-C", "IWOA-SA", "CTRF-style gate", PROPOSED]
lines = [r"\begin{tabular}{l" + "r" * len(SC) + "}", r"\toprule",
         r"Method & " + " & ".join(f"{p}\\,\\%" for _, p in SC) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(SC) + 1) + r"}{l}{\emph{Readings delivered (thousands)}} \\"]
for lab in LAB3:
    lines.append(SHOW.get(lab, lab) + " & " + " & ".join(cell(s, lab, "packets_delivered") for s, _ in SC) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{" + str(len(SC) + 1) + r"}{l}{\emph{First node dies (rounds)}} \\")
for lab in LAB3:
    lines.append(SHOW.get(lab, lab) + " & " + " & ".join(cell(s, lab, "FND") for s, _ in SC) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_attackers", lines)

# scalability (E2 + E1 at N = 100)
SC2 = [("E1_periodic", "100"), ("E2_N200", "200"), ("E2_N300", "300"), ("E2_N500", "500")]
LAB2 = ["LEACH", "LEACH-C", "PSO-C", "WOA-C", "IWOA-SA", "CTRF-style gate", "BT: SA only (no WOA)", PROPOSED]
NM2 = {"BT: SA only (no WOA)": "BT-WOA-SA, SA only"}
lines = [r"\begin{tabular}{l" + "r" * len(SC2) + "}", r"\toprule",
         "Method & " + " & ".join(f"$N={n}$" for _, n in SC2) + r" \\", r"\midrule",
         r"\multicolumn{" + str(len(SC2) + 1) + r"}{l}{\emph{Readings delivered (thousands)}} \\"]
for lab in LAB2:
    lines.append(NM2.get(lab, SHOW.get(lab, lab)) + " & " + " & ".join(cell(s_, lab, "packets_delivered") for s_, _ in SC2) + r" \\")
lines += [r"\midrule", r"\multicolumn{" + str(len(SC2) + 1) + r"}{l}{\emph{First node dies (rounds)}} \\"]
for lab in LAB2:
    lines.append(NM2.get(lab, SHOW.get(lab, lab)) + " & " + " & ".join(cell(s_, lab, "FND") for s_, _ in SC2) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_scalability", lines)

# attacker behaviour (E5 + E1 grey-hole), gate comparison
rows_ = pd.concat([pd.read_json(os.path.join(HERE, "results", "final", "rows", f), lines=True)
                   for f in os.listdir(os.path.join(HERE, "results", "final", "rows")) if f.endswith(".jsonl")])
gate = rows_.groupby(["scenario", "label"])[["gate_honest_excluded", "gate_malicious_excluded"]].mean()
SC5 = [("E1_periodic", "Grey-hole", "periodic"), ("E5_lowrate_periodic", "Low-rate", "periodic"),
       ("E5_onoff_periodic", "On-off", "periodic"), ("E1_event", "Grey-hole", "event"),
       ("E5_lowrate_event", "Low-rate", "event"), ("E5_onoff_event", "On-off", "event")]
GL = [("WOA-C", "none (WOA-C)"), ("CTRF-style gate", "CTRF-style"), (PROPOSED, "Bayesian")]
lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Attack & Gate & FND & Deliv. (k) & PDR & Hon. excl. (\%) & Att. excl. (\%) \\",
         r"\midrule"]
for s, att, tr in SC5:
    first = True
    for lab, nm in GL:
        if np.isnan(stat(s, lab, "FND")):
            continue
        he = gate.loc[(s, lab), "gate_honest_excluded"] if (s, lab) in gate.index else np.nan
        me = gate.loc[(s, lab), "gate_malicious_excluded"] if (s, lab) in gate.index else np.nan
        hx = "--" if not np.isfinite(he) else f"{100 * he:.1f}"
        mx = "--" if not np.isfinite(me) else f"{100 * me:.1f}"
        lead = f"{att}, {tr[:3]}." if first else ""
        lines.append(f"{lead} & {nm} & " + " & ".join(cell(s, lab, m) for m in ("FND", "packets_delivered", "PDR"))
                     + f" & {hx} & {mx}" + r" \\")
        first = False
    lines.append(r"\midrule" if s != SC5[-1][0] else "")
lines = [l for l in lines if l != ""]
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_attack_types", lines)

# prior / threshold sensitivity (E7)
e7 = summ[summ.scenario == "E7_prior_sensitivity"]
if not e7.empty:
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Prior & $z_{\max}$ & $z_i$, no evidence & FND & Deliv. (k) & Hon. excl. (\%) & Att. excl. (\%) \\",
             r"\midrule"]
    from scipy.stats import beta as B
    for a in (1, 4, 9, 19):
        for t in (0.5, 0.7):
            lab = f"Beta({a:g},1), z_max={t}"
            he = gate.loc[("E7_prior_sensitivity", lab), "gate_honest_excluded"]
            me = gate.loc[("E7_prior_sensitivity", lab), "gate_malicious_excluded"]
            z0 = B.cdf(0.85, a, 1)
            v = [fmt(stat("E7_prior_sensitivity", lab, m), m) for m in ("FND", "packets_delivered")]
            name = f"Beta({a},1)"
            row = f"{name} & {t} & {z0:.3f} & {v[0]} & {v[1]} & {100 * he:.1f} & {100 * me:.1f}" + r" \\"
            if a == 9 and t == 0.5:
                row = r"\textbf{" + name + r"} & \textbf{" + str(t) + "}" + row[len(f"{name} & {t}"):]
            lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_prior", lines)

# ns-3 validation
v = pd.read_csv(os.path.join(HERE, "results", "validation", "validation_summary.csv"))
sims = [("python", "Python"), ("ns3: own channels", "ns-3"), ("ns3: own channels + fading", "ns-3 + fading"),
        ("ns3: shared channel", "ns-3, one shared channel")]
lines = [r"\begin{tabular}{l" + "r" * len(sims) + "}", r"\toprule",
         "Method & " + " & ".join(n for _, n in sims) + r" \\", r"\midrule"]
for m in ["LEACH", "PSO-C", "WOA-C", PROPOSED]:
    c = [f"{v[(v.simulator == s) & (v.method == m)].PDR.iloc[0]:.4f}" for s, _ in sims]
    lines.append((r"\textbf{" + m + "}" if m == PROPOSED else m) + " & " + " & ".join(c) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
write("tab_ns3", lines)
print("tables ->", OUT, sorted(os.listdir(OUT)))
