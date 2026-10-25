"""Figures 2 to 7 of the manuscript, computed from the saved simulation outputs in data/.

Usage: python3 make_figures.py
Writes figures/*.pdf and results/figure_numbers.json (every number quoted from the figures).
"""
import json
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.linalg import expm

from cml_deescalation import lam_off_from
from common import auc
from npm1_cohort import relapsed

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.edgecolor": "#555555", "axes.labelcolor": "#222222",
                     "xtick.color": "#444444", "ytick.color": "#444444", "legend.frameon": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
BLUE, AQUA, ORANGE, GREY, LGREY = "#2a78d6", "#1baf7a", "#eb6834", "#8a8a85", "#cfcfca"
OUT = "figures/"
D = "data/"


def npm1_cohort(fn):
    R = json.load(open(D + fn))
    out = dict(
        y2=np.array([r["y_c2"] for r in R]),
        ext=np.array([r["extinct"] for r in R]),
        rel=np.array([relapsed(r) for r in R]),
        a=np.array([r["p"]["r"] - r["p"]["alpha"] for r in R]),
        m=np.array([r["p"]["s"] - r["p"]["beta"] - r["p"]["delta"] for r in R]),
        alpha=np.array([r["p"]["alpha"] for r in R]), beta=np.array([r["p"]["beta"] for r in R]))
    out["det"] = out["a"] * out["m"] - out["alpha"] * out["beta"]
    out["fate"] = np.where(out["ext"], "cure", np.where(out["a"] >= 0, "grow", np.where(out["det"] > 0, "cure", "persist")))
    return out


N = npm1_cohort("npm1_0.5_5.0_1e-11_1e-07.json")

# ---------------- Figure 2: drug-free regimes ----------------
fig, ax = plt.subplots(figsize=(3.4, 2.8))
rng = np.random.default_rng(0)
idx = rng.choice(np.where(~N["ext"])[0], 1500, replace=False)
x, y = N["a"][idx], N["det"][idx]
lin = 1e-5
ys = np.sign(y) * np.log10(1 + np.abs(y) / lin)
xl, yl = (-0.22, 0.2), (ys.min() * 1.05, ys.max() * 1.05)
ax.axvspan(0, xl[1], color=ORANGE, alpha=0.08, lw=0)
ax.fill_between([xl[0], 0], 0, yl[1], color=AQUA, alpha=0.10, lw=0)
ax.fill_between([xl[0], 0], yl[0], 0, color=BLUE, alpha=0.10, lw=0)
r = N["rel"][idx]
ax.scatter(x[~r], ys[~r], s=3, c=GREY, lw=0, alpha=0.6, label="no relapse by 3 years")
ax.scatter(x[r], ys[r], s=3, c=ORANGE, lw=0, alpha=0.8, label="relapse by 3 years")
ax.axvline(0, color="#444444", lw=0.6); ax.axhline(0, color="#444444", lw=0.6)
ax.set_xlim(*xl); ax.set_ylim(*yl)
ax.text(-0.21, yl[1] * 0.85, "cure", color="#0f6e4c", fontsize=8)
ax.text(-0.21, yl[0] * 0.85, "persistence", color="#1c5cab", fontsize=8)
ax.text(0.03, yl[1] * 0.85, "progression", color="#a8441c", fontsize=8)
ticks = np.array([-1e-2, -1e-3, -1e-4, 0, 1e-4, 1e-3, 1e-2])
ax.set_yticks(np.sign(ticks) * np.log10(1 + np.abs(ticks) / lin))
ax.set_yticklabels([r"$-10^{-2}$", r"$-10^{-3}$", r"$-10^{-4}$", "0", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$"])
ax.set_xlabel(r"$a = r-\alpha$ (day$^{-1}$)")
ax.set_ylabel(r"$\det M_{\mathrm{off}}$ (day$^{-2}$, symmetric log)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, markerscale=3, handletextpad=0.2)
fig.savefig(OUT + "fig2_regimes.pdf"); plt.close(fig)
fig2_rel = {f: float(N["rel"][(~N["ext"]) & (N["fate"] == f)].mean()) for f in ["cure", "persist", "grow"]}

# ---------------- Figure 3: identical response, different fate (two example patients) ----------------
al, be, m_, a_on = 0.0278, 0.0247, -0.0304, -0.040
pats = {"patient 1": 0.014, "patient 2": 0.111}         # kill rates; a = a_on + kappa
x0 = np.array([0.99, 0.01]); T1 = 365.0
t1 = np.linspace(0, T1, 200); t2 = np.linspace(0, 365, 200)
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
cols = {"patient 1": AQUA, "patient 2": ORANGE}
fig3 = {}
for name, k in pats.items():
    a = a_on + k
    Mon = np.array([[a - k, be], [al, m_]]); Moff = np.array([[a, be], [al, m_]]); Mhalf = np.array([[a - k / 2, be], [al, m_]])
    y_on = np.array([(expm(Mon * tt) @ x0).sum() for tt in t1]); xs = expm(Mon * T1) @ x0
    y_off = np.array([(expm(Moff * tt) @ xs).sum() for tt in t2])
    y_half = np.array([(expm(Mhalf * tt) @ xs).sum() for tt in t2])
    fig3[name] = dict(kappa=k, a=a, det_off=float(np.linalg.det(Moff)), lam_off=float(max(np.linalg.eigvals(Moff).real)),
                      lam_half=float(max(np.linalg.eigvals(Mhalf).real)))
    ls = "-" if name == "patient 1" else (0, (4, 2))
    axs[0].plot(t1, np.log10(y_on), color=cols[name], lw=1.6, ls=ls)
    axs[0].plot(T1 + t2, np.log10(y_off), color=cols[name], lw=1.6, ls=ls, label=name)
    axs[1].plot(t1, np.log10(y_on), color=cols[name], lw=1.6, ls=ls)
    axs[1].plot(T1 + t2, np.log10(y_half), color=cols[name], lw=1.6, ls=ls, label=name)
for ax, ttl in zip(axs, [r"($\mathit{a}$) full dose, then stop", r"($\mathit{b}$) full dose, then half dose"]):
    ax.axvline(T1, color=GREY, lw=0.6, ls=":")
    ax.axhline(-5, color=GREY, lw=0.6); ax.text(10, -4.85, "assay limit", fontsize=8, color=GREY)
    ax.set_xlabel("time (days)"); ax.set_title(ttl, fontsize=9, loc="left")
    ax.set_ylim(-6.5, 0.5)
axs[0].set_ylabel(r"$\log_{10}$ marker burden")
axs[0].legend(loc="lower left", fontsize=8)
fig.savefig(OUT + "fig3_identity.pdf"); plt.close(fig)

# ---------------- Figure 5: fate by MRD after course 2 ----------------
bins = [(-np.inf, 1e-5, "undetectable\n(<10$^{-5}$)"), (1e-5, 1e-4, "10$^{-5}$–10$^{-4}$"),
        (1e-4, 1e-3, "10$^{-4}$–10$^{-3}$"), (1e-3, np.inf, r"$\geq$10$^{-3}$")]
fig, ax = plt.subplots(figsize=(4.4, 2.8))
fig4 = []
for i, (lo, hi, lab) in enumerate(bins):
    msk = (N["y2"] >= lo) & (N["y2"] < hi) if np.isfinite(lo) else (N["y2"] < hi)
    sh = [np.mean(N["fate"][msk] == f) for f in ["cure", "persist", "grow"]]
    fig4.append(dict(bin=lab.replace("\n", " "), n=int(msk.sum()), shares=sh, relapse=float(N["rel"][msk].mean())))
    bottom = 0
    for s, c, f in zip(sh, [AQUA, BLUE, ORANGE], ["cure", "persistence", "progression"]):
        ax.bar(i, s - (0.004 if s > 0.01 else 0), bottom=bottom, color=c, width=0.62, label=f if i == 0 else None)
        if s >= 0.08:
            ax.text(i, bottom + s / 2, f"{100*s:.0f}%", ha="center", va="center", fontsize=8, color="white")
        bottom += s
    ax.text(i, 1.03, f"n={msk.sum()}", ha="center", fontsize=8, color="#444444")
ax.set_xticks(range(4)); ax.set_xticklabels([b[2] for b in bins], fontsize=8.5)
ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("blood MRD after course 2 (fraction of diagnostic burden)")
ax.set_ylabel("share of patients")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
fig.savefig(OUT + "fig5_fate_mix.pdf"); plt.close(fig)

# ---------------- Figure 6: within-course sampling, model-based estimate ----------------
with open(D + "npm1_estimator.log") as f:
    log = f.read()
blocks = log.split("== library ")[1:]
vals = {}
for b in blocks:
    lib = "correct" if b.startswith("prior_correct") else "wrong"
    for line in b.splitlines():
        mm = re.match(r"\s+(blood|marrow)\s+(standard|within-course).*AUC posterior growth rate ([0-9.]+)", line)
        if mm:
            vals[(lib, mm.group(1), mm.group(2))] = float(mm.group(3))
ceil = float(re.search(r"ceiling AUC \(true growth rate\) ([0-9.]+)", log).group(1))
fig, ax = plt.subplots(figsize=(4.4, 2.6))
groups = [("blood", "correct"), ("blood", "wrong"), ("marrow", "correct"), ("marrow", "wrong")]
w = 0.36
for i, (assay, lib) in enumerate(groups):
    s, wc = vals[(lib, assay, "standard")], vals[(lib, assay, "within-course")]
    ax.bar(i - w / 2 - 0.01, s - 0.5, bottom=0.5, width=w, color=LGREY, label="end-of-course samples" if i == 0 else None)
    ax.bar(i + w / 2 + 0.01, wc - 0.5, bottom=0.5, width=w, color=BLUE, label="plus within-course samples" if i == 0 else None)
    ax.text(i - w / 2, s + 0.008, f"{s:.2f}", ha="center", fontsize=8)
    ax.text(i + w / 2, wc + 0.008, f"{wc:.2f}", ha="center", fontsize=8)
ax.axhline(ceil, color=GREY, lw=0.8, ls="--"); ax.text(3.45, ceil + 0.008, "exact post-treatment rate", fontsize=8, color=GREY, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(["blood\ncorrect prior", "blood\nmisspecified", "marrow\ncorrect prior", "marrow\nmisspecified"], fontsize=8.5)
ax.set_ylim(0.5, 1.04); ax.set_ylabel("AUC for 3-year relapse")
ax.legend(loc="upper left", fontsize=8, ncol=2, bbox_to_anchor=(0, 1.12))
fig.savefig(OUT + "fig6_sampling_design.pdf"); plt.close(fig)

# ---------------- Figure 7: CML, exact kinetics and realistic qPCR ----------------
ideal = {"level at stop": [], "slow slope, full dose": [], "years in MR4.5": [], "slow slope, half dose": [], "two-level estimate": []}
for fn in ["cml_1e-08_0.0001_1.0_2.json", "cml_1e-09_1e-05_1.0_2.json", "cml_1e-10_1e-06_1.0_2.json"]:
    R = [r for r in json.load(open(D + fn)) if r["eligible"] and not r["extinct"]]
    rel = np.array([r["relapse"] for r in R]); rho = 0.5
    dfull = np.array([r["det_full"] for r in R]); dred = np.array([r["det_red"] for r in R])
    tfull = np.array([sum(r["lam_full"]) for r in R]); tred = np.array([sum(r["lam_red"]) for r in R])
    dh = (dred - rho * dfull) / (1 - rho); th = (tred - rho * tfull) / (1 - rho)
    lam_hat = (th + np.sqrt(np.maximum(th ** 2 - 4 * dh, 0))) / 2
    ideal["level at stop"].append(auc(np.log10(np.maximum([r["y_stop"] for r in R], 1e-14)), rel))
    ideal["slow slope, full dose"].append(auc([max(r["lam_full"]) for r in R], rel))
    ideal["years in MR4.5"].append(auc(-np.array([r["mr45_years"] for r in R]), rel))
    ideal["slow slope, half dose"].append(auc([max(r["lam_red"]) for r in R], rel))
    ideal["two-level estimate"].append(auc(lam_hat, rel))
real = {"depth at de-escalation": [], "half-dose slope (6 months)": [], "two-level, true kill": [],
        "two-level, population kill": [], "exact post-treatment rate": []}
undet = []
for fn in ["cml_deesc_1e-08_0.0001_5.json", "cml_deesc_1e-10_1e-06_6.json"]:
    R = json.load(open(D + fn)); rel = np.array([r["rel"] for r in R])
    sl = np.array([r["slope6"] if r["slope6"] is not None else np.nan for r in R])
    undet.append(float(np.mean(~np.isfinite(sl))))
    sli = np.where(np.isfinite(sl), sl, np.nanmin(sl) - 1e-3)
    km = float(np.median([r["kappa"] for r in R]))
    real["depth at de-escalation"].append(auc([r["depth"] for r in R], rel))
    real["half-dose slope (6 months)"].append(auc(sli, rel))
    real["two-level, true kill"].append(auc([lam_off_from(r["l1"], r["l2"], s, r["kappa"]) for r, s in zip(R, sli)], rel))
    real["two-level, population kill"].append(auc([lam_off_from(r["l1"], r["l2"], s, km) for r, s in zip(R, sli)], rel))
    real["exact post-treatment rate"].append(auc([r["lam_off"] for r in R], rel))
fig, axs = plt.subplots(2, 1, figsize=(5.4, 4.6))
for ax, dct, ttl, hi in [(axs[0], ideal, r"($\mathit{a}$) exact kinetics", "two-level estimate"),
                         (axs[1], real, r"($\mathit{b}$) qPCR with noise and detection limit", "exact post-treatment rate")]:
    names = list(dct)
    for i, n in enumerate(names):
        v = np.array(dct[n]); c = BLUE if n in ("two-level estimate", "two-level, true kill", "two-level, population kill") else LGREY
        if n == "exact post-treatment rate":
            c = GREY
        ax.barh(i, v.mean() - 0.5, left=0.5, color=c, height=0.62)
        ax.errorbar(v.mean(), i, xerr=[[v.mean() - v.min()], [v.max() - v.mean()]], color="#333333", lw=0.8, capsize=2)
        ax.text(v.max() + 0.012, i, f"{v.min():.2f}–{v.max():.2f}" if v.max() - v.min() >= 0.005 else f"{v.mean():.2f}",
                va="center", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
    ax.set_xlim(0.4, 1.12); ax.axvline(0.5, color="#444444", lw=0.6)
    ax.set_title(ttl, fontsize=9, loc="left"); ax.set_xlabel("AUC for molecular relapse")
axs[1].tick_params(labelleft=True)
fig.tight_layout()
fig.savefig(OUT + "fig7_cml.pdf"); plt.close(fig)

# ---------------- Figure 4: poles of M(k) against the kill rate k ----------------
def poles(a, m, ab, k):
    d = np.sqrt((a - k - m) ** 2 + 4 * ab)
    return (a - k + m + d) / 2, (a - k + m - d) / 2

FS = 8
xl = (-0.21, 0.13)
fig = plt.figure(figsize=(6.3, 4.6))
locus = {}
for name, k_op in pats.items():
    a = a_on + k_op
    p_slow, p_fast = poles(a, m_, al * be, 0.0); c_slow, c_fast = poles(a, m_, al * be, k_op)
    locus[name] = dict(open_loop=[float(p_slow), float(p_fast)], operating=[float(c_slow), float(c_fast)], kappa=k_op,
                       critical_gain=float(a - al * be / m_))
# ---- pole locations against the gain ----
ax = fig.add_subplot(111)
KT = 0.30
ks = np.linspace(0, KT, 800)
ax.axvspan(0, xl[1], color=ORANGE, alpha=0.07, lw=0)
ax.axvline(0, color="#777777", lw=0.7)
ax.text(0.03, 0.128, "unstable half-plane:\nmarker grows", fontsize=FS, color="#a8441c")
ax.text(-0.15, 0.238, "stable half-plane:\nmarker declines", fontsize=FS, color="#555555")
for name, k_op in pats.items():
    a = a_on + k_op
    lp, lm = poles(a, m_, al * be, ks)
    ls = "-" if name == "patient 1" else (0, (4, 2))
    ax.plot(lp, ks, color=cols[name], lw=1.7, ls=ls)
    ax.plot(lm, ks, color=cols[name], lw=1.7, ls=ls)
    for branch in (lp, lm):
        i = np.searchsorted(ks, 0.055 if name == "patient 1" else 0.045)
        ax.annotate("", xy=(branch[i + 25], ks[i + 25]), xytext=(branch[i], ks[i]),
                    arrowprops=dict(arrowstyle="-|>", color=cols[name], lw=1.2, mutation_scale=9))
    p0 = poles(a, m_, al * be, 0.0); pk = poles(a, m_, al * be, k_op)
    ax.plot(p0, [0, 0], "x", color=cols[name], ms=8, mew=2.0, clip_on=False, zorder=5)
    ax.axhline(k_op, color=cols[name], lw=0.6, ls=":")
    ax.plot(pk, [k_op, k_op], "s", color=cols[name], ms=5.5, mec="white", mew=0.6, zorder=6)
ax.axvline(m_, color="#333333", lw=0.7, ls="--", ymax=0.97)
ax.plot([m_], [KT], "o", ms=8, mfc="white", mec="#222222", mew=1.6, clip_on=False, zorder=7)
ax.annotate("zero at $s=m$\n(protected cells' own growth):\nslow poles end here as $k\\to\\infty$",
            xy=(m_, KT), xytext=(0.006, 0.25), fontsize=FS, color="#222222",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
s_slow, s_fast = locus["patient 1"]["operating"]
k1, k2 = pats["patient 1"], pats["patient 2"]
for sx in (s_slow, s_fast):
    ax.plot([sx, sx], [k1, k2], color="#222222", lw=0.6, ls="-")
ax.annotate("same poles under treatment\n$\\Rightarrow$ identical MRD response", xy=(s_slow, (k1 + k2) / 2),
            xytext=(0.006, 0.175), fontsize=FS, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.6))
ax.annotate("drug-free pole $>0$:\nrelapses after stopping", xy=(locus["patient 2"]["open_loop"][0], 0.0),
            xytext=(0.05, 0.075), fontsize=FS, color="#a8441c", arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.6))
ax.annotate("drug-free pole $<0$:\ncured after stopping", xy=(locus["patient 1"]["open_loop"][0], 0.0),
            xytext=(-0.205, 0.026), fontsize=FS, color="#0f6e4c", arrowprops=dict(arrowstyle="-", color=AQUA, lw=0.6))
ax.text(-0.205, 0.288, "fast poles $\\to -\\infty$ as $k\\to\\infty$", fontsize=FS, color="#555555")
ax.set_xlim(*xl); ax.set_ylim(0, KT); ax.tick_params(labelsize=FS)
ax.set_xlabel(r"pole location $s$ on the real axis (day$^{-1}$)", fontsize=FS)
ax.set_ylabel(r"kill rate (feedback gain) $k$ (day$^{-1}$)", fontsize=FS)
handles = [Line2D([], [], color="#333333", marker="x", ls="none", ms=7, mew=1.8, label="drug-free (open-loop) pole, $k=0$: decides fate"),
           Line2D([], [], color="#222222", marker="o", mfc="white", ls="none", ms=7, mew=1.4, label="zero, $k\\to\\infty$"),
           Line2D([], [], color="#333333", marker="s", ls="none", ms=5.5, label="pole under treatment, $k=\\kappa$ (dotted line)"),
           Line2D([], [], color=AQUA, lw=1.7, label="patient 1 (cured)"),
           Line2D([], [], color=ORANGE, lw=1.7, ls=(0, (4, 2)), label="patient 2 (relapses)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=FS, columnspacing=1.5)
fig.savefig(OUT + "fig4_root_locus.pdf", bbox_inches="tight"); plt.close(fig)

# cohort check of Proposition 3 (reported in the text, not plotted)
ms_, lam_, kap_ = [], [], []
for fn in ["cml_1e-08_0.0001_1.0_2.json", "cml_1e-09_1e-05_1.0_2.json", "cml_1e-10_1e-06_1.0_2.json"]:
    R = [r for r in json.load(open(D + fn)) if r["eligible"] and not r["extinct"]]
    ms_ += [r["p"]["s"] - r["p"]["beta"] - r["p"]["delta"] for r in R]
    lam_ += [max(r["lam_full"]) for r in R]; kap_ += [r["p"]["kappa"] for r in R]
ms_, lam_, kap_ = map(np.array, (ms_, lam_, kap_))
gap = lam_ - ms_
locus_stats = dict(n=int(gap.size), median_gap=float(np.median(gap)), share_gap_below_1e3=float(np.mean(np.abs(gap) < 1e-3)),
                   corr=float(np.corrcoef(lam_, ms_)[0, 1]), min_gap=float(gap.min()), share_m_nonneg=float(np.mean(ms_ >= 0)))

summary = dict(fig2_relapse_by_regime=fig2_rel, fig3=fig3, fig4=fig4, fig5=dict(vals={"|".join(k): v for k, v in vals.items()}, ceiling=ceil),
               fig6=dict(ideal={k: [float(x) for x in v] for k, v in ideal.items()},
                         real={k: [float(x) for x in v] for k, v in real.items()}, undetectable=undet),
               locus=dict(patients=locus, cml=locus_stats))
with open("results/figure_numbers.json", "w") as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
