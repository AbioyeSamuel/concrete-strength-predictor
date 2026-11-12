"""Temperature QC table (Table 5) and revised figures for AICE-D-26-00364."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTD = "revision_outputs/"
import os; os.makedirs(OUTD, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#444", "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300})
BLUE, ORANGE, GREEN, GOLD, GREY = "#2B78D3", "#EB6837", "#1BAE7B", "#EDA200", "#555555"
R = json.load(open("revision_outputs/final.json"))

# ---------------- temperature QC ----------------
h = pd.read_excel("data/heat-of-hydration-data.xlsx"); h.columns = [c.strip() for c in h.columns]
h["sess"] = h["Grade"].notna().cumsum()
h["ts"] = pd.to_datetime(h["Date"] + " " + h["Time"], format="%m/%d/%Y %H:%M:%S")
NAMES = {1: "M15, 1 d", 2: "M20, 1 d", 3: "M25, 1 d", 4: "M30, 1 d", 5: "M15, 3 d", 6: "M20, 3 d", 7: "M25, 3 d", 8: "M30, 3 d"}
CH = ["Temp A", "Temp B", "Temp C", "Temp D"]
cls = lambda v: np.where(v <= 5, "drop", np.where(((v >= 6) & (v <= 14)) | (v == 45), "fault", "valid"))
rows, allvalid, rep_n, rep_d = [], [], 0, 0
QC = {}
for s, g in h.groupby("sess"):
    g = g.sort_values("ts"); dt = g["ts"].diff().dt.total_seconds().dropna()
    vals = np.concatenate([g[c].values for c in CH]); k = cls(vals)
    valid = vals[k == "valid"]; allvalid.append(valid)
    reps = [(g[c].diff() == 0).sum() for c in CH]; n_pairs = 4 * (len(g) - 1)
    rep_n += sum(reps); rep_d += n_pairs
    dead = [c[-1] for c in CH if (g[c] <= 5).mean() > 0.9]
    # largest sustained rise per channel: 10-min rolling median minus first-30-min median, valid readings only
    rises = {}
    gi = g.set_index("ts")
    for c in CH:
        v = gi[c].where(cls(gi[c].values) == "valid")
        if v.notna().mean() < 0.5: continue
        rm = v.rolling("10min", min_periods=10).median(); base = v[: gi.index[0] + pd.Timedelta("30min")].median()
        rises[c[-1]] = (float(rm.max() - base), float((rm.idxmax() - gi.index[0]).total_seconds() / 3600))
    QC[s] = dict(session=NAMES[s], records=len(g), span=(g.ts.max() - g.ts.min()).total_seconds() / 3600,
                 active=dt[dt <= 120].sum() / 3600, drop=(k == "drop").mean() * 100, fault=(k == "fault").mean() * 100,
                 med=np.median(valid), q1=np.percentile(valid, 25), q3=np.percentile(valid, 75),
                 p1=np.percentile(valid, 1), p99=np.percentile(valid, 99), repeat=sum(reps) / n_pairs * 100,
                 dead=dead, rises=rises, median_dt=float(dt.median()))
allvalid = np.concatenate(allvalid)
allch = np.concatenate([h[c].values for c in CH]); kk = cls(allch)
QCsum = dict(n_channel=len(allch), n_drop=int((kk == "drop").sum()), n_6_14=int(((allch >= 6) & (allch <= 14)).sum()),
             n_45=int((allch == 45).sum()), n_44=int((allch == 44).sum()), n_valid=len(allvalid),
             pct_24_25=float(np.isin(allvalid, [24, 25]).mean() * 100),
             pct_23_26=float(((allvalid >= 23) & (allvalid <= 26)).mean() * 100),
             pct_24_25_drop_only=float(np.isin(allch[allch > 5], [24, 25]).mean() * 100),
             repeat_all=rep_n / rep_d * 100, n_E5=int((h["Temp E"] == 5).sum()), n_E=len(h),
             integer_only=bool(all((h[c] % 1 == 0).all() for c in CH + ["Temp E"])))
json.dump(dict(QC={str(k): v for k, v in QC.items()}, QCsum=QCsum), open(OUTD + "qc.json", "w"), indent=1, default=float)
for s, q in QC.items():
    print(f"{q['session']}: rec {q['records']} span {q['span']:.1f} act {q['active']:.1f} drop {q['drop']:.1f} fault {q['fault']:.1f} "
          f"med {q['med']:.0f} ({q['q1']:.0f}-{q['q3']:.0f}) P1-P99 {q['p1']:.0f}-{q['p99']:.0f} rep {q['repeat']:.1f} dead {q['dead']} "
          f"rises {{{', '.join(f'{c}:{r[0]:+.1f}K@{r[1]:.1f}h' for c, r in q['rises'].items())}}}")
print(QCsum)

# ---------------- Fig 1 workflow ----------------
fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
boxes = [((1, 25), "Material\ncharacterisation", "Specific gravity, setting\ntime, absorption, grading"),
         ((26, 25), "Mix proportioning\n& casting", "M15, M20, M25, M30\n32 cubes (150 mm)"),
         ((51, 25), "In-situ temperature\nmonitoring", "DS18B20 + ESP32 probe\n→ cloud sheet\n(not a model input)"),
         ((76, 25), "Compressive\nstrength testing", "Ages 1 and 3 days\n4 cubes per mix–age"),
         ((26, 2), "Dataset assembly", "Cement, sand, granite,\nwater, age → f_c (n = 32)"),
         ((51, 2), "ML models and\nbaselines", "XGBoost, GBR, SVR, ANN;\nmix–age mean, linear"),
         ((76, 2), "Validation and\ninterpretation", "Hold-out, repeated CV,\nleave-one-mix-out, nested CV")]
W, H = 23, 19
for (x, y), t, d in boxes:
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0,rounding_size=1.2", fc="#EDF4FC", ec=BLUE, lw=1.6))
    ax.text(x + W / 2, y + H - 4.6, t, ha="center", va="center", fontsize=9.6, weight="bold", color="#13315C")
    ax.text(x + W / 2, y + 5.2, d, ha="center", va="center", fontsize=7.9, color="#333")
arr = dict(arrowstyle="-|>", color=GREY, lw=1.4)
for x in [24, 49, 74]: ax.annotate("", (x + 2, 34.5), (x, 34.5), arrowprops=arr)
for x in [49, 74]: ax.annotate("", (x + 2, 11.5), (x, 11.5), arrowprops=arr)
ax.plot([37.5, 37.5], [25, 23.2], color=GREY, lw=1.4); ax.plot([87.5, 87.5, 37.5], [25, 23.2, 23.2], color=GREY, lw=1.4)
ax.annotate("", (37.5, 21.2), (37.5, 23.3), arrowprops=arr)
fig.savefig(OUTD + "fig1_flow.png", bbox_inches="tight", pad_inches=0.05); plt.close(fig)

# ---------------- Fig 3 grading ----------------
# Sand: percentages passing from the sieve analysis (thesis FM calculation); granite: as stated in the text.
sand = {9.5: 100, 4.75: 89.0, 2.36: 40.8, 1.18: 9.5, 0.6: 2.1, 0.3: 0, 0.15: 0}
gran = {37.5: 100, 20: 68.7, 10: 0}
c33 = {9.5: (100, 100), 4.75: (95, 100), 2.36: (80, 100), 1.18: (50, 85), 0.6: (25, 60), 0.3: (5, 30), 0.15: (0, 10)}
fig, ax = plt.subplots(figsize=(6.6, 3.8))
xs = sorted(c33); ax.fill_between(xs, [c33[x][0] for x in xs], [c33[x][1] for x in xs], color="#BBBBBB", alpha=0.35, lw=0,
                                  label="ASTM C33 fine-aggregate limits")
ax.plot(sorted(sand), [sand[k] for k in sorted(sand)], "o-", color=BLUE, ms=4, label="Sand")
ax.plot(sorted(gran), [gran[k] for k in sorted(gran)], "s-", color=ORANGE, ms=4, label="Granite")
ax.set_xscale("log"); ticks = [0.075, 0.15, 0.3, 0.6, 1.18, 2.36, 4.75, 10, 20, 37.5]
ax.set_xticks(ticks); ax.set_xticklabels(["0.075", "0.15", "0.3", "0.6", "1.18", "2.36", "4.75", "10", "20", "37.5"], fontsize=8)
ax.minorticks_off(); ax.set_xlim(0.06, 50); ax.set_ylim(-2, 102)
ax.set_xlabel("Sieve size (mm, log scale)"); ax.set_ylabel("Cumulative percentage passing (%)")
ax.grid(True, color="#E3E3E3", lw=0.6); ax.legend(frameon=False, fontsize=8, loc="center right")
fig.savefig(OUTD + "fig3_psd.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 4 temperature histories, sessions x channels ----------------
fig, axs = plt.subplots(8, 4, figsize=(10, 13), sharey=True)
for i, (s, g) in enumerate(h.groupby("sess")):
    g = g.sort_values("ts"); t = (g["ts"] - g["ts"].iloc[0]).dt.total_seconds() / 3600
    for j, c in enumerate(CH):
        ax = axs[i, j]; v = g[c].values.astype(float); k = cls(v)
        vv = np.where(k == "valid", v, np.nan)
        gap = np.r_[False, np.diff(t.values) > 120 / 3600]; vv = vv.copy(); tt = t.values.copy()
        vv[gap] = np.nan
        ax.plot(tt, vv, color=BLUE, lw=0.7)
        f = k == "fault"; ax.plot(tt[f], np.clip(v[f], 15, 46), "x", color=ORANGE, ms=2.2, mew=0.6)
        dpct = (k == "drop").mean() * 100
        if dpct > 90:
            ax.text(0.5, 0.5, f"dead channel\n({dpct:.0f}% dropouts)", transform=ax.transAxes, ha="center", va="center",
                    fontsize=7, color=ORANGE)
        elif dpct > 0:
            ax.text(0.98, 0.92, f"{dpct:.0f}% dropouts", transform=ax.transAxes, ha="right", va="top", fontsize=6, color=GREY)
        ax.set_ylim(14, 47); ax.set_xlim(0, max(t.max(), 1)); ax.tick_params(labelsize=6.5)
        ax.grid(True, color="#EEE", lw=0.5)
        if i == 0: ax.set_title(f"Probe {c[-1]}", fontsize=9, weight="bold")
        if j == 0: ax.set_ylabel(NAMES[s] + "\n°C", fontsize=8)
        if i == 7: ax.set_xlabel("Time from start (h)", fontsize=7.5)
fig.tight_layout(h_pad=0.6, w_pad=0.4)
fig.savefig(OUTD + "fig4_temps.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 5 distribution ----------------
fig, ax = plt.subplots(figsize=(7.4, 3.6))
data = []
for s, g in h.groupby("sess"):
    vals = np.concatenate([g[c].values for c in CH]); data.append(vals[cls(vals) == "valid"])
bp = ax.boxplot(data, whis=(1, 99), showfliers=False, patch_artist=True, widths=0.55,
                medianprops=dict(color=ORANGE, lw=1.6), boxprops=dict(facecolor="#EDF4FC", edgecolor=BLUE),
                whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
for i, d in enumerate(data, 1): ax.plot([i, i], [d.min(), d.max()], "_", color=GREY, ms=8)
ax.set_xticks(range(1, 9)); ax.set_xticklabels([NAMES[s] for s in range(1, 9)], fontsize=8)
ax.set_ylabel("Valid probe reading (°C)"); ax.grid(True, axis="y", color="#E3E3E3", lw=0.6)
fig.savefig(OUTD + "fig5_dist.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 6 strength ----------------
df = pd.read_excel("venv/Scripts/compressive-strength-data.xlsx"); df.columns = ["cement", "coarse", "fine", "water", "age", "y"]
df["mix"] = df.cement.round(2).map({0.7: "M15", 0.88: "M20", 1.22: "M25", 1.5: "M30"})
st = df.groupby(["mix", "age"]).y.agg(["mean", "std"]).reset_index()
fig, ax = plt.subplots(figsize=(6.2, 3.8)); xm = np.arange(4); wdt = 0.36
for k, (a, col) in enumerate([(1, BLUE), (3, ORANGE)]):
    d = st[st.age == a]
    b = ax.bar(xm + (k - 0.5) * wdt, d["mean"], wdt, yerr=d["std"], capsize=3, color=col, label=f"{a} day" + ("s" if a > 1 else ""),
               error_kw=dict(lw=0.9, ecolor="#333"))
    for xx, m in zip(xm + (k - 0.5) * wdt, d["mean"]): ax.text(xx, m + 0.9, f"{m:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(xm); ax.set_xticklabels(["M15", "M20", "M25", "M30"]); ax.set_ylabel("Compressive strength (MPa)")
ax.set_ylim(0, 15.8); ax.legend(frameon=False); ax.grid(True, axis="y", color="#E3E3E3", lw=0.6); ax.set_axisbelow(True)
fig.savefig(OUTD + "fig6_strength.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 8 predicted vs measured (no fitted lines) ----------------
yt = [r["y"] for r in R["test_rows"]]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 6.2))
for ax, (n, col) in zip(axs.flat, [("XGBoost", BLUE), ("GBR", ORANGE), ("SVR", GREEN), ("ANN", GOLD)]):
    p = R["test_pred"][n]; ax.plot([0, 12], [0, 12], "--", color=GREY, lw=0.9)
    ax.scatter(yt, p, s=34, color=col, edgecolor="k", lw=0.4, zorder=3)
    hm = R["holdout"][n]
    ax.text(0.04, 0.95, f"{n}\nR² = {hm['R2']:.4f}\nRMSE = {hm['RMSE']:.3f} MPa", transform=ax.transAxes, va="top", fontsize=8)
    ax.set_xlim(0, 12); ax.set_ylim(0, 12); ax.set_aspect("equal"); ax.grid(True, color="#EEE", lw=0.5)
    ax.set_xlabel("Measured (MPa)", fontsize=8.5); ax.set_ylabel("Predicted (MPa)", fontsize=8.5)
fig.tight_layout(); fig.savefig(OUTD + "fig8_pred.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 10 importance over seeds ----------------
imp = np.array(R["gbr_imp_raw_seeds"]); labels = ["Cement", "Coarse agg.", "Fine agg.", "Water", "Age"]
order = [4, 3, 1, 0, 2]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
m, sd, lo, hi = imp.mean(0), imp.std(0), imp.min(0), imp.max(0)
yy = np.arange(5)[::-1]
ax.barh(yy, m[order], xerr=sd[order], color=[BLUE if i == 4 else "#8DB6E8" for i in order], capsize=3,
        error_kw=dict(lw=0.9))
for y_, i in zip(yy, order):
    ax.plot([lo[i], hi[i]], [y_ - 0.3, y_ - 0.3], color=GREY, lw=1.0)
    ax.text(max(hi[i], m[i] + sd[i]) + 2, y_, f"{m[i]:.1f} ({lo[i]:.0f}–{hi[i]:.0f})", va="center", fontsize=7.5)
ax.set_yticks(yy); ax.set_yticklabels([labels[i] for i in order]); ax.set_xlim(0, 125)
ax.set_xlabel("Relative importance (% of most important feature)"); ax.grid(True, axis="x", color="#EEE", lw=0.5)
fig.savefig(OUTD + "fig10_importance.png", bbox_inches="tight"); plt.close(fig)
print("figs ok")
