"""Final analysis for the AICE-D-26-00364 revision. All numbers in the revised manuscript and
response letter come from this script (ML part) and revision_temperature_qc_and_figures.py (temperature part)."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

import os; os.makedirs("revision_outputs", exist_ok=True)
OUT = {}
df = pd.read_excel("venv/Scripts/compressive-strength-data.xlsx")
df.columns = ["cement", "coarse", "fine", "water", "age", "y"]
FEAT = ["cement", "coarse", "fine", "water", "age"]
X = df[FEAT].values.astype(float); y = df["y"].values.astype(float)
MIX = df["cement"].round(2).map({0.7: "M15", 0.88: "M20", 1.22: "M25", 1.5: "M30"}).values
GRP = np.array([["M15", "M20", "M25", "M30"].index(m) for m in MIX])


def met(yt, yp):
    mse = mean_squared_error(yt, yp)
    return dict(R2=r2_score(yt, yp), MSE=mse, RMSE=float(np.sqrt(mse)), MAE=mean_absolute_error(yt, yp),
                MAPE=float(np.mean(np.abs((yt - yp) / yt)) * 100))


class Lookup:
    def fit(self, X, y):
        self.d = {(a, b): y[(X[:, 0] == a) & (X[:, 4] == b)].mean() for a, b in set(zip(X[:, 0], X[:, 4]))}
        self.age = {b: y[X[:, 4] == b].mean() for b in set(X[:, 4])}
        return self
    def predict(self, X):
        return np.array([self.d.get((r[0], r[4]), self.age[r[4]]) for r in X])
    def get_params(self, deep=True): return {}
    def set_params(self, **p): return self


def make(name, seed=0):
    return {"XGBoost": lambda: make_pipeline(StandardScaler(), XGBRegressor()),
            "GBR": lambda: make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=seed)),
            "SVR": lambda: make_pipeline(StandardScaler(), SVR()),
            "ANN": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                                                                       learning_rate_init=0.001, random_state=seed)),
            "Lookup": lambda: Lookup(),
            "Linear": lambda: make_pipeline(StandardScaler(), LinearRegression())}[name]()


NAMES = ["XGBoost", "GBR", "SVR", "ANN", "Lookup", "Linear"]
rng = np.random.default_rng(2026)

# 1 ---- hold-out
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.1, random_state=0)
OUT["test_idx_0based"] = te.tolist(); OUT["train_idx_0based"] = sorted(tr.tolist())
OUT["test_rows"] = [dict(idx=int(i), mix=MIX[i], age=int(X[i, 4]), y=float(y[i])) for i in te]
OUT["holdout"], OUT["test_pred"] = {}, {}
fitted = {}
for n in NAMES:
    m = make(n); m.fit(X[tr], y[tr]); p = m.predict(X[te]); fitted[n] = m
    OUT["holdout"][n] = met(y[te], p); OUT["test_pred"][n] = p.round(4).tolist()
# group predictions (Table 9)
groups = sorted(set(zip(MIX, X[:, 4])), key=lambda g: (g[0], g[1]))
OUT["table9"] = []
for mx, ag in groups:
    msk = (MIX == mx) & (X[:, 4] == ag); row = X[msk][0:1]
    OUT["table9"].append(dict(mix=mx, age=int(ag), measured=float(y[msk].mean()),
                              train_mean=float(y[np.intersect1d(np.where(msk)[0], tr)].mean()),
                              **{n: float(fitted[n].predict(row)[0]) for n in ["XGBoost", "GBR", "SVR", "ANN"]}))
# ANN seed sensitivity on the hold-out split
OUT["ann_seed_holdout_R2"] = []
for s in range(10):
    m = make("ANN", s); m.fit(X[tr], y[tr]); OUT["ann_seed_holdout_R2"].append(r2_score(y[te], m.predict(X[te])))
# parameter count
mlp = fitted["ANN"][-1]
OUT["ann_params"] = int(sum(c.size for c in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))

# 2 ---- repeated 5-fold CV, RepeatedKFold(random_state=0) (protocol of the submitted manuscript)
def repcv(n, rs=0):
    r2, rm = [], []
    for a, b in RepeatedKFold(n_splits=5, n_repeats=10, random_state=rs).split(X):
        m = make(n); m.fit(X[a], y[a]); p = m.predict(X[b])
        r2.append(r2_score(y[b], p)); rm.append(np.sqrt(mean_squared_error(y[b], p)))
    return np.array(r2), np.array(rm)


OUT["repcv"] = {}
for n in NAMES:
    r2, rm = repcv(n)
    boots = [rng.choice(r2, len(r2)).mean() for _ in range(2000)]
    OUT["repcv"][n] = dict(R2_mean=r2.mean(), R2_sd=r2.std(), RMSE_mean=rm.mean(), RMSE_sd=rm.std(),
                           R2_CI=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
OUT["repcv_seed_sensitivity"] = {}
for rs in [1, 42]:
    OUT["repcv_seed_sensitivity"][rs] = {n: float(repcv(n, rs)[0].mean()) for n in ["XGBoost", "GBR", "SVR", "ANN", "Lookup", "Linear"]}

# 3 ---- leave-one-mix-out
OUT["lomo"] = {}
for n in NAMES:
    pooled = np.zeros(len(y)); folds = {}
    for a, b in LeaveOneGroupOut().split(X, y, GRP):
        m = make(n); m.fit(X[a], y[a]); p = m.predict(X[b]); pooled[b] = p
        folds[MIX[b][0]] = dict(R2=r2_score(y[b], p), RMSE=float(np.sqrt(mean_squared_error(y[b], p))),
                               bias=float(np.mean(p - y[b])))
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y)); bs.append(r2_score(y[i], pooled[i]))
    OUT["lomo"][n] = dict(pooled=met(y, pooled), CI=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                          folds=folds)

# 4 ---- nested CV: tuned vs default on the same outer folds
grids = {
    "SVR": (make_pipeline(StandardScaler(), SVR()),
            {"svr__C": [1, 10, 100, 1000], "svr__gamma": ["scale", 0.05, 0.2, 1.0], "svr__epsilon": [0.01, 0.1, 0.3]}),
    "XGBoost": (make_pipeline(StandardScaler(), XGBRegressor()),
                {"xgbregressor__max_depth": [2, 3, 6], "xgbregressor__n_estimators": [50, 100, 300],
                 "xgbregressor__learning_rate": [0.05, 0.3], "xgbregressor__reg_lambda": [1, 10]}),
    "GBR": (make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state=0)),
            {"gradientboostingregressor__max_depth": [2, 3], "gradientboostingregressor__n_estimators": [50, 100, 300],
             "gradientboostingregressor__learning_rate": [0.05, 0.1]}),
    "ANN": (make_pipeline(StandardScaler(), MLPRegressor(max_iter=2000, random_state=0)),
            {"mlpregressor__hidden_layer_sizes": [(8,), (16,), (100, 50, 25)], "mlpregressor__alpha": [1e-4, 1e-2, 1.0],
             "mlpregressor__learning_rate_init": [1e-3, 1e-2]}),
}
OUT["nested"] = {}
outer = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=0).split(X))
for n, (est, grid) in grids.items():
    tuned, default, chosen = [], [], []
    for a, b in outer:
        gs = GridSearchCV(est, grid, cv=KFold(4, shuffle=True, random_state=1), scoring="neg_mean_squared_error")
        gs.fit(X[a], y[a]); tuned.append(r2_score(y[b], gs.predict(X[b])))
        chosen.append({k.split("__")[-1]: str(v) for k, v in gs.best_params_.items()})
        m = make(n); m.fit(X[a], y[a]); default.append(r2_score(y[b], m.predict(X[b])))
    OUT["nested"][n] = dict(tuned_mean=float(np.mean(tuned)), tuned_sd=float(np.std(tuned)),
                            default_mean=float(np.mean(default)), default_sd=float(np.std(default)),
                            chosen=pd.Series([json.dumps(c, sort_keys=True) for c in chosen]).value_counts().head(3).to_dict())

# 5 ---- GBR importance over ten estimator seeds (28 training records, normalised to max)
imps = []
for s in range(10):
    g = GradientBoostingRegressor(random_state=s).fit(StandardScaler().fit_transform(X[tr]), y[tr])
    v = g.feature_importances_; imps.append(v / v.max() * 100)
imps = np.array(imps)
OUT["gbr_imp_seeds"] = {f: dict(mean=float(imps[:, j].mean()), sd=float(imps[:, j].std()), min=float(imps[:, j].min()),
                                max=float(imps[:, j].max()), age_top_all=bool((imps.argmax(1) == 4).all()))
                        for j, f in enumerate(FEAT)}
OUT["gbr_imp_raw_seeds"] = imps.round(2).tolist()
xg = XGBRegressor().fit(X[tr], y[tr]).feature_importances_
OUT["xgb_gain_share_%"] = dict(zip(FEAT, (xg / xg.sum() * 100).round(2).tolist()))
OUT["corr"] = pd.DataFrame(np.c_[X, y], columns=FEAT + ["fc"]).corr().round(3).to_dict()

json.dump(OUT, open("revision_outputs/final.json", "w"),
          indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("done")
