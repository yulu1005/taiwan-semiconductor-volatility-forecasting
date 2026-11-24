# 05_model1_v7.py — 依 V7 說明書執行 Model 1：117 個候選（4 個波動率模型基準 ＋ 23 個線性 ＋ 90 個 LightGBM）
#   V7 2.5：擴張式 walk-forward，訓練至 2021／2022／2023 年底，驗證 2022／2023／2024；2021 為校準年（訓練至 2020）
#   V7 2.6：三個驗證年度「可交易股票」的平均 QLIKE 最低者；差距 1% 內選超參數較少、懲罰較強者
#   V7 2.4.1：第 18 項 GARCH 特徵缺值（上市未滿 250 日）的樣本不納入訓練；當週也沒有預測（不進入交易名單）
#   V7 2.8：conformal 90% 預測區間以 2021 校準年殘差估計；另報告各年實際覆蓋率
#   2025–2026 事後評估：以訓練至 2024 年底的主模型預測（V7 未規定重訓頻率，採一次訓練）
import itertools, numpy as np, pandas as pd                  # 載入：組合、數值、表格
from common import CFG, PROC, VTAB, VER, vproc, nw_mean_ci   # 載入：設定、輸出位置、HAC 區間
from models import load, split, feat_cols, make_model, log_runs, FLOOR  # 載入：模型工具（與先前版本相同的實作）
assert VER == "v7", "請在 setup.yaml 設 features.active: v7"  # 只用於 V7
MC = CFG["v3"]["models"]; G = MC["lgbm"]                     # 候選網格沿用 setup.yaml（與 V7 2.6 表列一致）
DEC = "is_decision_wk"                                       # 每週最後一個交易日決策
COLS = feat_cols(20, VER)                                    # 18 項特徵
HAR_COLS = ["rv1", "rv5", "rv20"]                            # HAR：日、週、月波動
df = load()                                                  # 合格樣本
panel = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])  # 日資料
r = np.log(panel.aclose / panel.groupby("sid").aclose.shift())  # 收盤日報酬（EWMA 用；RiskMetrics 標準做法）
panel["ewma"] = np.sqrt(252 * (r ** 2).groupby(panel.sid).transform(lambda s: s.ewm(alpha=0.06, adjust=False, min_periods=20).mean()))  # EWMA λ=0.94
df = df.merge(panel[["date", "sid", "ewma"]], on=["date", "sid"], how="left")  # 合併
FOLDS = [("fold1", "2021-12-31", "2022-01-01", "2022-12-31"), ("fold2", "2022-12-31", "2023-01-01", "2023-12-31"),
         ("fold3", "2023-12-31", "2024-01-01", "2024-12-31"), ("calib2021", "2020-12-31", "2021-01-01", "2021-12-31")]  # 三個驗證年＋校準年

# ---------- 1. 候選族群（117 個；看結果前由 V7 2.6 決定） ----------
POP = [("hv", {}, None), ("ewma", {}, None), ("garch", {}, None), ("har", {}, None)]  # 4 個波動率模型基準
POP += [("ridge", {"alpha": a}, None) for a in MC["ridge_alpha"]]  # Ridge 11 個（10⁻⁴～10⁶）
POP += [("lasso", {"alpha_frac": f}, None) for f in MC["sparse_alpha_frac"]]  # Lasso 6 個
POP += [("enet", {"alpha_frac": f, "l1_ratio": MC["enet_l1_ratio"]}, None) for f in MC["sparse_alpha_frac"]]  # ElasticNet 6 個
for obj, nl, mc in itertools.product(G["objective"], G["num_leaves"], G["min_child_samples"]):  # 3 損失 × 3 葉節點 × 2 葉樣本
    POP.append(("lgbm", {"objective": obj, "num_leaves": nl, "min_child_samples": mc, "n_estimators": max(G["checkpoints"])}, G["checkpoints"]))  # × 5 個樹數
def name(kind, prm, ck=None):                                # 定義：候選名稱
    if kind == "lgbm": return f"lgbm_{prm['objective']}_L{prm['num_leaves']}_m{prm['min_child_samples']}_t{ck}"
    return kind + ("" if not prm else "_" + "_".join(f"{k}{v}" for k, v in prm.items() if k != "l1_ratio"))
N_CAND = sum(len(c) if c else 1 for _, _, c in POP)          # 候選總數
assert N_CAND == 117, N_CAND                                 # V7 2.6：4＋23＋90＝117
print(f"候選 {N_CAND} 個；特徵 {len(COLS)} 項：{COLS}")

def predict(kind, prm, cks, tr, te):                         # 定義：一個設定在一個 fold 的預測（回傳 {名稱: 預測}）
    if kind == "hv": return {name(kind, prm): te.rv20.values}  # HV20
    if kind == "ewma": return {name(kind, prm): te.ewma.values}  # EWMA
    if kind == "garch": return {name(kind, prm): te.garch_oo5.values}  # GARCH(1,1)（與第 18 項特徵完全相同的計算）
    if kind == "har": return {name(kind, prm): make_model("ridge", {"alpha": 1e-4}).fit(tr[HAR_COLS], tr.y).predict(te[HAR_COLS])}  # HAR
    p = dict(prm)
    if p.get("objective") == "huber": p["huber_delta"] = G["huber_delta_frac"] * tr.y.std()  # Huber 轉折點
    m = make_model(kind, p).fit(tr[COLS], tr.y)              # 訓練
    if cks: return {name(kind, prm, c): m.predict(te[COLS], num_iteration=c) for c in cks}  # LightGBM 每個樹數
    return {name(kind, prm): m.predict(te[COLS])}            # 線性模型

def qlike_vec(y, p): x = (y / p) ** 2; return x - np.log(x) - 1  # 每筆 QLIKE
def mets(t):                                                 # 定義：一組預測的指標
    ic = t.groupby("date").apply(lambda g: g.p.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).dropna()  # 每週 IC
    return {"QLIKE": qlike_vec(t.y.values, t.p.values).mean(), "MAE": (t.p - t.y).abs().mean(),
            "RMSE": np.sqrt(((t.p - t.y) ** 2).mean()), "bias": t.p.mean() - t.y.mean(), "IC": ic.mean(), "n": len(t)}

# ---------- 2. 每個 fold 訓練與預測 ----------
rows, preds, floor_hits = [], [], 0
for fold, tr_end, v0, v1 in FOLDS:
    tr, te = split(df, tr_end, v0, v1, dec_col=DEC)          # 訓練（依標籤結束日 purge）與每週決策日
    tr = tr[tr.garch_oo5.notna()]                            # V7 2.4.1：GARCH 缺值的樣本不納入訓練
    te = te[te.tradable.fillna(False).astype(bool) & te.garch_oo5.notna()]  # 評估：可交易且有預測的股票
    for kind, prm, cks in POP:
        for cfg, p in predict(kind, prm, cks, tr, te).items():
            p = np.asarray(p, float); floor_hits += int((p < FLOOR).sum()) if kind not in ("hv", "ewma", "garch") else 0  # 記錄被截到下限的次數
            t = te[["date", "sid", "y", "label_end"]].assign(p=np.maximum(p, FLOOR))  # 截到下限（年化 1%，程式既有設定）
            fam = cfg.split("_t")[0].rsplit("_L", 1)[0] if kind == "lgbm" else kind
            rows.append({"family": fam, "config": cfg, "fold": fold, **mets(t)}); preds.append(t.assign(config=cfg, family=fam, fold=fold))
    print("完成", fold, "訓練", len(tr), "評估", len(te))
R = pd.DataFrame(rows); P = pd.concat(preds)

# ---------- 3. 選模（V7 2.6） ----------
val = R[R.fold.str.startswith("fold")]                        # 三個驗證年度
S = val.groupby(["family", "config"])[["QLIKE", "MAE", "RMSE", "bias", "IC"]].mean().reset_index()  # 三年平均
def simple_key(c):                                           # 家族內簡單程度（越小越簡單）
    if c.startswith("ridge_alpha"): return -float(c.split("alpha")[1])
    if c.startswith(("lasso", "enet")): return -float(c.split("frac")[1])
    if c.startswith("lgbm"): import re; L, m, t = map(int, re.findall(r"_L(\d+)_m(\d+)_t(\d+)", c)[0]); return L * 1e6 + t * 1e2 - m
    return 0
NHP = {"hv": 0, "ewma": 0, "garch": 0, "har": 0, "ridge": 1, "lasso": 1, "enet": 2}  # 超參數個數（LightGBM 視為 4）
def pick(s):                                                 # QLIKE 最低；1% 內選超參數較少、懲罰較強
    b = s.QLIKE.min(); near = s[s.QLIKE <= b * 1.01].assign(h=lambda d: d.family.map(NHP).fillna(4), k=lambda d: d.config.map(simple_key))
    return near.sort_values(["h", "k"], kind="stable").iloc[0].config
PRIMARY = pick(S)                                            # 主模型：117 個候選中選出
fams = ["ridge", "lasso", "enet", "lgbm_l2", "lgbm_l1", "lgbm_huber"]
SEL = {f: pick(S[S.family == f]) for f in fams}; SEL["primary"] = PRIMARY
S.sort_values("QLIKE").round(4).to_csv(VTAB / "model1_population.csv", index=False)
pd.Series(SEL).to_csv(VTAB / "selected_configs.csv", header=["config"])
print("最低 QLIKE 前 5：\n", S.sort_values("QLIKE").head(5).round(4).to_string(index=False)); print("主模型：", PRIMARY)

# ---------- 4. V7 2.8 表：主模型與四個基準 ----------
BASE = ["hv", "ewma", "garch", "har"]
show = [PRIMARY] + BASE
tab = S[S.config.isin(show)].set_index("config").loc[show, ["QLIKE", "MAE", "RMSE", "bias", "IC"]]
yr = R[R.config.isin(show) & R.fold.str.startswith("fold")].pivot(index="config", columns="fold", values="QLIKE").loc[show]
tab = tab.join(yr); tab.round(4).to_csv(VTAB / "model1_table_2_8.csv"); print(tab.round(4).to_string())
Pv = P[P.fold.str.startswith("fold")]
W = Pv[Pv.config.isin(show)].assign(ql=lambda d: qlike_vec(d.y.values, d.p.values)).pivot_table(index=["date", "sid"], columns="config", values="ql")
pc = []
for b in BASE:
    d = (W[PRIMARY] - W[b]).groupby(level=0).mean().dropna(); m, lo, hi, _ = nw_mean_ci(d.values)  # 每週平均差異的 HAC 區間（負＝主模型較好）
    ps = W.groupby(level=1)[[PRIMARY, b]].mean(); win = int((ps[PRIMARY] < ps[b]).sum())  # 主模型較好的股票數
    yrs = {f: float(R[(R.config == PRIMARY) & (R.fold == f)].QLIKE.iloc[0] < R[(R.config == b) & (R.fold == f)].QLIKE.iloc[0]) for f in ("fold1", "fold2", "fold3")}  # 每年是否較好
    pc.append({"baseline": b, "diff": m, "lo95": lo, "hi95": hi, "ci_excludes_0": hi < 0 or lo > 0, "stocks_primary_better": win, "stocks": len(ps), "years_primary_better": int(sum(yrs.values()))})
PC = pd.DataFrame(pc); PC.round(4).to_csv(VTAB / "model1_vs_baselines.csv", index=False); print(PC.round(4).to_string(index=False))

# ---------- 5. 2025–2026 事後評估：訓練至 2024 年底 ----------
tr, te = split(df, "2024-12-31", "2025-01-01", CFG["data"]["study_end"], label_cap=CFG["data"]["study_end"], dec_col=DEC)
tr = tr[tr.garch_oo5.notna()]; te = te[te.tradable.fillna(False).astype(bool) & te.garch_oo5.notna()]
kind_of = lambda c: next(x for x in POP if any(name(x[0], x[1], ck) == c for ck in (x[2] or [None])))  # 由名稱找回設定
H, hrows = [], []
for cfg in show:
    kind, prm, cks = kind_of(cfg); ck = int(cfg.rsplit("_t", 1)[1]) if kind == "lgbm" else None
    p = predict(kind, prm, [ck] if ck else None, tr, te)[cfg]
    t = te[["date", "sid", "y", "label_end"]].assign(p=np.maximum(np.asarray(p, float), FLOOR), config=cfg, fold="posthoc2025")
    H.append(t); hrows.append({"config": cfg, **mets(t)})
HP = pd.concat(H); pd.DataFrame(hrows).round(4).to_csv(VTAB / "model1_posthoc_2025_2026.csv", index=False)
print("2025–2026 事後評估：\n", pd.DataFrame(hrows).round(4).to_string(index=False))

# ---------- 6. conformal 90% 預測區間（2021 校準年殘差） ----------
cal = P[(P.config == PRIMARY) & (P.fold == "calib2021")]
s = np.abs(np.log(cal.y / cal.p)).values; n = len(s)        # 非一致性分數：|ln(實際 ÷ 預測)|（比例誤差）
qhat = np.quantile(s, min(1.0, np.ceil((n + 1) * 0.9) / n), method="higher")  # 有限樣本修正的 90% 分位數
cov = []
for lab, t in [*[(f, Pv[(Pv.config == PRIMARY) & (Pv.fold == f)]) for f in ("fold1", "fold2", "fold3")], ("posthoc2025", HP[HP.config == PRIMARY])]:
    inside = (t.y >= t.p * np.exp(-qhat)) & (t.y <= t.p * np.exp(qhat))
    cov.append({"period": lab, "coverage": inside.mean(), "n": len(t), "median_width": float((t.p * (np.exp(qhat) - np.exp(-qhat))).median())})
CV = pd.DataFrame(cov); CV.round(4).to_csv(VTAB / "model1_conformal.csv", index=False)
print(f"conformal：q̂={qhat:.3f}（區間＝預測 × e^±q̂，約 ×{np.exp(-qhat):.2f}～×{np.exp(qhat):.2f}）\n", CV.round(3).to_string(index=False))
print("被截到下限（年化 1%）的 ML 預測次數：", floor_hits)

# ---------- 7. 存檔與研究帳本 ----------
pd.concat([P, HP]).to_parquet(vproc("oos_predictions.parquet"), index=False)  # 全部樣本外預測（三個驗證年、校準年、事後評估）
pd.Series({"qhat": qhat, "floor_hits": floor_hits, "n_candidates": N_CAND}).to_csv(VTAB / "model1_meta.csv", header=["value"])
log_runs(R.assign(trial_family="model1_v7_" + R.family, fset=VER, lookback=20, eval_window=R.fold, TS_corr_level=np.nan,
                  decision=np.where(R.config == PRIMARY, "selected", np.where(R.family.isin(BASE), "baseline", "rejected"))).to_dict("records"))
