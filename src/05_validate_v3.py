# 05_validate_v3.py — v3 模型族群驗證（不回測）：Ridge/Lasso/ElasticNet/LightGBM（3 損失 × 容量 × checkpoint），
#                     依 QLIKE 選模，並在不交易的情況下比較兩種風險等級規則
import itertools, numpy as np, pandas as pd                  # 載入：組合、數值、表格
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # 畫圖
from common import CFG, VTAB, VFIG, VER, vproc, nw_mean_ci, setup_fonts  # 載入：設定、v3 輸出位置、HAC
from models import load, split, feat_cols, make_model, log_runs, FLOOR  # 載入：模型工具
assert VER in ("v3", "v4"), "請在 setup.yaml 設 features.active: v3 或 v4"   # v3／v4 共用（交易規則都在 v3 區塊）
setup_fonts()                                                # 中文字型
V3 = CFG["v3"]; MC = V3["models"]; TR = V3["portfolio"]["tier_rules"]  # v3 設定
DEC = "is_decision_wk"                                       # 每週最後交易日決策
COLS = feat_cols(20, VER)                                    # v3：17 項；v4：18 項
df = load()                                                  # 合格樣本（含 v3 股票池與決策日）
FOLDS = CFG["protocol"]["folds"]; CALIB = {"train": ["2016-11-01", "2020-12-31"], "valid": ["2021-01-01", "2021-12-31"]}  # 驗證 fold 與校準 fold

# ---------- 1. 族群（看結果之前就由 setup.yaml 決定） ----------
POP = [("hv", {}, None), ("train_mean", {}, None)]           # 基準
POP += [("ridge", {"alpha": a}, None) for a in MC["ridge_alpha"]]  # Ridge 11 個
POP += [("lasso", {"alpha_frac": f}, None) for f in MC["sparse_alpha_frac"]]  # Lasso 6 個
POP += [("enet", {"alpha_frac": f, "l1_ratio": MC["enet_l1_ratio"]}, None) for f in MC["sparse_alpha_frac"]]  # ElasticNet 6 個
G = MC["lgbm"]                                               # LightGBM 網格
for obj, nl, mc in itertools.product(G["objective"], G["num_leaves"], G["min_child_samples"]):  # 3 × 3 × 2
    POP.append(("lgbm", {"objective": obj, "num_leaves": nl, "min_child_samples": mc, "n_estimators": max(G["checkpoints"])}, G["checkpoints"]))  # 每個設定 5 個 checkpoint
def name(kind, prm, ck=None):                                # 定義：設定名稱
    if kind == "lgbm": return f"lgbm_{prm['objective']}_L{prm['num_leaves']}_m{prm['min_child_samples']}_t{ck}"  # LightGBM
    return kind + ("" if not prm else "_" + "_".join(f"{k}{v}" for k, v in prm.items() if k != "l1_ratio"))  # 其他
n_trials = sum(len(c) if c else 1 for _, _, c in POP)        # 試驗數（每個 checkpoint 算一次）
print(f"族群：{len(POP)} 個設定、{n_trials} 個候選（含 checkpoint）；特徵 {len(COLS)} 項；決策日＝每週最後交易日")  # 印出

# ---------- 2. 每個 fold 訓練與預測 ----------
def qlike(y, p): r = (y / p) ** 2; return float(np.mean(r - np.log(r) - 1))  # 定義：QLIKE（低估罰得重）
def mets(t):                                                 # 定義：一組預測的指標
    ic = t.groupby("date").apply(lambda g: g.p.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).dropna()  # 每週 IC
    lv = t.groupby("date")[["p", "y"]].mean()                # 每週平均水準
    return {"MAE": (t.p - t.y).abs().mean(), "QLIKE": qlike(t.y.values, t.p.values), "bias": t.p.mean() - t.y.mean(),
            "IC": ic.mean(), "level_corr": lv.corr().iloc[0, 1], "n": len(t)}  # 回傳
rows, preds = [], []                                         # 收集
for k, fd in enumerate(FOLDS + [CALIB]):                     # 逐一 fold
    fold = f"fold{k+1}" if k < 3 else "calib2021"            # 名稱
    tr, te = split(df, fd["train"][1], *fd["valid"], dec_col=DEC)  # 訓練（含 purge）與每週決策日（標籤封住）
    for kind, prm, cks in POP:                               # 逐一設定
        if kind == "hv": outs = {name(kind, prm): te["rv20"].values}  # 基準：20 日波動
        elif kind == "train_mean": outs = {name(kind, prm): np.full(len(te), tr.y.mean())}  # 基準：訓練平均
        else:
            p = dict(prm)                                    # 複製參數
            if p.get("objective") == "huber": p["huber_delta"] = G["huber_delta_frac"] * tr.y.std()  # Huber 轉折點＝訓練標籤標準差 × 0.5
            m = make_model(kind, p).fit(tr[COLS], tr.y)      # 訓練
            if cks: outs = {name(kind, prm, c): m.predict(te[COLS], num_iteration=c) for c in cks}  # 每個 checkpoint 只用前 c 棵樹
            else: outs = {name(kind, prm): m.predict(te[COLS])}  # 線性模型
        for cfg, p in outs.items():                          # 逐一候選
            p = np.maximum(np.asarray(p, float), FLOOR); t = te[["date", "sid", "y", "label_end", "tradable", "rv5"]].assign(p=p)  # 截下限
            fam = cfg.split("_t")[0].rsplit("_L", 1)[0] if kind == "lgbm" else kind  # 家族（LightGBM 依損失分）
            r = {"family": fam, "config": cfg, "fold": fold, **{f"all_{a}": b for a, b in mets(t).items()}, **{f"trd_{a}": b for a, b in mets(t[t.tradable.fillna(False)]).items()}}  # 全部合格／可交易兩組指標
            rows.append(r); preds.append(t.assign(config=cfg, family=fam, fold=fold))  # 收集
    print("完成", fold, len(te), "筆")                         # 進度
R = pd.DataFrame(rows); P = pd.concat(preds)                 # 轉表

# ---------- 3. 依 QLIKE 選模（三個驗證年度、可交易股票） ----------
val = R[R.fold != "calib2021"]                               # 三個驗證年度
S = val.groupby(["family", "config"])[[c for c in R.columns if c.startswith(("all_", "trd_"))]].mean().reset_index()  # 三年平均
def simple_key(c):                                           # 定義：簡單程度（越小越簡單）
    if c.startswith("ridge_alpha"): return -float(c.split("alpha")[1])  # Ridge：alpha 越大越簡單
    if c.startswith(("lasso", "enet")): return -float(c.split("frac")[1])  # 稀疏：比例越大越簡單
    if c.startswith("lgbm"): import re; L, m, t = map(int, re.findall(r"_L(\d+)_m(\d+)_t(\d+)", c)[0]); return L * 1e6 + t * 1e2 - m  # 葉少、樹少、葉樣本多
    return 0                                                 # 基準
NHP = {"ridge": 1, "lasso": 1, "enet": 2}                   # 超參數個數（跨家族比簡單：個數少者簡單；LightGBM 視為 4）
def pick(s):                                                 # 定義：QLIKE 最低；1% 內選較簡單
    b = s.trd_QLIKE.min(); near = s[s.trd_QLIKE <= b * 1.01]  # 候選
    near = near.assign(h=near.family.map(NHP).fillna(4), k=near.config.map(simple_key))  # 先比超參數個數，再比家族內簡單程度
    return near.sort_values(["h", "k"], kind="stable").iloc[0].config  # 最簡單者
fams = ["ridge", "lasso", "enet", "lgbm_l2", "lgbm_l1", "lgbm_huber"]  # 家族
SEL = {f: pick(S[S.family == f]) for f in fams}              # 各家族最佳
SEL["primary"] = pick(S[S.config.isin(SEL.values())])        # 主模型：族群中最佳
S["selected"] = S.config.isin(SEL.values())                  # 標記
S.sort_values("trd_QLIKE").round(4).to_csv(VTAB / "model_population_validation.csv", index=False)  # 存全部候選
pd.Series(SEL).to_csv(VTAB / "selected_configs.csv", header=["config"])  # 存選定
print("選定：", SEL)                                          # 印出

# ---------- 4. 風險等級：兩種規則都在驗證期比較（不交易） ----------
def tier(x, lo, hi): return np.where(x < lo, "低", np.where(x <= hi, "中", "高"))  # 定義：分三級
def level_thr(train_end, valid_start):                       # 定義：訓練期每週平均實現波動的 P33/P67（可交易股票）
    t = df[df[DEC].fillna(False).astype(bool) & df.y.notna() & df.tradable.fillna(False) & (df.date <= pd.Timestamp(train_end)) & (df.label_end < pd.Timestamp(valid_start))]  # 訓練期
    m = t.groupby("date").y.mean(); return np.percentile(m, TR["level"]["lo_pct"]), np.percentile(m, TR["level"]["hi_pct"])  # 門檻
TH = {fd["valid"][0][:4]: level_thr(fd["train"][1], fd["valid"][0]) for fd in FOLDS}  # 各年門檻
tiers = []                                                   # 收集
for cfg in ["hv"] + list(dict.fromkeys(SEL.values())):  # HV20 與各家族最佳
    t = P[(P.config == cfg) & (P.fold != "calib2021") & P.tradable.fillna(False)]  # 可交易股票
    w = t.groupby("date").agg(p=("p", "mean"), y=("y", "mean"), h5=("rv5", "mean"))  # 每週平均
    lo, hi = zip(*[TH[str(d.year)] for d in w.index])        # 每週對應年份的門檻
    for rule in ("level", "ratio"):                          # 兩種規則
        if rule == "level": pt, rt = tier(w.p, np.array(lo), np.array(hi)), tier(w.y, np.array(lo), np.array(hi))  # 簡報：水準 vs 門檻
        else: pt, rt = tier(w.p / w.h5, TR["ratio"]["lo"], TR["ratio"]["hi"]), tier(w.y / w.h5, TR["ratio"]["lo"], TR["ratio"]["hi"])  # 書面提案：R_t
        tiers.append({"config": cfg, "rule": rule, "weeks": len(w), "hit_rate": (pt == rt).mean(),
                      "danger_low_when_high": ((pt == "低") & (rt == "高")).mean(), "high_recall": ((pt == "高") & (rt == "高")).sum() / max((rt == "高").sum(), 1),
                      "pred_low": (pt == "低").mean(), "pred_mid": (pt == "中").mean(), "pred_high": (pt == "高").mean(),
                      "real_low": (rt == "低").mean(), "real_high": (rt == "高").mean()})  # 記錄
T = pd.DataFrame(tiers); T.round(3).to_csv(VTAB / "tier_rule_comparison.csv", index=False)  # 存檔
tp = T[T.config == SEL["primary"]].sort_values(["danger_low_when_high", "hit_rate"], ascending=[True, False])  # 主模型的兩種規則
RULE = tp.iloc[0].rule                                       # 依事先寫定的準則選規則
pd.Series({"tier_rule": RULE, "primary_model": SEL["primary"]}).to_csv(VTAB / "strategy_choice.csv", header=["value"])  # 存選擇
print(T.round(3).to_string()); print("風險等級規則（依事先準則）：", RULE)  # 印出

# ---------- 5. 成對比較：主模型 vs HV20、v3 主模型 vs v1 LightGBM（MAE 損失）的偏誤 ----------
def weekly(cfg, col):                                        # 定義：每週可交易股票的指標序列
    t = P[(P.config == cfg) & (P.fold != "calib2021") & P.tradable.fillna(False)]  # 可交易
    g = t.groupby("date")                                    # 依週
    return g.apply(lambda x: (x.y / x.p) ** 2 - np.log((x.y / x.p) ** 2) - 1, include_groups=False).groupby(level=0).mean() if col == "qlike" else g.apply(lambda x: (x.p - x.y).abs().mean(), include_groups=False)  # 每週 QLIKE 或 MAE
pc = []                                                      # 收集
for a in ["hv", SEL["lgbm_l1"]]:                             # 對照：HV20、MAE 損失的 LightGBM（v1 用的損失）
    for col in ("qlike", "mae"):                             # 兩種指標
        d = (weekly(SEL["primary"], col) - weekly(a, col)).dropna(); m, lo, hi, _ = nw_mean_ci(d.values)  # 差異與 HAC 區間
        pc.append({"from": a, "to": SEL["primary"], "metric": col, "diff": m, "lo95": lo, "hi95": hi, "improves": hi < 0})  # 記錄
pd.DataFrame(pc).round(4).to_csv(VTAB / "paired_comparison.csv", index=False); print(pd.DataFrame(pc).round(4))  # 存檔與印出

# ---------- 6. 圖：範本 06（Ridge 懲罰曲線）、範本 07（checkpoint 學習曲線） ----------
fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))             # 三張
r = S[S.family == "ridge"].assign(a=lambda d: d.config.str.split("alpha").str[1].astype(float)).sort_values("a")  # Ridge
axs[0].plot(np.log10(r.a), r.trd_QLIKE, "o-", color="#2a78d6"); axs[0].set_xlabel("log10(alpha)"); axs[0].set_title("Ridge：懲罰強度 vs QLIKE")  # 懲罰曲線
col = {"l2": "#2a78d6", "l1": "#eda100", "huber": "#eb6834"}  # 損失顏色
lg = S[S.family.str.startswith("lgbm")].assign(base=lambda d: d.config.str.rsplit("_t", n=1).str[0], t=lambda d: d.config.str.rsplit("_t", n=1).str[1].astype(int))  # LightGBM
for b, g in lg.groupby("base"):                              # 每個設定一條線
    o = b.split("_")[1]; g = g.sort_values("t")              # 損失與排序
    axs[1].plot(g.t, g.trd_QLIKE, color=col[o], lw=1, alpha=.8); axs[2].plot(g.t, g.trd_bias, color=col[o], lw=1, alpha=.8)  # QLIKE 與偏誤
for o, c in col.items(): axs[1].plot([], [], color=c, label=o)  # 圖例
axs[1].legend(frameon=False); axs[1].set_xlabel("樹的數量（checkpoint）"); axs[1].set_title("LightGBM：QLIKE 學習曲線")  # 標籤
axs[2].axhline(0, color="#8a8984", lw=.8); axs[2].set_xlabel("樹的數量（checkpoint）"); axs[2].set_title("LightGBM：平均偏誤（預測 − 實際）")  # 偏誤
fig.suptitle(f"{VER} 驗證期 2022–2024（可交易股票、每週決策）：Ridge 懲罰到 1e4 前都平坦；LightGBM 50 棵樹最好，MAE 損失低估、平方誤差高估、Huber 最接近無偏", x=.01, ha="left", fontsize=12)  # 總標題
fig.tight_layout(); fig.savefig(VFIG / "v3_model_population.png", dpi=200, bbox_inches="tight")  # 存圖

# ---------- 7. 線性主模型留下哪些特徵（每個 fold 的標準化係數） ----------
cf = []                                                      # 收集
for fam in ("lasso", "enet", "ridge"):                       # 三種線性模型
    kind, cfg = fam, SEL[fam]                                # 選定設定
    prm = {"alpha": float(cfg.split("alpha")[1])} if kind == "ridge" else {"alpha_frac": float(cfg.split("frac")[1]), "l1_ratio": MC["enet_l1_ratio"] if kind == "enet" else 1.0}  # 參數
    for k, fd in enumerate(FOLDS):                           # 三個 fold
        tr, _ = split(df, fd["train"][1], *fd["valid"], dec_col=DEC)  # 訓練集
        m = make_model(kind, prm).fit(tr[COLS], tr.y)        # 訓練
        cf.append(pd.Series(m.m.coef_, index=COLS, name=f"{fam}_fold{k+1}"))  # 標準化後係數（每個特徵 1 個標準差的效果）
CF = pd.concat(cf, axis=1); CF.round(4).to_csv(VTAB / "linear_coefficients.csv")  # 存檔
print(CF.round(4).to_string())                               # 印出

# ---------- 8. 研究帳本與預測 ----------
P.to_parquet(vproc("oos_validation.parquet"), index=False)   # 存全部樣本外預測（含 2021 校準 fold）
log = R.assign(trial_family="model_v3_" + R.family, fset=VER, lookback=20, eval_window=R.fold,
               decision=np.where(R.config.isin(SEL.values()), "selected", np.where(R.family.isin(["hv", "train_mean"]), "baseline", "rejected")))  # 帳本欄位
log_runs(log.rename(columns={"all_MAE": "MAE", "all_IC": "IC", "all_level_corr": "TS_corr_level", "all_n": "n"}).to_dict("records"))  # 寫入帳本
print(S[S.selected][["family", "config", "trd_MAE", "trd_QLIKE", "trd_bias", "trd_IC", "trd_level_corr"]].sort_values("trd_QLIKE").round(4).to_string())  # 印出選定者
