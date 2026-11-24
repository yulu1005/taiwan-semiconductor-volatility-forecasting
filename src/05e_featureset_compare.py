# 05e_featureset_compare.py — 特徵組比較：現行 17 項 vs 組員提出的 30 項 vs 兩者聯集（驗證期 2022–2024，不交易）
# 規則與 05_validate_v3 相同：每週最後交易日決策、全部合格股票訓練、可交易股票評估、QLIKE 選模
import itertools, numpy as np, pandas as pd                  # 載入：組合、數值、表格
from common import CFG, PROC, VTAB, VER, nw_mean_ci          # 載入：設定、路徑、HAC 區間
from models import load, split, make_model, feat_cols, log_runs, FLOOR  # 載入：模型工具
from features_30 import F30                                  # 載入：30 項特徵名稱
assert VER == "v3", "請在 setup.yaml 設 features.active: v3"   # 只用於 v3
DEC = "is_decision_wk"; MC = CFG["v3"]["models"]             # 決策日欄位、模型設定
df = load().merge(pd.read_parquet(PROC / "features_30.parquet"), on=["date", "sid"], how="left")  # 合格樣本＋30 項特徵
S17 = feat_cols(20, "v3")                                    # 現行 17 項
SETS = {"17項": S17, "30項": F30, "聯集": S17 + [c for c in F30 if c not in S17]}  # 三組特徵
if (VTAB / "feature_select_list.csv").exists(): SETS["精選"] = pd.read_csv(VTAB / "feature_select_list.csv").feature.tolist()  # 05f 只用 2016–2020 篩出的精選組
if (VTAB / "feature_select_list_std.csv").exists(): SETS["標準版"] = pd.read_csv(VTAB / "feature_select_list_std.csv").feature.tolist()  # 05f 標準版（15–20 項）
FOLDS = CFG["protocol"]["folds"] + [{"train": ["2016-11-01", "2020-12-31"], "valid": ["2021-01-01", "2021-12-31"]}]  # 3 個驗證 fold＋2021 校準
POP = [("lasso", {"alpha_frac": a}, None) for a in MC["sparse_alpha_frac"]]  # Lasso 6 個
POP += [("ridge", {"alpha": a}, None) for a in MC["ridge_alpha"]]  # Ridge 11 個
POP += [("lgbm", {"objective": "l2", "num_leaves": nl, "min_child_samples": m, "n_estimators": 200}, [50, 100, 200])
        for nl, m in itertools.product([7, 15, 31], [50, 100])]  # LightGBM（平方誤差）6 個設定 × 3 個樹數
def name(kind, prm, ck=None):                                # 定義：設定名稱
    if kind == "lgbm": return f"lgbm_l2_L{prm['num_leaves']}_m{prm['min_child_samples']}_t{ck}"  # LightGBM
    return f"{kind}_{list(prm.values())[0]}"                 # 線性模型
rows, preds = [], []                                         # 收集
for k, fd in enumerate(FOLDS):                               # 逐一 fold
    fold = f"fold{k+1}" if k < 3 else "calib2021"            # 名稱
    tr, te = split(df, fd["train"][1], *fd["valid"], dec_col=DEC)  # 訓練（含 purge）與每週決策日
    te = te[te.tradable.fillna(False).astype(bool)]          # 只評估可交易股票
    for sname, cols in SETS.items():                         # 逐一特徵組
        for kind, prm, cks in POP:                           # 逐一模型設定
            m = make_model(kind, prm).fit(tr[cols], tr.y)    # 訓練
            outs = {name(kind, prm, c): m.predict(te[cols], num_iteration=c) for c in cks} if cks else {name(kind, prm): m.predict(te[cols])}  # 預測
            for cfg, p in outs.items():                      # 逐一候選
                p = np.maximum(np.asarray(p, float), FLOOR); t = te[["date", "sid", "y"]].assign(p=p)  # 截下限
                x = (t.y / t.p) ** 2; ql = x - np.log(x) - 1  # 每筆 QLIKE
                ic = t.groupby("date").apply(lambda g: g.p.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).mean()  # 每週 IC
                rows.append({"set": sname, "family": kind, "config": cfg, "fold": fold, "QLIKE": ql.mean(),
                             "MAE": (t.p - t.y).abs().mean(), "RMSE": np.sqrt(((t.p - t.y) ** 2).mean()), "IC": ic})  # 記錄
                preds.append(t.assign(set=sname, config=cfg, fold=fold, ql=ql.values))  # 存預測
    print("完成", fold)                                      # 進度
R = pd.DataFrame(rows); P = pd.concat(preds)                 # 轉表
V = R[R.fold != "calib2021"].groupby(["set", "family", "config"])[["QLIKE", "MAE", "RMSE", "IC"]].mean().reset_index()  # 三個驗證年度平均
best = V.loc[V.groupby(["set", "family"]).QLIKE.idxmin()]   # 每組特徵 × 每種模型的最佳設定（依 QLIKE）
yr = R.pivot_table(index=["set", "config"], columns="fold", values="QLIKE").reset_index()  # 各年 QLIKE（含 2021 校準年）
best = best.merge(yr, on=["set", "config"]).sort_values(["family", "QLIKE"])  # 合併各年
best.round(4).to_csv(VTAB / "featureset_compare.csv", index=False); print(best.round(4).to_string(index=False))  # 存檔與印出
def weekly(sname, cfg): return P[(P.set == sname) & (P.config == cfg) & (P.fold != "calib2021")].groupby("date").ql.mean()  # 每週平均 QLIKE
pc = []                                                      # 成對比較（同一種模型的最佳設定）
for fam in ["lasso", "ridge", "lgbm"]:                       # 三種模型
    b = best[best.family == fam].set_index("set").config     # 各特徵組的最佳設定
    for other in [k for k in SETS if k != "17項"]:          # 與 17 項比較
        d = (weekly(other, b[other]) - weekly("17項", b["17項"])).dropna(); m, lo, hi, _ = nw_mean_ci(d.values)  # 差異（正＝比 17 項差）
        pc.append({"model": fam, "compare": f"{other} − 17項", "diff": m, "lo95": lo, "hi95": hi})  # 記錄
pc = pd.DataFrame(pc); pc.round(4).to_csv(VTAB / "featureset_paired.csv", index=False); print(pc.round(4).to_string(index=False))  # 存檔與印出
cf = []                                                      # 30 項中 Lasso 使用哪些
bl = best[(best.family == "lasso") & (best.set == "30項")].config.iloc[0]; frac = float(bl.split("_")[1])  # 30 項的最佳 Lasso
for k, fd in enumerate(CFG["protocol"]["folds"]):            # 三個 fold
    tr, _ = split(df, fd["train"][1], *fd["valid"], dec_col=DEC)  # 訓練集
    cf.append(pd.Series(make_model("lasso", {"alpha_frac": frac}).fit(tr[F30], tr.y).m.coef_, index=F30, name=f"fold{k+1}"))  # 標準化係數
CF = pd.concat(cf, axis=1); CF.round(4).to_csv(VTAB / "featureset_30_lasso_coef.csv"); print(CF.round(4).to_string())  # 存檔與印出
log_runs(R.assign(trial_family="featureset_compare", fset=R.set, eval_window=R.fold, lookback=20, decision="comparison",
                  TS_corr_level=np.nan, n=np.nan).to_dict("records"))  # 每個候選都記入研究帳本
