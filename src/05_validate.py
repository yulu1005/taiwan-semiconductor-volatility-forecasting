# 05_validate.py — 三個驗證年度的 walk-forward 模型比較、參數選擇、成對比較、SHAP 穩定度（不碰封存測試）
import itertools, numpy as np, pandas as pd                  # 載入：組合工具、數值運算、表格
import shap                                                  # 載入：SHAP 解釋工具
from common import CFG, PROC, VTAB as TAB, VER, vproc, nw_mean_ci  # 載入：設定、版本化輸出資料夾、HAC 工具
from models import load, split, fit_predict, metrics, feat_cols, make_model, log_runs, FLOOR  # 載入：模型工具

df = load()                                                  # 讀合格樣本（含 v2 特徵）
ACT = "all15" if VER == "v1" else VER                        # 目前特徵組（v1 在程式中叫 all15）
FSETS = [ACT, "hv3"] + (["all15"] if ACT != "all15" else [])  # 評估的特徵組：目前組、只用歷史波動（H2）、以及 v1（依新封存規則重算，供比較）
print("特徵組：", FSETS, "| 輸出到", TAB)                      # 印出
FOLDS = CFG["protocol"]["folds"]                             # 三個驗證 fold
CALIB = {"train": ["2016-11-01", "2020-12-31"], "valid": ["2021-01-01", "2021-12-31"]}  # 額外 fold：只用來校準 2022 年的預測區間
G = CFG["models"]["lgbm_grid"]                               # LightGBM 參數組合
GRID = [("hv", {}), ("train_mean", {})] + [("ridge", {"alpha": a}) for a in CFG["models"]["ridge_alpha"]] + \
       [("lgbm", dict(zip(["num_leaves", "min_child_samples", "n_estimators"], v)))
        for v in itertools.product(G["num_leaves"], G["min_child_samples"], G["n_estimators"])]  # 全部試驗設定

def cfg_name(kind, prm): return kind + ("" if not prm else "_" + "_".join(f"{k}{v}" for k, v in prm.items()))  # 定義：設定名稱

rows, preds = [], []                                         # 收集指標與預測
for fset in FSETS:                                           # 逐一特徵組
    for kind, prm in GRID:                                   # 逐一設定
        if kind in ("hv", "train_mean") and fset != ACT: continue  # 基準不分特徵組合，只跑一次
        for k, fd in enumerate(FOLDS + [CALIB]):             # 逐一 fold（最後一個是校準 fold）
            fold = f"fold{k+1}" if k < 3 else "calib2021"    # fold 名稱
            tr, te = split(df, fd["train"][1], *fd["valid"]) # 切資料（含 purge）
            p, neg = fit_predict(kind, prm, tr, te, feat_cols(20, fset))  # 訓練與預測
            r = metrics(te, p)                               # 計算指標
            rows.append({"family": kind, "config": cfg_name(kind, prm), "fset": fset, "lookback": 20, "fold": fold,
                         "train_window": f"{CFG['data']['study_start']}~{fd['train'][1]}", "eval_window": f"{fd['valid'][0]}~{fd['valid'][1]}",
                         "n_train": len(tr), "neg_pred_clipped": neg, **r})  # 記錄一次執行
            preds.append(te[["date", "sid", "y", "label_end"]].assign(p=p, config=cfg_name(kind, prm), fset=fset, fold=fold))  # 記錄預測
res = pd.DataFrame(rows)                                     # 轉成表格
val = res[res.fold != "calib2021"]                           # 只用三個驗證 fold 選參數
summ = val.groupby(["family", "config", "fset"]).agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"),
                                                     IC=("IC", "mean"), TS_corr=("TS_corr_level", "mean")).reset_index()  # 三年平均

def pick(fam, fset):                                         # 定義：選參數（平均 MAE 最低；差距 1% 內選較簡單者）
    s = summ[(summ.family == fam) & (summ.fset == fset)].copy()  # 這個家族的結果
    best = s.MAE.min(); near = s[s.MAE <= best * 1.01]       # 最佳值與 1% 內的候選
    if fam == "ridge":                                       # Ridge：alpha 越大越簡單
        near = near.assign(a=near.config.str.split("alpha").str[1].astype(float)).sort_values("a", ascending=False)
    else:                                                    # LightGBM：葉少、樹少、葉樣本多較簡單
        near = near.assign(c=near.config.str.extract(r"num_leaves(\d+)_min_child_samples(\d+)_n_estimators(\d+)").astype(int).apply(
            lambda r: (r[0], r[2], -r[1]), axis=1)).sort_values("c")
    return near.iloc[0].config                               # 回傳選中的設定
SEL = {"ridge": pick("ridge", ACT), "lgbm": pick("lgbm", ACT),
       "ridge_hv3": pick("ridge", "hv3"), "lgbm_hv3": pick("lgbm", "hv3")}  # 各家族選定設定
if ACT != "all15": SEL.update({"ridge_v1": pick("ridge", "all15"), "lgbm_v1": pick("lgbm", "all15")})  # v1 依新封存規則重選
SELF = {"ridge": ACT, "lgbm": ACT, "ridge_hv3": "hv3", "lgbm_hv3": "hv3", "ridge_v1": "all15", "lgbm_v1": "all15"}  # 每個選定設定所屬的特徵組
summ["selected"] = summ.apply(lambda r: (r.fset == ACT and r.config in ("hv", "train_mean")) or
                              any(r.fset == SELF[k] and r.config == c for k, c in SEL.items()), axis=1)  # 標記選中
summ.sort_values(["fset", "family", "MAE"]).round(4).to_csv(TAB/"model_grid_validation.csv", index=False)  # 存全部試驗結果
res["decision"] = np.where(res.config.isin(SEL.values()) | res.family.isin(["hv", "train_mean"]), "selected/baseline", "rejected")  # 記錄決策
if ACT != "all15": res.loc[res.fset == "all15", "decision"] = "rerun_v1_seal_fix"  # v1 重算：同一組設定，不是新試驗
res["trial_family"] = "model_" + res.family + "_" + res.fset  # 試驗家族
log_runs(res.to_dict("records"))                             # 寫入 run_log.csv
print("SELECTED:", SEL)                                      # 印出選中的設定

# ---------- 回看期 10/20/40 穩定性（只對選中的設定，特徵同步調整） ----------
lk = []                                                      # 收集結果
for L in ((10, 20, 40) if ACT == "all15" else ()):           # 三種回看期（只對 v1 定義；v2 特徵窗長固定）
    for kind, cfgn in (("hv", "hv"), ("ridge", SEL["ridge"]), ("lgbm", SEL["lgbm"])):  # 三種方法
        prm = dict(GRID[[cfg_name(*g) for g in GRID].index(cfgn)][1])  # 取回參數
        for k, fd in enumerate(FOLDS):                       # 三個驗證 fold
            tr, te = split(df.dropna(subset=feat_cols(L)), fd["train"][1], *fd["valid"])  # 只用該回看期特徵齊全的列
            p, _ = fit_predict(kind, prm, tr, te, feat_cols(L))  # 訓練與預測
            lk.append({"family": kind, "config": cfgn, "fset": "all15", "lookback": L, "fold": f"fold{k+1}",
                       "eval_window": f"{fd['valid'][0]}~{fd['valid'][1]}", "trial_family": "lookback", "decision": "robustness", **metrics(te, p)})  # 記錄
if lk:                                                       # 有跑回看期時
    lk = pd.DataFrame(lk); log_runs(lk.to_dict("records"))   # 寫入紀錄
    lk.groupby(["family", "lookback"])[["MAE", "RMSE", "IC"]].mean().round(4).to_csv(TAB/"lookback_robustness.csv")  # 存檔
    print(lk.groupby(["family", "lookback"])[["MAE", "IC"]].mean().round(4))  # 印出

# ---------- 選中模型的驗證預測、分年度、分股票表 ----------
P = pd.concat(preds)                                         # 所有預測
keep = {"hv": ("hv", ACT), "train_mean": ("train_mean", ACT), **{k: (c, SELF[k]) for k, c in SEL.items()}}  # 要保留的模型
oos = pd.concat([P[(P.config == c) & (P.fset == f)].assign(model=m) for m, (c, f) in keep.items()])  # 選出這些模型的預測
oos.to_parquet(vproc("oos_validation.parquet"), index=False)  # 存檔（含 2021 校準 fold；v2 另存檔名）
v = oos[oos.fold != "calib2021"]                             # 三個驗證年度
tab = v.groupby(["model", "fold"]).apply(lambda g: pd.Series(metrics(g, g.p.values)), include_groups=False).reset_index()  # 每模型每年指標
tab.round(4).to_csv(TAB/"validation_by_year.csv", index=False)  # 存分年度表
pooled = v.groupby("model").apply(lambda g: pd.Series(metrics(g, g.p.values)), include_groups=False)  # 三年合併指標
pooled.round(4).to_csv(TAB/"validation_pooled.csv")          # 存檔
bys = v.assign(ae=(v.p - v.y).abs()).pivot_table(index="sid", columns="model", values="ae", aggfunc="mean")  # 分股票 MAE
bys.round(4).to_csv(TAB/"validation_mae_by_stock.csv")      # 存檔
print(pooled[["MAE", "RMSE", "IC", "IC_lo", "IC_hi", "TS_corr_level"]].round(4))  # 印出

# ---------- 成對比較：每一階 vs 前一階（以決策日為單位，HAC 區間） ----------
def daily(m):                                                # 定義：每個決策日的平均絕對誤差與 IC
    g = v[v.model == m].groupby("date")                      # 依日期分組
    return pd.DataFrame({"ae": g.apply(lambda x: (x.p - x.y).abs().mean(), include_groups=False),
                         "ic": g.apply(lambda x: x.p.rank().corr(x.y.rank()), include_groups=False)})  # 兩個序列
pairs = [("train_mean", "hv"), ("hv", "ridge"), ("ridge", "lgbm"), ("hv", "lgbm"), ("ridge_hv3", "ridge"), ("lgbm_hv3", "lgbm")]  # 比較組合
if ACT != "all15": pairs += [("ridge_v1", "ridge"), ("lgbm_v1", "lgbm")]  # v1 → v2
pc = []                                                      # 收集
for a, b in pairs:                                           # 逐一比較
    A, B = daily(a), daily(b)                                # 兩個模型的每日序列
    dm = nw_mean_ci((B.ae - A.ae).values); di = nw_mean_ci((B.ic - A.ic).values)  # 差異與 HAC 區間
    pc.append({"from": a, "to": b, "dMAE": dm[0], "dMAE_lo": dm[1], "dMAE_hi": dm[2], "dIC": di[0], "dIC_lo": di[1], "dIC_hi": di[2],
               "MAE_improves": dm[2] < 0, "IC_improves": di[1] > 0})  # 區間不含 0 才算改善
pd.DataFrame(pc).round(4).to_csv(TAB/"paired_comparison.csv", index=False)  # 存檔
print(pd.DataFrame(pc).round(4))                             # 印出

# ---------- SHAP：LightGBM 各 fold 的重要度與方向是否穩定 ----------
prm = dict(GRID[[cfg_name(*g) for g in GRID].index(SEL["lgbm"])][1])  # 選中的 LightGBM 參數
cols = feat_cols(20, ACT); imp, sgn = {}, {}                 # 特徵欄位與收集用字典
for k, fd in enumerate(FOLDS):                               # 逐一 fold
    tr, te = split(df, fd["train"][1], *fd["valid"])         # 切資料
    m = make_model("lgbm", prm).fit(tr[cols], tr.y)          # 訓練
    sv = shap.TreeExplainer(m).shap_values(te[cols])         # 計算 SHAP 值
    imp[f"fold{k+1}"] = np.abs(sv).mean(0)                   # 平均絕對 SHAP（重要度）
    sgn[f"fold{k+1}"] = [pd.Series(te[c].values).corr(pd.Series(sv[:, i])) for i, c in enumerate(cols)]  # 特徵值與 SHAP 的相關（方向；略過缺值）
I = pd.DataFrame(imp, index=[c.replace("_L20", "") for c in cols]); S = pd.DataFrame(sgn, index=I.index)  # 轉成表格
out = I.add_prefix("meanabs_").join(S.add_prefix("sign_corr_"))  # 合併
out["rank_fold1"], out["rank_fold3"] = I.fold1.rank(ascending=False), I.fold3.rank(ascending=False)  # 各 fold 排名
out.sort_values("meanabs_fold3", ascending=False).round(4).to_csv(TAB/"shap_stability_folds.csv")  # 存檔
print("SHAP rank corr fold1-fold2-fold3:", I.rank().corr(method="spearman").round(3).values.tolist())  # 印出跨 fold 排名相關
pd.Series(SEL).to_csv(TAB/"selected_configs.csv", header=["config"])  # 存選定設定
