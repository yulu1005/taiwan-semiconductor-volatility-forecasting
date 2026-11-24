# 07_holdout.py — 封存測試（2025-01-01～2026-11-01）：設定凍結後只執行一次
import datetime as dt, numpy as np, pandas as pd             # 載入：時間、數值運算、表格
from common import ROOT, CFG, PROC, TAB, nw_mean_ci          # 載入：設定、路徑、HAC 工具
from models import load, metrics, log_runs                   # 載入：樣本、指標、紀錄
from strategies import predict_period, thresholds, make_target  # 載入：預測、門檻、目標權重
from bt import Market, run, perf, block_boot_sharpe_diff     # 載入：回測引擎

FLAG = ROOT/"outputs"/"HOLDOUT_USED.txt"                     # 封存測試使用紀錄檔
if FLAG.exists(): raise SystemExit("封存測試已經執行過；若要重跑必須在研究帳本中記錄並說明。")  # 防止重複使用
H = CFG["protocol"]["holdout"]                               # 封存設定
TE, T0, T1 = H["train"][1], *H["test"]                       # 訓練終點、測試起點、測試終點
df = load(); mk = Market()                                   # 樣本與市場資料
SEL = pd.read_csv(TAB/"selected_configs.csv", index_col=0).config  # 驗證期選定（已凍結）

pred = {m: predict_period(df, "hv" if m == "hv" else SEL[m], TE, T0, T1) for m in ("hv", "ridge", "lgbm")}  # 最終模型預測
thr = thresholds(df, TE, T0)                                 # 門檻：只用 2016-11～2024 訓練資料
for m, p in pred.items(): p.to_parquet(PROC/f"bt_pred_hold_{m}.parquet", index=False)  # 存預測

# ---------- 預測評估（標籤結束日不得晚於研究終點） ----------
ev = {m: p[p.y.notna() & (p.label_end <= pd.Timestamp(T1))] for m, p in pred.items()}  # 有完整標籤的列
tm = df[df.y.notna() & (df.label_end < pd.Timestamp(T0)) & (df.date >= pd.Timestamp(CFG["data"]["study_start"]))].y.mean()  # 訓練期平均
ev["train_mean"] = ev["hv"].assign(p=tm)                     # 訓練平均基準
hm = pd.DataFrame({m: metrics(e, e.p.values) for m, e in ev.items()}).T  # 各模型指標
hm.round(4).to_csv(TAB/"holdout_prediction_metrics.csv")     # 存檔
pc = []                                                      # 成對比較
def daily(e): g = e.groupby("date"); return pd.DataFrame({"ae": g.apply(lambda x: (x.p - x.y).abs().mean(), include_groups=False),
                                                          "ic": g.apply(lambda x: x.p.rank().corr(x.y.rank()), include_groups=False)})  # 每日誤差與 IC
for a, b in [("train_mean", "hv"), ("hv", "ridge"), ("ridge", "lgbm"), ("hv", "lgbm")]:  # 比較組合
    A, B = daily(ev[a]), daily(ev[b]); dm = nw_mean_ci((B.ae - A.ae).values); di = nw_mean_ci((B.ic - A.ic).values)  # 差異與區間
    pc.append({"from": a, "to": b, "dMAE": dm[0], "dMAE_lo": dm[1], "dMAE_hi": dm[2], "dIC": di[0], "dIC_lo": di[1], "dIC_hi": di[2]})  # 記錄
pd.DataFrame(pc).round(4).to_csv(TAB/"holdout_paired_comparison.csv", index=False)  # 存檔
print(hm[["MAE", "RMSE", "IC", "IC_lo", "IC_hi", "TS_corr_level"]].round(4)); print(pd.DataFrame(pc).round(4))  # 印出

# ---------- 回測（與驗證期同一套程式與規則） ----------
grid = pd.read_csv(TAB/"decision_dates.csv", parse_dates=["decision_date"]).decision_date  # 決策日
end = mk.dates[mk.dates <= pd.Timestamp(T1)][-1]             # 最後交易日（≤ 研究終點）
dec = list(grid[(grid >= T0) & (grid < end)])                # 封存期決策日
f = lambda d: thr                                            # 門檻固定
S = {"EW100_股票等權": dict(fn=make_target(pred["hv"], None, static="ew100")),
     "Fixed_50-20-30": dict(fn=make_target(pred["hv"], None, static="fixed")),
     "Passive_0050": dict(fn=make_target(pred["hv"], None, static="bm"), cap=False)}  # 基準
for m in ("hv", "ridge", "lgbm"):                            # 三種預測
    for a in ("ew", "iv"): S[f"{m.upper()}_dyn_{a.upper()}"] = dict(fn=make_target(pred[m], f, alloc=a))  # 動態配置
S["LGBM_dyn_EW_noGold"] = dict(fn=make_target(pred["lgbm"], f, gold=0.0))  # 無黃金
S["LGBM_dyn_EW_overlay15"] = dict(fn=make_target(pred["lgbm"], f), overlay=True)  # 停損覆蓋
EQ, PERF, TIER = {}, [], None                                # 收集
for name, kw in S.items():                                   # 逐一策略
    fn = kw.pop("fn"); eq, tr = run(mk, dec, fn, end=end, **kw)  # 回測
    EQ[name] = eq; PERF.append({"strategy": name, **perf(eq, tr)}); tr.to_csv(TAB/f"trades_hold_{name}.csv", index=False)  # 存結果
    if name == "LGBM_dyn_EW": TIER = pd.DataFrame(fn.log)   # 主策略風險等級
pf = pd.DataFrame(PERF).set_index("strategy"); pf.round(4).to_csv(TAB/"backtest_holdout_perf.csv", encoding="utf-8-sig")  # 存績效
pd.concat({k: v for k, v in EQ.items()}, axis=0).to_parquet(PROC/"equity_hold_full.parquet")  # 存每日狀態
TIER.to_csv(TAB/"tier_log_hold_primary.csv", index=False)    # 存風險等級
sweep = []                                                   # 成本掃描
for name, fa in (("LGBM_dyn_EW", (pred["lgbm"], f)), ("HV_dyn_EW", (pred["hv"], f)), ("EW100_股票等權", None)):  # 三個策略
    for mult in [None] + CFG["costs"]["sensitivity_multipliers"]:  # 毛報酬與各倍數
        fn = make_target(*fa) if fa else make_target(pred["hv"], None, static="ew100")  # 目標函數
        eq, tr = run(mk, dec, fn, end=end, gross=(mult is None), cost_mult=(mult or 0))  # 回測
        sweep.append({"strategy": name, "cost_mult": "gross" if mult is None else mult, "per_leg_bp_ex_tax": 0 if mult is None else 19.25 * mult, **perf(eq, tr)})  # 記錄
pd.DataFrame(sweep).to_csv(TAB/"cost_sweep_holdout.csv", index=False)  # 存檔
R = pd.DataFrame({k: v.value for k, v in EQ.items()}).pct_change().dropna()  # 日報酬
bs = [dict(zip(["A", "B", "dSharpe", "lo95", "hi95"], (a, b, *block_boot_sharpe_diff(R[a], R[b]))))
      for a, b in [("LGBM_dyn_EW", "HV_dyn_EW"), ("LGBM_dyn_EW", "Fixed_50-20-30"), ("LGBM_dyn_EW", "EW100_股票等權"), ("LGBM_dyn_EW", "Passive_0050"),
                   ("HV_dyn_EW", "Fixed_50-20-30"), ("LGBM_dyn_EW", "LGBM_dyn_IV"), ("LGBM_dyn_EW", "LGBM_dyn_EW_noGold")]]  # 重抽樣比較
pd.DataFrame(bs).round(3).to_csv(TAB/"sharpe_diff_bootstrap_holdout.csv", index=False)  # 存檔
log_runs([{"trial_family": "HOLDOUT", "family": "holdout", "config": k, "fold": "holdout", "eval_window": f"{T0}~{end.date()}",
           "decision": "confirmatory (single use)", **{c: pf.loc[k, c] for c in ["Sharpe", "CAGR", "MDD", "turnover_2way"]}} for k in pf.index]
         + [{"trial_family": "HOLDOUT", "family": m, "config": "final", "fold": "holdout", "eval_window": f"{T0}~{T1}",
             "decision": "confirmatory (single use)", **hm.loc[m].to_dict()} for m in hm.index])  # 寫入 run_log
FLAG.write_text(f"holdout used once at {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n")  # 標記已使用
print(pf[["CAGR", "ann_vol", "Sharpe", "Sharpe_lo", "Sharpe_hi", "MDD", "turnover_2way", "avg_stock_w", "avg_gold_w"]].round(4).to_string())  # 印出
print(pd.DataFrame(bs).round(3)); print(TIER.tier.value_counts())  # 印出
