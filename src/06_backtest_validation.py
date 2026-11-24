# 06_backtest_validation.py — 驗證期（2022–2024）回測：策略比較、成本敏感度、變體、風險（不碰封存測試）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC, TAB, VTAB, VER, vproc         # 載入：設定、共用表格（決策日）、版本化輸出
ACT = "all15" if VER == "v1" else VER                        # 目前特徵組
from models import load, log_runs                            # 載入：樣本與紀錄工具
from strategies import predict_period, thresholds, make_target  # 載入：預測、門檻、目標權重
from bt import Market, run, perf, block_boot_sharpe_diff, STK, GOLD, BM  # 載入：回測引擎

df = load()                                                  # 合格樣本
mk = Market()                                                # 市場資料矩陣
SEL = pd.read_csv(VTAB/"selected_configs.csv", index_col=0).config  # 驗證選定的模型設定
FOLDS = CFG["protocol"]["folds"]                             # 三個驗證 fold
PRIMARY = ("lgbm", "ew")                                     # 事先指定主策略：驗證 MAE 最低的模型 + 等權（虛無配置）

def build(model_key, q=(33, 67)):                            # 定義：三個 fold 串接成 2022–2024 的預測與門檻
    preds, thr = [], {}                                      # 收集
    for fd in FOLDS:                                         # 逐一 fold
        cfg = "hv" if model_key == "hv" else SEL[model_key]  # 模型設定
        preds.append(predict_period(df, cfg, fd["train"][1], *fd["valid"], fset=ACT))  # 該年預測
        thr[fd["valid"][0][:4]] = thresholds(df, fd["train"][1], fd["valid"][0], q)  # 該年門檻（只用訓練期）
    return pd.concat(preds), (lambda d: thr[str(d.year)]), thr  # 回傳預測、門檻查詢函數、門檻表

grid = pd.read_csv(TAB/"decision_dates.csv", parse_dates=["decision_date"]).decision_date  # 全部決策日
dec = list(grid[(grid >= "2022-01-01") & (grid <= "2024-12-31")])  # 驗證期決策日
END = pd.Timestamp("2024-12-31")                             # 驗證期終點
P = {m: build(m) for m in ("hv", "ridge", "lgbm")}           # 三種方法的預測與門檻
pd.DataFrame([{"model": m, "year": y, "thr_lo_p33": v[0], "thr_hi_p67": v[1]} for m in P for y, v in P[m][2].items()]).round(4).to_csv(
    VTAB/"tier_thresholds_validation.csv", index=False)       # 存門檻（長表格式）
for m in P: P[m][0].to_parquet(vproc(f"bt_pred_val_{m}.parquet"), index=False)  # 存預測

S = {"EW100_股票等權": dict(fn=make_target(P["hv"][0], None, static="ew100")),
     "Fixed_50-20-30": dict(fn=make_target(P["hv"][0], None, static="fixed")),
     "Passive_0050": dict(fn=make_target(P["hv"][0], None, static="bm"), cap=False)}  # 基準策略
for m in ("hv", "ridge", "lgbm"):                            # 三種預測
    for a in ("ew", "iv"):                                   # 兩種股內配置
        S[f"{m.upper()}_dyn_{a.upper()}"] = dict(fn=make_target(P[m][0], P[m][1], alloc=a))  # 動態配置策略
pm = f"{PRIMARY[0].upper()}_dyn_{PRIMARY[1].upper()}"        # 主策略名稱
S[pm + "_noGold"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1], gold=0.0))  # 變體：無黃金（黃金比例歸現金）
S[pm + "_60-50-40"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1], tiers=(0.6, 0.5, 0.4)))  # 變體：調整幅度較小
q25 = build("lgbm", (25, 75))                                # 25/75 分位數門檻
S[pm + "_thr25-75"] = dict(fn=make_target(q25[0], q25[1]))   # 變體：門檻 25/75
S[pm + "_noCap_noADV"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1]), cap=False, adv_cap=0)  # 變體：無個股上限、無容量限制
S[pm + "_ADV1pct"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1]), adv_cap=0.01)  # 變體：容量 1%
S[pm + "_overlay15"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1]), overlay=True)  # 變體：回撤 15% 停止加碼
S[pm + "_capital1M"] = dict(fn=make_target(P["lgbm"][0], P["lgbm"][1]), start_cash=1_000_000)  # 變體：本金 100 萬（備註版）

EQ, PERF, TIERS = {}, [], {}                                 # 收集淨值、績效、風險等級
for name, kw in S.items():                                   # 逐一策略
    fn = kw.pop("fn")                                        # 取出目標函數
    eq, tr = run(mk, dec, fn, end=END, **kw)                 # 執行回測
    EQ[name] = eq; PERF.append({"strategy": name, **perf(eq, tr)})  # 存淨值與績效
    if fn.log: TIERS[name] = pd.DataFrame(fn.log)            # 存風險等級
    tr.to_csv(VTAB/f"trades_val_{name}.csv", index=False)     # 存交易明細
    print(name, round(PERF[-1]["Sharpe"], 3), round(PERF[-1]["CAGR"], 4), round(PERF[-1]["MDD"], 4))  # 印出摘要
pf = pd.DataFrame(PERF).set_index("strategy")                # 績效表
pd.concat({k: v.value for k, v in EQ.items()}, axis=1).to_parquet(vproc("equity_val.parquet"))  # 存淨值
pd.concat({k: v for k, v in EQ.items()}, axis=0).to_parquet(vproc("equity_val_full.parquet"))  # 存完整每日狀態
TIERS[pm].to_csv(VTAB/"tier_log_val_primary.csv", index=False)  # 存主策略的風險等級紀錄

# ---------- 毛報酬與成本掃描（主策略、歷史波動規則、等權基準） ----------
sweep = []                                                   # 收集
for name, fnargs in ((pm, (P["lgbm"][0], P["lgbm"][1])), ("HV_dyn_EW", (P["hv"][0], P["hv"][1])), ("EW100_股票等權", None)):  # 三個策略
    for mult in [None] + CFG["costs"]["sensitivity_multipliers"]:  # None＝完全毛報酬（不含稅）
        fn = make_target(*fnargs) if fnargs else make_target(P["hv"][0], None, static="ew100")  # 目標函數
        eq, tr = run(mk, dec, fn, end=END, gross=(mult is None), cost_mult=(mult or 0))  # 回測
        x = perf(eq, tr); sweep.append({"strategy": name, "cost_mult": "gross" if mult is None else mult,
                                        "per_leg_bp_ex_tax": 0 if mult is None else (14.25 + 5) * mult, **x})  # 記錄
sw = pd.DataFrame(sweep); sw.to_csv(VTAB/"cost_sweep_validation.csv", index=False)  # 存檔
be = []                                                      # 兩平成本
for name, g in sw.groupby("strategy"):                       # 逐一策略
    gr = g[g.cost_mult == "gross"].iloc[0]                   # 毛報酬那一列
    be.append({"strategy": name, "gross_CAGR": gr.CAGR, "turnover_2way": gr.turnover_2way,
               "breakeven_bp_per_leg": gr.CAGR / gr.turnover_2way * 1e4,  # c* = 毛報酬 ÷ 換手率
               "assumed_bp_per_leg_incl_tax_avg": (14.25 + 5) + 30 / 2,  # 假設成本：手續費+滑價每邊 19.25bp，稅 30bp 平均到兩邊
               "gross_Sharpe": gr.Sharpe, "net_Sharpe": g[g.cost_mult == 1].iloc[0].Sharpe})  # 毛與淨 Sharpe
pd.DataFrame(be).round(4).to_csv(VTAB/"breakeven_validation.csv", index=False)  # 存檔
print(pd.DataFrame(be).round(4))                             # 印出

# ---------- Sharpe 差異的週區塊重抽樣區間 ----------
R = pd.DataFrame({k: v.value for k, v in EQ.items()}).pct_change().dropna()  # 各策略日報酬
bs = []                                                      # 收集
for a, b in [(pm, "HV_dyn_EW"), (pm, "Fixed_50-20-30"), (pm, "EW100_股票等權"), (pm, "Passive_0050"), ("HV_dyn_EW", "Fixed_50-20-30"),
             ("RIDGE_dyn_EW", "HV_dyn_EW"), (pm, "LGBM_dyn_IV"), (pm, pm + "_noGold")]:  # 比較組合
    d, lo, hi = block_boot_sharpe_diff(R[a], R[b]); bs.append({"A": a, "B": b, "dSharpe": d, "lo95": lo, "hi95": hi})  # 計算
pd.DataFrame(bs).round(3).to_csv(VTAB/"sharpe_diff_bootstrap_validation.csv", index=False)  # 存檔
print(pd.DataFrame(bs).round(3))                             # 印出

pf.round(4).to_csv(VTAB/"backtest_validation_perf.csv", encoding="utf-8-sig")  # 存績效表
log_runs([{"trial_family": "backtest_" + ("variant" if k.startswith(pm + "_") else "strategy"), "family": "backtest", "config": k, "fset": ACT,
           "fold": "val2022-2024", "eval_window": "2022-01-01~2024-12-31", "decision": "primary" if k == pm else "comparison",
           **{c: pf.loc[k, c] for c in ["Sharpe", "CAGR", "MDD", "turnover_2way"]}} for k in pf.index])  # 寫入 run_log
print(pf[["CAGR", "ann_vol", "Sharpe", "Sharpe_lo", "Sharpe_hi", "MDD", "turnover_2way", "cost_drag_ann", "avg_stock_w", "avg_gold_w", "unfilled_avg"]].round(4).to_string())  # 印出
