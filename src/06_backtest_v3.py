# 06_backtest_v3.py — v3 理想策略的驗證期回測（2022–2024）【已寫好、尚未執行；要回測時才跑】
# 先決條件：python src/05_validate_v3.py 已跑完（產生 outputs/v3/tables/selected_configs.csv、strategy_choice.csv）
# 規則全部來自 setup.yaml 的 v3 區塊（事先寫定）：
#   每週最後交易日收盤決策 → 下一交易日開盤成交；只交易通過 3.5 億流動性門檻的股票
#   第一層：反波動率權重、單檔 ≤ 20%、超出部分重新分配；第二層：三級（股票, 黃金）＝(70,15)/(50,20)/(30,20)，其餘現金
#   滑價＝max(0.05%, 半個升降單位 ÷ 成交價)
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, VTAB, VER, vproc                     # 載入：設定、v3 輸出位置
from models import load, log_runs                            # 載入：樣本與帳本
from strategies import predict_period, thresholds, make_target  # 載入：預測、門檻、目標權重
from bt import Market, run, perf, block_boot_sharpe_diff     # 載入：回測引擎
assert VER == "v3", "請在 setup.yaml 設 features.active: v3"   # 只用於 v3

V3 = CFG["v3"]; PO = V3["portfolio"]; DEC = "is_decision_wk"  # v3 設定與決策日欄位
TIERS = tuple(tuple(PO["risk_tiers"][k]) for k in ("low", "mid", "high"))  # ((0.70,0.15),(0.50,0.20),(0.30,0.20))
RATIO = (PO["tier_rules"]["ratio"]["lo"], PO["tier_rules"]["ratio"]["hi"])  # R_t 門檻 0.9、1.1
Q = (PO["tier_rules"]["level"]["lo_pct"], PO["tier_rules"]["level"]["hi_pct"])  # 水準門檻分位數 33、67
SEL = pd.read_csv(VTAB / "selected_configs.csv", index_col=0).config  # 05_validate_v3 選定的各家族最佳
CHOICE = pd.read_csv(VTAB / "strategy_choice.csv", index_col=0).value  # 依事先準則選定的風險等級規則與主模型
RULE, PRIMARY = CHOICE["tier_rule"], CHOICE["primary_model"]  # 例如 level、lasso_alpha_frac0.1

df = load(); mk = Market()                                   # 樣本與市場資料
FOLDS = CFG["protocol"]["folds"]; END = pd.Timestamp("2024-12-31")  # 三個驗證年度
dec = sorted(df.loc[df[DEC].fillna(False).astype(bool) & (df.date >= "2022-01-01") & (df.date <= END), "date"].unique())  # 驗證期每週決策日

def build(cfg):                                              # 定義：三個 fold 串接成 2022–2024 的預測與門檻
    preds, thr = [], {}                                      # 收集
    for fd in FOLDS:                                         # 逐一 fold（每年用前一年底以前的資料重新訓練）
        preds.append(predict_period(df, cfg, fd["train"][1], *fd["valid"], fset="v3", dec_col=DEC, tradable_only=True))  # 該年預測（只留可交易股票）
        thr[fd["valid"][0][:4]] = thresholds(df, fd["train"][1], fd["valid"][0], Q, dec_col=DEC, tradable_only=True)  # 該年門檻（只用訓練期）
    return pd.concat(preds), (lambda d: thr[str(pd.Timestamp(d).year)])  # 回傳預測與門檻查詢

MODELS = ["hv"] + list(dict.fromkeys(SEL.drop("primary").values))  # 回測族群：HV20 + 各家族最佳（事先寫定）
P = {m: build(m) for m in MODELS}                            # 每個模型的預測
for m in P: P[m][0].to_parquet(vproc(f"bt_pred_val_{m}.parquet"), index=False)  # 存預測
base = P["hv"][0]                                            # 基準策略只需要「當週可交易股票清單」
S = {"EW100_股票等權": dict(fn=make_target(base, None, static="ew100")),
     "Fixed_50-20-30": dict(fn=make_target(base, None, static="fixed", tiers=TIERS, alloc="iv_cap"), cap=False),
     "Passive_0050": dict(fn=make_target(base, None, static="bm"), cap=False)}  # 三個基準
for m in MODELS:                                             # 每個模型 × 兩種規則（主規則之外的那個當作對照）
    for rule in ("level", "ratio"):                          # 兩種規則
        S[f"{m}_{rule}"] = dict(fn=make_target(P[m][0], P[m][1], tiers=TIERS, alloc="iv_cap", rule=rule, ratio=RATIO), cap=False)  # 動態策略
PM = f"{PRIMARY}_{RULE}"                                     # 事先指定的主策略

EQ, PERF, LOGS = {}, [], {}                                  # 收集
for name, kw in S.items():                                   # 逐一策略
    fn = kw.pop("fn")                                        # 目標函數
    eq, tr = run(mk, dec, fn, end=END, slip_model="tick", **kw)  # 回測（升降單位滑價）
    EQ[name] = eq; PERF.append({"strategy": name, **perf(eq, tr)})  # 存結果
    if fn.log: LOGS[name] = pd.DataFrame(fn.log)             # 存風險等級
    print(name, round(PERF[-1]["Sharpe"], 3), round(PERF[-1]["CAGR"], 4), round(PERF[-1]["MDD"], 4))  # 印出
pf = pd.DataFrame(PERF).set_index("strategy"); pf.round(4).to_csv(VTAB / "backtest_validation_perf.csv", encoding="utf-8-sig")  # 存績效
pd.concat({k: v.value for k, v in EQ.items()}, axis=1).to_parquet(vproc("equity_val.parquet"))  # 存淨值
LOGS[PM].to_csv(VTAB / "tier_log_val_primary.csv", index=False)  # 存主策略風險等級

R = pd.DataFrame({k: v.value for k, v in EQ.items()}).pct_change().dropna()  # 日報酬
bs = [dict(zip(["A", "B", "dSharpe", "lo95", "hi95"], (PM, b, *block_boot_sharpe_diff(R[PM], R[b])))) for b in R.columns if b != PM]  # 主策略 vs 其他
pd.DataFrame(bs).round(3).to_csv(VTAB / "sharpe_diff_bootstrap_validation.csv", index=False)  # 存檔
log_runs([{"trial_family": "backtest_v3", "family": "backtest", "config": k, "fset": "v3", "fold": "val2022-2024",
           "eval_window": "2022-01-01~2024-12-31", "decision": "primary" if k == PM else "comparison",
           **{c: pf.loc[k, c] for c in ["Sharpe", "CAGR", "MDD", "turnover_2way"]}} for k in pf.index])  # 每個策略都算一次試驗
print(pf[["CAGR", "ann_vol", "Sharpe", "MDD", "turnover_2way", "cost_drag_ann", "avg_stock_w", "avg_gold_w"]].round(4).to_string())  # 印出
