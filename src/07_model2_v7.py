# 07_model2_v7.py — 依 V7 說明書執行 Model 2 的選擇程序與回測
#   順序（V7 5.5）：① 相關性窗口（20／60 日，策略 B，QLIKE）→ ② 目標波動率（策略 A，2017–2020 中位數）
#                  → ③ α（策略 D，2022–2024 平均扣成本 Sharpe）→ ④ 凍結規則，回測 A、B、C、D
#   V7 未規定、本程式採用的處理（皆在看結果前寫定，並列於報告）：
#     a. 可交易但沒有 Model 1 預測（GARCH 未滿 250 日）的股票，當週四組都不納入（維持同一份 Uₜ）
#     b. 相關係數窗口內有停牌缺值：整列刪除（V7 5.3「日期對齊、不得有缺值」）；剩餘天數 < 0.75L 時記為估計失敗，
#        改用該窗口的成對相關（pairwise）後一樣做常數相關收縮
#     c. 例外週（Nₜ < 4）不納入目標波動率中位數（V7 4.3：「不視為正常 Model 2 結果」）
#     d. 組合實現波動（5.5）若任一檔在 5 段中有缺值，該週不納入窗口比較
#     e. 回測：股數可為小數；成交日停牌的股票無法交易、維持原持股；其餘全數成交（V7 8.4 未設參與率上限）
#     f. 2025–2026 事後評估使用訓練至 2024 年底的 Model 1（V7 未規定重訓頻率）
import json, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis                 # 載入：Deflated Sharpe 需要的分配函數
from common import CFG, PROC, VTAB, VER                      # 載入：設定與路徑
from models import load, split, make_model, FLOOR, log_runs
from model2_v7 import window_correlation, allocate
from model2_execution import execute_plan
from model2_historical_rate import deposit_rate
assert VER == "v7"
OUT = VTAB / "model2"; OUT.mkdir(exist_ok=True)              # 輸出資料夾
STRAT = {"A": ("hv", "fixed0"), "B": ("ml", "fixed0"), "C": ("hv", "chosen"), "D": ("ml", "chosen")}  # 四組策略：波動來源、α
CAPITAL = 3e8                                                # 3 億元
COST = dict(comm=0.001425, tax=0.003, slip=0.0005)           # V7 8.4：手續費、證交稅、滑價 5 bps

# ---------- 0. 資料 ----------
df = load()                                                  # 合格樣本（含特徵、可交易、每週決策日）
panel = pd.read_parquet(PROC / "panel.parquet")
cal = pd.DatetimeIndex(sorted(panel[(panel.sid == "0050") & panel.valid].date.unique()))  # 交易日曆
O = panel[panel.valid].pivot(index="date", columns="sid", values="aopen").reindex(cal)  # 還原開盤（停牌為空）
Cl = panel[panel.valid].pivot(index="date", columns="sid", values="aclose").reindex(cal)  # 還原收盤
RO = np.log(O / O.shift(1))                                  # 開盤到開盤日報酬（相鄰交易日都有效才有值）

# ---------- 1. Model 1 預測（主模型逐年重訓：訓練至前一年底，預測當年每個決策日） ----------
SEL = pd.read_csv(VTAB / "selected_configs.csv", index_col=0).config["primary"]  # 05_model1_v7 選出的主模型
assert SEL.startswith("lasso_alpha_frac"), SEL                # 本程式只處理 05 選出的 Lasso
kind, prm = "lasso", {"alpha_frac": float(SEL.split("frac")[1])}  # 模型種類與超參數
COLS = list(CFG["features"]["v7"])
dec = df[df.is_decision_wk.fillna(False).astype(bool)].copy()  # 每週決策日的列
preds = []
for Y, tr_end, t0, t1 in [(2021, "2020-12-31", "2021-01-01", "2021-12-31"), (2022, "2021-12-31", "2022-01-01", "2022-12-31"),
                          (2023, "2022-12-31", "2023-01-01", "2023-12-31"), (2024, "2023-12-31", "2024-01-01", "2024-12-31"),
                          (2025, "2024-12-31", "2025-01-01", "2026-11-01")]:  # 2025 列代表 2025-01～2026-10 事後評估
    tr, _ = split(df, tr_end, t0, t1, dec_col="is_decision_wk"); tr = tr[tr.garch_oo5.notna()]  # 與 05 相同的訓練集
    m = make_model(kind, prm).fit(tr[COLS], tr.y)
    te = dec[(dec.date >= t0) & (dec.date <= t1) & dec.garch_oo5.notna()]  # 不論標籤是否已知，都產生預測（回測需要）
    preds.append(te[["date", "sid"]].assign(ml=np.maximum(m.predict(te[COLS]), FLOOR)))
ML = pd.concat(preds)
dec = dec.merge(ML, on=["date", "sid"], how="left")
ML.to_parquet(PROC / "model2_sigma_ml_v7.parquet", index=False)  # 存下 Model 2 用的 Model 1 預測（12_correlation_checks 也會用）
chk = pd.read_parquet(PROC / "oos_predictions_v7.parquet")   # 與 05 的樣本外預測核對
chk = chk[chk.config == SEL][["date", "sid", "p"]].merge(dec[["date", "sid", "ml"]], on=["date", "sid"])
print("與 Model 1 樣本外預測的最大差距：", float((chk.p - chk.ml).abs().max()), "筆數", len(chk))

def universe(t, need_ml):                                    # 定義：決策日 t 的當週可交易名單 Uₜ 與兩種波動
    x = dec[(dec.date == t) & dec.tradable.fillna(False).astype(bool) & dec.rv20.notna() & dec.garch_oo5.notna()]  # 規則 a
    if need_ml: x = x[x.ml.notna()]
    return x.sort_values("sid")

_CC = {}                                                     # 相關矩陣快取（同一天、同名單、同窗口只算一次）
def window_corr(t, sids, L, shrink=True):                    # 定義：決策日 t 以前（含）L 個交易日的相關矩陣（規則 b）
    k = (t, tuple(sids), L, shrink)
    if k not in _CC: _CC[k] = _window_corr(t, sids, L, shrink)
    return _CC[k]
def _window_corr(t, sids, L, shrink):
    i = cal.get_loc(t); W = RO.iloc[max(0, i - L + 1): i + 1][sids]
    return window_correlation(W, L, shrink)

DECS = sorted(dec.date.unique())
_PC = {}
def weekly_plan(src, alpha, target, L, years, shrink=True):  # 定義：一組設定在多個決策日的目標權重（有快取）
    k = (src, alpha, target, L, tuple(sorted(years)), shrink)
    if k not in _PC: _PC[k] = _weekly_plan(*k[:4], set(k[4]), shrink)
    return _PC[k]
def _weekly_plan(src, alpha, target, L, years, shrink):
    out = []
    for t in DECS:
        if t.year not in years: continue
        x = universe(t, need_ml=(t.year >= 2021))            # 規則 a：2021 年起四組使用同一份名單（皆需有 Model 1 預測）；2017–2020 只有策略 A（定目標波動率），不需 ML
        if len(x) == 0: continue
        sig = (x.ml if src == "ml" else x.rv20).values.astype(float)
        R, lam, nobs, ev, fail = window_corr(t, list(x.sid), L, shrink)
        res = allocate(sig, R, alpha, target)
        out.append(dict(date=t, sids=list(x.sid), sig=sig, R=R, lam=lam, nobs=nobs, min_eig=ev, est_fail=fail, **res))
    return out

# ---------- 2. 相關性窗口（V7 5.5）：策略 B，2022–2024，QLIKE ----------
def realized_fixed_weight(t, sids, q):                       # 5.5 式：固定權重組合的 5 日開盤到開盤實現波動（規則 d）
    i = cal.get_loc(t)
    if i + 6 >= len(cal): return np.nan
    r = RO.iloc[i + 2: i + 7][sids]                          # r_{t+k} = ln(O_{t+k+1}/O_{t+k})，k=1..5 → 日曆列 t+2 … t+6
    if r.isna().any().any(): return np.nan
    return float(np.sqrt(252 / 5 * ((r.values @ q) ** 2).sum()))
def window_score(L, shrink=True):
    rows = []
    for p in weekly_plan("ml", 0.0, None, L, {2022, 2023, 2024}, shrink):
        real = realized_fixed_weight(p["date"], p["sids"], p["q"])
        if np.isnan(real): continue
        x = (real / p["sigma_p"]) ** 2
        rows.append(dict(date=p["date"], L=L, shrink=shrink, sigma_p=p["sigma_p"], real=real, qlike=x - np.log(x) - 1, lam=p["lam"], nobs=p["nobs"], est_fail=p["est_fail"]))
    return pd.DataFrame(rows)
WS = pd.concat([window_score(20), window_score(60), window_score(120), window_score(60, shrink=False), window_score(20, shrink=False)])
WS["year"] = pd.DatetimeIndex(WS.date).year
wtab = WS.groupby(["L", "shrink", "year"]).qlike.mean().unstack()
wtab["三年平均"] = wtab[[2022, 2023, 2024]].mean(axis=1)
wtab = wtab.join(WS.groupby(["L", "shrink"]).agg(週數=("qlike", "size"), λ中位數=("lam", "median"), 預測波動中位數=("sigma_p", "median"), 實現波動中位數=("real", "median"), 估計失敗週=("est_fail", "sum")))
print("V7 5.5 窗口比較（策略 B）：\n", wtab.round(4).to_string())
q20, q60 = wtab.loc[(20, True), "三年平均"], wtab.loc[(60, True), "三年平均"]
L_SEL = 60 if abs(q20 - q60) <= 0.01 * min(q20, q60) else (20 if q20 < q60 else 60)  # 較低者勝；差距 1% 內選 60
r0050 = {t: (O["0050"].iloc[cal.get_loc(t) + 6] / O["0050"].iloc[cal.get_loc(t) + 1] - 1) if cal.get_loc(t) + 6 < len(cal) else np.nan for t in DECS}
WS["r0050"] = WS.date.map(r0050)
cut = WS[WS.shrink & WS.L.isin([20, 60])].drop_duplicates("date").r0050.quantile(0.10)
crash_tab = WS[WS.shrink & WS.L.isin([20, 60]) & (WS.r0050 <= cut)].groupby("L").agg(急跌週數=("qlike", "size"), QLIKE=("qlike", "mean"), 實現波動=("real", "mean"))
print("急跌週（0050 持有期報酬最低 10%）：\n", crash_tab.round(4).to_string()); print("選定窗口：", L_SEL)
wtab.round(4).to_csv(OUT / "window_comparison.csv"); crash_tab.round(4).to_csv(OUT / "window_crash_weeks.csv")

# ---------- 3. 目標波動率（V7 7.5）：策略 A，2017–2020，中位數 ----------
PA = weekly_plan("hv", 0.0, None, L_SEL, {2017, 2018, 2019, 2020})
TA = pd.DataFrame([dict(date=p["date"], n=len(p["sids"]), sigma_p=p["sigma_p"], exception=p["exception"]) for p in PA])
normal = TA[~TA.exception]
SIGMA_TARGET = float(normal.sigma_p.median())                # 規則 c：例外週不納入
SENS_T = {"p25": float(normal.sigma_p.quantile(.25)), "p75": float(normal.sigma_p.quantile(.75))}
ew = panel[panel.sid.isin(CFG["universe"]["stocks"]) & panel.valid & (panel.date >= "2017-01-01") & (panel.date <= "2020-12-31")].sort_values(["sid", "date"])
ew["r"] = ew.groupby("sid").aclose.pct_change()
SENS_T["ew_hist"] = float(ew.groupby("date").r.mean().std() * np.sqrt(252))  # 15 檔等權重組合的歷史年化波動（敏感度）
print(f"目標波動率 σ_target = {SIGMA_TARGET:.4f}（正常週 {len(normal)}，例外週 {int(TA.exception.sum())} 不納入）；敏感度：", {k: round(v, 4) for k, v in SENS_T.items()})
TA.to_csv(OUT / "target_strategyA_2017_2020.csv", index=False)

# ---------- 4. 回測引擎（V7 8.4） ----------
ADV = panel.assign(amt=panel.amount.where(panel.valid)).sort_values(["sid", "date"]).groupby("sid").amt.transform(lambda x: x.rolling(20, min_periods=15).mean())  # 20 日平均成交金額（元）
ADV = pd.Series(ADV.values, index=pd.MultiIndex.from_frame(panel.loc[ADV.index, ["date", "sid"]])).to_dict()
def backtest(plan, start, end, slip=COST["slip"], comm=COST["comm"]):
    return execute_plan(plan, cal, O, Cl, ADV, start, end, capital=CAPITAL,
                        deposit_rate=deposit_rate, slip=slip, comm=comm, tax=COST["tax"])


def metrics(N, T, Wk):                                       # 定義：V7 8.3 的指標
    r = N.nav.pct_change().dropna(); d = N.index.to_series().diff().dt.days.reindex(r.index)
    rf = pd.Series([deposit_rate(x) for x in r.index], index=r.index) * d / 365  # 無風險利率＝同一定存利率
    ex = r - rf; yrs = len(r) / 252; avg = N.nav.mean()
    sr = ex.mean() / ex.std() * np.sqrt(252)
    dd = (N.nav / N.nav.cummax() - 1).min()
    ratio = (T.value.abs() / T.adv20).dropna() if len(T) else pd.Series(dtype=float)
    return {"Sharpe": sr, "年化報酬": (N.nav.iloc[-1] / N.nav.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "年化波動": r.std() * np.sqrt(252), "最大回撤": dd, "平均股票曝險": N.expo.mean(),
            "年化換手率": Wk.turnover.sum() / avg / yrs, "年化成本占資產": Wk.cost.sum() / avg / yrs,
            "風險預算偏差（週最大值平均）": Wk[~Wk.exception].rb_dev.mean(), "例外週": int(Wk.exception.sum()),
            "成交額/均額 中位數": ratio.median() if len(ratio) else np.nan, "成交額/均額 P95": ratio.quantile(.95) if len(ratio) else np.nan,
            "成交額/均額 最大": ratio.max() if len(ratio) else np.nan, "天數": len(r), "_ex": ex}

def yearly_sharpe(N):                                        # 每個日曆年的扣成本 Sharpe
    r = N.nav.pct_change().dropna(); d = N.index.to_series().diff().dt.days.reindex(r.index)
    ex = r - pd.Series([deposit_rate(x) for x in r.index], index=r.index) * d / 365
    return ex.groupby(ex.index.year).apply(lambda e: e.mean() / e.std() * np.sqrt(252))

VAL = ("2022-01-01", "2024-12-31"); POST = ("2025-01-01", "2026-10-30")  # 驗證期、事後評估期（研究終點 2026-11-01 前最後一個交易日）
def run(src, alpha, target=None, L=None, period=VAL, shrink=True, **kw):
    yrs = set(range(pd.Timestamp(period[0]).year - 1, pd.Timestamp(period[1]).year + 1))
    plan = weekly_plan(src, alpha, target or SIGMA_TARGET, L or L_SEL, yrs, shrink)
    return backtest(plan, *period, **kw)

# ---------- 5. 2021 健全性檢查（V7 7.5 第 4 步；不據以調整） ----------
san = []
for nm, src in (("A", "hv"), ("B", "ml")):
    pl = weekly_plan(src, 0.0, SIGMA_TARGET, L_SEL, {2021})
    a = np.array([p["a"] for p in pl]); san.append({"策略": nm, "週數": len(a), "a<1 週比例": (a < 1).mean(), "平均股票曝險": a.mean()})
print("2021 健全性檢查：\n", pd.DataFrame(san).round(3).to_string(index=False)); pd.DataFrame(san).to_csv(OUT / "sanity_2021.csv", index=False)

# ---------- 6. α 選擇（V7 4.4）：策略 D，2022–2024 ----------
AL = []
for a in (0.0, 0.5, 1.0, 2.0):
    N, T, Wk = run("ml", a); ys = yearly_sharpe(N); m = metrics(N, T, Wk)
    AL.append({"alpha": a, **{f"Sharpe {y}": ys.get(y, np.nan) for y in (2022, 2023, 2024)}, "三年平均 Sharpe": ys.loc[[2022, 2023, 2024]].mean(), "整段 Sharpe": m["Sharpe"], "_ex": m["_ex"]})
ALT = pd.DataFrame(AL); best = ALT["三年平均 Sharpe"].max()
ALPHA = float(ALT[ALT["三年平均 Sharpe"] >= best - 0.05].alpha.min())  # 差距 < 0.05 選較小的 α
print("V7 4.4 α 選擇（策略 D）：\n", ALT.drop(columns="_ex").round(4).to_string(index=False)); print("選定 α：", ALPHA)
ALT.drop(columns="_ex").round(4).to_csv(OUT / "alpha_selection.csv", index=False)

# ---------- 7. 凍結後回測 A–D：驗證期與事後評估期 ----------
res, curves, excess = [], {}, {}
for per_name, per in (("2022–2024 驗證期", VAL), ("2025–2026 事後評估", POST)):
    for s, (src, ak) in STRAT.items():
        a = 0.0 if ak == "fixed0" else ALPHA
        N, T, Wk = run(src, a, period=per); m = metrics(N, T, Wk); excess[(per_name, s)] = m.pop("_ex")
        res.append({"期間": per_name, "策略": s, "α": a, **m}); curves[(per_name, s)] = N
        Wk.to_csv(OUT / f"weekly_{s}_{per[0][:4]}.csv", index=False); T.to_csv(OUT / f"trades_{s}_{per[0][:4]}.csv", index=False)
RES = pd.DataFrame(res); RES.round(4).to_csv(OUT / "backtest_A_to_D.csv", index=False)
pd.concat({f"{k[0]}|{k[1]}": v.nav for k, v in curves.items()}, axis=1).to_csv(OUT / "nav_curves.csv")
print(RES.round(4).to_string(index=False))

# ---------- 8. 被動參考（V7 未列為比較組；只報告）：0050 買進持有 ----------
ref = []
for per_name, per in (("2022–2024 驗證期", VAL), ("2025–2026 事後評估", POST)):
    c = Cl["0050"][(cal >= per[0]) & (cal <= per[1])].dropna(); r = c.pct_change().dropna(); d = c.index.to_series().diff().dt.days.reindex(r.index)
    ex = r - pd.Series([deposit_rate(x) for x in r.index], index=r.index) * d / 365
    ref.append({"期間": per_name, "Sharpe": ex.mean() / ex.std() * np.sqrt(252), "年化報酬": (c.iloc[-1] / c.iloc[0]) ** (252 / len(r)) - 1, "年化波動": r.std() * np.sqrt(252), "最大回撤": (c / c.cummax() - 1).min()})
pd.DataFrame(ref).round(4).to_csv(OUT / "reference_0050.csv", index=False); print("0050：\n", pd.DataFrame(ref).round(4).to_string(index=False))

# ---------- 9. Deflated Sharpe（V7 2.6：試驗次數 6＝B×2 窗口＋D×4 α；對最終策略 D 計算） ----------
def dsr(ex, trial_srs):                                      # Bailey & López de Prado (2014)
    n_trials = len(trial_srs)
    sr = ex.mean() / ex.std(); T_ = len(ex); g3 = skew(ex); g4 = kurtosis(ex, fisher=False)
    v = np.var(trial_srs, ddof=1); emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1 / n_trials) + emc * norm.ppf(1 - 1 / (n_trials * np.e)))
    return float(norm.cdf((sr - sr0) * np.sqrt(T_ - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))), float(sr0 * np.sqrt(252))
trials, post_windows = [], []
for L in (20, 60):
    N, T, Wk = run("ml", 0.0, L=L)
    m = metrics(N, T, Wk)
    e = m.pop("_ex")
    trials.append(dict(config=f"B_window_{L}", strategy="B", window=L, alpha=0.0,
                       qlike=wtab.loc[(L, True), "三年平均"], sharpe=m["Sharpe"],
                       daily_sharpe=e.mean() / e.std(), selected=L == L_SEL,
                       annual_turnover=m["年化換手率"], max_drawdown=m["最大回撤"]))
    N, T, Wk = run("ml", 0.0, L=L, period=POST)
    m = metrics(N, T, Wk)
    m.pop("_ex")
    post_windows.append({"窗口": L, "選定": L == L_SEL, "策略": "B", **m})
pd.DataFrame(post_windows).round(4).to_csv(OUT / "window_posthoc.csv", index=False, encoding="utf-8-sig")
for _, row in ALT.iterrows():
    e = row["_ex"]
    a = row.alpha
    trials.append(dict(config=f"D_alpha_{a:g}", strategy="D", window=L_SEL, alpha=a,
                       sharpe=row["整段 Sharpe"], mean_yearly_sharpe=row["三年平均 Sharpe"],
                       daily_sharpe=e.mean() / e.std(), selected=a == ALPHA))
ledger = pd.DataFrame(trials).assign(trial_family="model2_v7", fset=VER, eval_window="2022–2024",
                                    shrinkage="constant_correlation", counts_for_dsr=True)
ledger.to_csv(OUT / "model2_trials.csv", index=False, encoding="utf-8-sig")
log_runs(ledger.rename(columns={"qlike": "QLIKE", "sharpe": "Sharpe"}))
trial = ledger.daily_sharpe.to_numpy()
DS = []
for per_name in ("2022–2024 驗證期", "2025–2026 事後評估"):
    for s in ("B", "D"):
        p_, sr0 = dsr(excess[(per_name, s)], trial); DS.append({"期間": per_name, "策略": s, "Deflated Sharpe 機率": p_, "門檻 Sharpe（年化）": sr0})
print("Deflated Sharpe：\n", pd.DataFrame(DS).round(4).to_string(index=False)); pd.DataFrame(DS).round(4).to_csv(OUT / "deflated_sharpe.csv", index=False)

# ---------- 10. 敏感度與穩健性（只報告，不參與選擇） ----------
sens = []
for lab, kw in [("目標＝A 第 25 百分位", dict(target=SENS_T["p25"])), ("目標＝A 第 75 百分位", dict(target=SENS_T["p75"])),
                ("目標＝等權重歷史波動", dict(target=SENS_T["ew_hist"])), ("相關 120 日", dict(L=120)), ("不收縮", dict(shrink=False)),
                ("滑價 0 bps", dict(slip=0.0)), ("滑價 10 bps", dict(slip=0.001)), ("滑價 20 bps", dict(slip=0.002)), ("手續費 6 折", dict(comm=COST["comm"] * 0.6))]:
    for s in ("A", "B", "C", "D"):
        src, ak = STRAT[s]; a = 0.0 if ak == "fixed0" else ALPHA
        for per_name, per in (("2022–2024 驗證期", VAL), ("2025–2026 事後評估", POST)):
            N, T, Wk = run(src, a, period=per, **kw); m = metrics(N, T, Wk); m.pop("_ex")
            sens.append({"設定": lab, "期間": per_name, "策略": s, "Sharpe": m["Sharpe"], "年化報酬": m["年化報酬"], "年化波動": m["年化波動"], "最大回撤": m["最大回撤"], "平均股票曝險": m["平均股票曝險"]})
pd.DataFrame(sens).round(4).to_csv(OUT / "sensitivity.csv", index=False)
json.dump({"window": L_SEL, "sigma_target": SIGMA_TARGET, "alpha": ALPHA, "sens_targets": SENS_T, "primary_model": SEL}, open(OUT / "frozen_rules.json", "w"), ensure_ascii=False, indent=2)
print("凍結規則：", json.load(open(OUT / "frozen_rules.json")))
