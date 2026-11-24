# 13_fixed_split_check.py — 用本專案的程式，照小組「V7 固定切分」簡報的設計重跑一次，比對兩邊數字
#   固定切分：Train 2016-11～2021-12；Validation 2022–2023（選窗口、α）；Test 2024-01～2026-10（Model 1 用 Train＋Validation 重訓一次）
#   做法：重用 07_model2_v7.py 的程式（資料、Uₜ、相關矩陣、配置、回測引擎），只換掉三件事：
#     ① Model 1 預測：Validation 用只在 Train 訓練的模型；Test 用訓練到 2023 年底（標籤也在 2023 年內結束）的模型，各訓練一次
#     ② 窗口比較只用 2022–2023　③ 結果寫到 outputs/v7/tables/fixed_split（不覆蓋 V7 正式結果）
#   與小組的差異（無法得知、照本專案設定）：Lasso 懲罰值（小組 Lasso_03／Lasso_05；本專案用 05 選出的 α_frac 0.1）、目標波動率（本專案 20.9%）
import json, pathlib, numpy as np, pandas as pd              # 載入：JSON、路徑、數值、表格
HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "07_model2_v7.py").read_text(encoding="utf-8")  # 讀入 07 的程式碼
src = src[:src.index("# ---------- 5.")]                    # 只要前半段（資料、預測、窗口、目標、回測引擎）
def swap(a, b):                                              # 定義：替換一段程式，找不到就報錯（避免默默沒換到）
    global src
    assert a in src, a[:50]; src = src.replace(a, b)
swap('OUT = VTAB / "model2"', 'OUT = VTAB / "fixed_split"')  # ③ 另存資料夾
swap('''for Y, tr_end, t0, t1 in [(2021, "2020-12-31", "2021-01-01", "2021-12-31"), (2022, "2021-12-31", "2022-01-01", "2022-12-31"),
                          (2023, "2022-12-31", "2023-01-01", "2023-12-31"), (2024, "2023-12-31", "2024-01-01", "2024-12-31"),
                          (2025, "2024-12-31", "2025-01-01", "2026-11-01")]:''',
     '''for Y, tr_end, t0, t1 in [(2021, "2020-12-31", "2021-01-01", "2021-12-31"), (2022, "2021-12-31", "2022-01-01", "2023-12-31"),
                          (2024, "2023-12-31", "2024-01-01", "2026-11-01")]:''')  # ① 固定切分：Validation 一個模型、Test 一個模型
swap('ML.to_parquet(PROC / "model2_sigma_ml_v7.parquet", index=False)', 'pass')  # 不覆蓋 V7 正式的預測檔
swap('for p in weekly_plan("ml", 0.0, None, L, {2022, 2023, 2024}, shrink):', 'for p in weekly_plan("ml", 0.0, None, L, {2022, 2023}, shrink):')  # ② 只用 2022–2023
swap('WS = pd.concat([window_score(20), window_score(60), window_score(120), window_score(60, shrink=False), window_score(20, shrink=False)])',
     'WS = pd.concat([window_score(20), window_score(60)])')
swap('wtab["三年平均"] = wtab[[2022, 2023, 2024]].mean(axis=1)', 'wtab["三年平均"] = wtab[[2022, 2023]].mean(axis=1)')
exec(compile(src, "07_model2_v7(fixed split)", "exec"))      # 執行修改後的前半段

TEST = ("2024-01-01", "2026-10-30")                          # Test 期間
rows, navs = [], {}
for s, srcv in (("A：HV20", "hv"), ("B：Model 1", "ml")):
    N, T, Wk = run(srcv, 0.0, L=L_SEL, period=TEST); m = metrics(N, T, Wk); ex = m.pop("_ex"); navs[s] = N.nav
    rows.append({"策略": s, "窗口": L_SEL, **m, "期末 NAV": N.nav.iloc[-1], "週數": len(Wk)})
R = pd.DataFrame(rows); R.round(4).to_csv(OUT / "test_2024_2026.csv", index=False, encoding="utf-8-sig")
d = (navs["B：Model 1"].pct_change() - navs["A：HV20"].pct_change()).dropna()  # B − A 日報酬差（HAC）
from common import nw_mean_ci; mean, lo, hi, se = nw_mean_ci(d.values)  # HAC 平均與 95% 區間
print(f"B − A 年化報酬差 {mean * 252:.2%}，95% 區間 [{lo * 252:.2%}, {hi * 252:.2%}]")
print(R[["策略", "窗口", "Sharpe", "年化報酬", "年化波動", "最大回撤", "平均股票曝險", "年化換手率", "期末 NAV", "週數"]].round(4).to_string(index=False))
json.dump({"window": L_SEL, "sigma_target": SIGMA_TARGET}, open(OUT / "fixed_split_rules.json", "w"), ensure_ascii=False)

# ---------- Model 1 的 Test 評估（固定切分：訓練到 2023 年底、標籤也在 2023 內結束，評估 2024-01～2026-10） ----------
from models import make_model as _mk                          # 載入：建立模型
from common import nw_mean_ci as _nw
pnl = pd.read_parquet(PROC / "panel.parquet").sort_values(["sid", "date"])
rr = np.log(pnl.aclose / pnl.groupby("sid").aclose.shift())  # EWMA 用收盤報酬（與 05 相同）
pnl["ewma"] = np.sqrt(252 * (rr ** 2).groupby(pnl.sid).transform(lambda s: s.ewm(alpha=0.06, adjust=False, min_periods=20).mean()))
D = df.merge(pnl[["date", "sid", "ewma"]], on=["date", "sid"], how="left")
trm, _ = split(D, "2023-12-31", "2024-01-01", "2024-12-31", dec_col="is_decision_wk"); trm = trm[trm.garch_oo5.notna()]  # 訓練集（purge 到 2024 前）
tem = D[D.is_decision_wk.fillna(False).astype(bool) & D.y.notna() & (D.date >= "2024-01-01") & (D.label_end <= pd.Timestamp(CFG["data"]["study_end"]))
        & D.tradable.fillna(False).astype(bool) & D.garch_oo5.notna()].copy()  # Test：可交易、有預測、標籤在研究終點前結束
P = {"Lasso（α_frac 0.1）": np.maximum(_mk(kind, prm).fit(trm[COLS], trm.y).predict(tem[COLS]), FLOOR), "HV20": tem.rv20.values,
     "EWMA": tem.ewma.values, "GARCH": tem.garch_oo5.values,
     "HAR": np.maximum(_mk("ridge", {"alpha": 1e-4}).fit(trm[["rv1", "rv5", "rv20"]], trm.y).predict(tem[["rv1", "rv5", "rv20"]]), FLOOR)}
ql = lambda p: (lambda x: x - np.log(x) - 1)((tem.y.values / p) ** 2)  # 每筆 QLIKE
M1 = pd.DataFrame([{"模型": k, "QLIKE": ql(v).mean(), "筆數": len(tem)} for k, v in P.items()])
wk = pd.DataFrame({"date": tem.date.values, "d": ql(P["Lasso（α_frac 0.1）"]) - ql(P["HV20"])}).groupby("date").d.mean()  # 每週配對差
m_, lo_, hi_, _ = _nw(wk.values)
M1.round(4).to_csv(OUT / "model1_test_2024_2026.csv", index=False, encoding="utf-8-sig")
print(M1.round(4).to_string(index=False)); print(f"Lasso − HV20 QLIKE 配對差 {m_:.4f}，HAC 95% [{lo_:.4f}, {hi_:.4f}]")
