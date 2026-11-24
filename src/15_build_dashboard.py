# 15_build_dashboard.py — 把 V7 的最後結果做成「互動式儀表板」（一個 HTML 檔，資料直接內嵌，雙擊就能用瀏覽器打開）
#   讀：outputs/v7/tables/ 底下的結果表（05、07、10、12、14 產生）
#   寫：outputs/v7/dashboard/dashboard.html（本機用，完整 HTML）
#       outputs/v7/dashboard/dashboard_body.html（只有內容，給 claude.ai 發布用；平台會自己包上 <html>）
#   版面與互動程式在 src/dashboard_template.html；這支程式只負責「整理資料、塞進樣板」
import json, numpy as np, pandas as pd                       # 載入：JSON、數值、表格
from common import PROC, VTAB, ROOT, nw_mean_ci               # 載入：路徑與 HAC 平均值工具
M2, OUT = VTAB / "model2", ROOT / "outputs" / "v7" / "dashboard"  # Model 2 結果資料夾、輸出資料夾
OUT.mkdir(parents=True, exist_ok=True)                       # 不存在就建立
PER = {"val": ("2022–2024 驗證期", "2022"), "post": ("2025–2026 事後評估", "2025")}  # 兩個期間：顯示名稱、檔名年份
r4 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)  # 小工具：四捨五入到 4 位，空值變 null

# ---------- 1. 回測：淨值、曝險、KPI ----------
nav = pd.read_csv(M2 / "nav_curves.csv", index_col=0, parse_dates=True)  # 每日淨值（四組策略、兩個期間）
panel = pd.read_parquet(PROC / "panel.parquet")              # 日資料（取 0050 當參考）
c50 = panel[(panel.sid == "0050") & panel.valid].set_index("date").aclose  # 0050 還原收盤價
bt = pd.read_csv(M2 / "backtest_A_to_D.csv")                 # 回測總表
ref = pd.read_csv(M2 / "reference_0050.csv").set_index("期間")  # 0050 指標
D = {"periods": {}}                                          # 要塞進網頁的資料
for k, (name, yr) in PER.items():                            # 逐一處理兩個期間
    a, b = nav[f"{name}|A"].dropna(), nav[f"{name}|B"].dropna()  # A、B 的淨值
    r = c50.reindex(a.index).ffill()                         # 同日期的 0050
    wa = pd.read_csv(M2 / f"weekly_A_{yr}.csv", parse_dates=["date"])  # A 每週配置
    wb = pd.read_csv(M2 / f"weekly_B_{yr}.csv", parse_dates=["date"])  # B 每週配置
    d = (b.pct_change() - a.pct_change()).dropna()           # B − A 每日報酬差
    m, lo, hi, se = nw_mean_ci(d.values)                     # HAC 平均與標準誤
    kp = {}                                                  # 這個期間的 KPI
    for s in ("A", "B"):                                     # A、B 兩組
        row = bt[(bt.期間 == name) & (bt.策略 == s)].iloc[0]  # 該組那一列
        kp[s] = {"sharpe": r4(row.Sharpe), "ret": r4(row.年化報酬), "vol": r4(row.年化波動), "mdd": r4(row.最大回撤),
                 "expo": r4(row.平均股票曝險), "turn": r4(row.年化換手率), "cost": r4(row.年化成本占資產)}
    kp["ref"] = {"sharpe": r4(ref.loc[name, "Sharpe"]), "ret": r4(ref.loc[name, "年化報酬"]), "vol": r4(ref.loc[name, "年化波動"]), "mdd": r4(ref.loc[name, "最大回撤"])}
    D["periods"][k] = {"name": name,
        "nav": {"x": [t.strftime("%Y-%m-%d") for t in a.index], "A": [r4(v) for v in a / a.iloc[0]],
                "B": [r4(v) for v in b / b.iloc[0]], "ref": [r4(v) for v in r / r.iloc[0]]},  # 期初 = 1
        "expo": {"x": [t.strftime("%Y-%m-%d") for t in wb.date], "A": [r4(v) for v in wa.a], "B": [r4(v) for v in wb.a]},  # 每週股票曝險
        "kpi": kp, "diff": {"ann": r4(m * 252), "t": r4(m / se)}}  # B − A 年化差與 HAC t 值
ds = pd.read_csv(M2 / "deflated_sharpe.csv")                 # Deflated Sharpe
for k, (name, _) in PER.items():
    D["periods"][k]["dsr"] = r4(ds[(ds.期間 == name) & (ds.策略 == "B")]["Deflated Sharpe 機率"].iloc[0])

# ---------- 2. 敏感度 ----------
sen = pd.read_csv(M2 / "sensitivity.csv")                    # 敏感度結果
rows = []                                                    # 每個設定 × 期間一列
for k, (name, _) in PER.items():
    rows.append({"p": k, "set": "主設定", "A": D["periods"][k]["kpi"]["A"]["sharpe"], "B": D["periods"][k]["kpi"]["B"]["sharpe"]})
    for s in dict.fromkeys(sen.設定):                         # 依原順序逐一設定
        x = sen[(sen.設定 == s) & (sen.期間 == name)].set_index("策略").Sharpe
        rows.append({"p": k, "set": s, "A": r4(x["A"]), "B": r4(x["B"])})
D["sens"] = rows

# ---------- 3. Model 1：各年 QLIKE、預測區間涵蓋率、每週預測 ----------
t = pd.read_csv(VTAB / "model1_table_2_8.csv").set_index("config")  # 驗證期三年
ph = pd.read_csv(VTAB / "model1_posthoc_2025_2026.csv").set_index("config")  # 事後評估
names = {"lasso_alpha_frac0.1": "Lasso（主模型）", "har": "HAR", "ewma": "EWMA", "garch": "GARCH", "hv": "HV20"}
D["m1"] = [{"model": v, "y2022": r4(t.loc[k, "fold1"]), "y2023": r4(t.loc[k, "fold2"]), "y2024": r4(t.loc[k, "fold3"]),
            "avg": r4(t.loc[k, "QLIKE"]), "post": r4(ph.loc[k, "QLIKE"])} for k, v in names.items()]
cv = pd.read_csv(VTAB / "model1_conformal.csv")              # 預測區間涵蓋率
D["coverage"] = [{"p": p, "cov": r4(c)} for p, c in zip(["2022", "2023", "2024", "2025–26"], cv.coverage)]
pr = pd.read_csv(VTAB / "volatility_predictions_weekly.csv", dtype={"股票": str})  # 14 產生的每週預測
D["preds"] = {}                                              # 每檔一組陣列
for sid, g in pr.groupby("股票"):
    g = g.sort_values("決策日")
    D["preds"][sid] = {"x": list(g.決策日), "ml": [r4(v) for v in g["預測波動_Lasso"]], "lo": [r4(v) for v in g["區間下限_90%"]],
                       "hi": [r4(v) for v in g["區間上限_90%"]], "hv": [r4(v) for v in g.HV20], "y": [r4(v) for v in g["實際波動_未來5日"]],
                       "tr": [bool(v) for v in g.可交易]}

# ---------- 4. Model 2 的三個選擇 ----------
fr = json.load(open(M2 / "frozen_rules.json"))               # 凍結規則
wc = pd.read_csv(M2 / "window_comparison.csv")               # 窗口比較
D["window"] = [{"label": f"{int(r.L)} 日" + ("" if r.shrink else "（不收縮）"), "q": r4(r["三年平均"]), "sel": bool(r.L == fr["window"] and r.shrink)} for _, r in wc.sort_values("三年平均").iterrows()]
al = pd.read_csv(M2 / "alpha_selection.csv")                 # α 選擇
D["alpha"] = [{"a": r4(r.alpha), "s": r4(r["三年平均 Sharpe"]), "sel": bool(r.alpha == fr["alpha"])} for _, r in al.iterrows()]
D["rules"] = {"window": fr["window"], "target": r4(fr["sigma_target"]), "alpha": fr["alpha"], "p25": r4(fr["sens_targets"]["p25"]), "p75": r4(fr["sens_targets"]["p75"])}
post_windows = M2 / "window_posthoc.csv"
if post_windows.exists():
    D["postWindows"] = [
        [int(row["窗口"]), bool(row["選定"]), r4(row.Sharpe), r4(row["年化報酬"]),
         r4(row["年化波動"]), r4(row["最大回撤"])]
        for _, row in pd.read_csv(post_windows).iterrows()
    ]
screen_summary = VTAB / "feature_selection" / "v7" / "summary.json"
if screen_summary.exists():
    D["featureScreen"] = json.loads(screen_summary.read_text(encoding="utf-8"))

# ---------- 5. 驗證：PIT、相關性 ----------
ps = pd.read_csv(VTAB / "pit" / "pit_summary.csv")           # PIT 結果
D["pit"] = []
for c, g in ps.groupby("截止日"):
    n = int(g[g.欄位 == "（列）"]["完整資料列數"].iloc[0])     # 比較列數
    gar = g[g.欄位 == "garch_oo5"]["不一致筆數"].iloc[0]      # GARCH 不一致
    other = g[~g.欄位.isin(["（列）", "garch_oo5"])]["不一致筆數"].fillna(0).sum()  # 其他欄位不一致
    D["pit"].append({"cut": c, "n": n, "other": int(other), "garch": int(gar)})
rules = pd.read_csv(VTAB / "correlation" / "feature_rules_2016_2020.csv")  # 特徵規則
top = pd.read_csv(VTAB / "correlation" / "feature_corr_top_pairs.csv").iloc[0]  # 相關最高的一對
cm = pd.read_csv(VTAB / "correlation" / "corr_matrix_checks_summary.csv")  # 相關矩陣檢查
ws = pd.read_csv(VTAB / "correlation" / "weight_stability.csv")  # 權重穩定性
g = rules[rules.特徵 == "garch_oo5"].iloc[0]
D["corr"] = {"maxPair": f"{top.level_0}–{top.level_1}", "maxRho": r4(top.rho), "garchIC": r4(g["平均 IC"]), "garchMaxRho": r4(g["與其他特徵最大 |ρ|"]),
             "weeks": int(cm["週數"].iloc[0]), "condShrink": r4(cm[(cm.L == 20) & (cm.收縮)]["條件數最大"].iloc[0]),
             "condRaw": r4(cm[(cm.L == 20) & (~cm.收縮)]["條件數最大"].iloc[0]),
             "stab": {f"{r.期間}|{r.策略}": r4(r["mean"]) for _, r in ws[(ws.L == 20) & ws.收縮].iterrows()}}
D["asof"] = str(pr.決策日.max())                              # 資料截止日

# ---------- 6. 塞進樣板、寫檔 ----------
body = (ROOT / "src" / "dashboard_template.html").read_text(encoding="utf-8")  # 讀樣板
body = body.replace("/*__DATA__*/null", json.dumps(D, ensure_ascii=False, separators=(",", ":")))  # 資料放進去
(OUT / "dashboard_body.html").write_text(body, encoding="utf-8")  # 發布用（只有內容）
full = ("<!doctype html>\n<html lang=\"zh-Hant\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")
(OUT / "dashboard.html").write_text(full, encoding="utf-8")  # 本機用（雙擊開啟）
print("已寫出", OUT / "dashboard.html", f"{len(full) / 1024:.0f} KB")
