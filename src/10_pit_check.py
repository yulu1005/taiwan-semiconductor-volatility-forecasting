# 10_pit_check.py — 特徵的時點正確性（Point-in-Time, PIT）驗證
#
# 想法（截斷重算法）：如果某個特徵在日期 t 只用到 t 以前的資料，那麼「把 t 之後的原始資料全部刪掉、從頭重跑」，
#   t 以前的特徵值必須和用完整資料算出來的一模一樣。只要有任何一個值不同，就代表這個特徵偷看了未來。
#   比喻：考試前把明天的報紙全部收走，如果你今天的答案還是一樣，就證明你沒有偷看明天的報紙。
#
# 範圍：原始資料（行情、除權息、減資、手動分割）→ 02 還原權值 → 03 樣本與合格標記 → 03b／features_30 特徵 →
#       universe 可交易名單與每週決策日 → 03d 開盤到開盤 GARCH。也就是 Model 1、Model 2 用到的所有輸入。
# 截止日：4 個星期五（2019-06-28、2022-09-30、2024-12-27、2026-03-27），涵蓋特徵篩選期、驗證期、封存前、事後評估期。
# 比較：截止日（含）以前的每一列。數值容許誤差：|差| ≤ 1e-9 ＋ 1e-6 ×|值|（截止日之後的除權息會讓整段還原價格乘上同一個倍數，
#       數學上比率不變，但電腦的小數會差到 1e-16 左右，這不是偷看未來）。
# GARCH 另外檢查：①輸入的開盤到開盤報酬是否相同（應只差浮點誤差）②每次估計的參數差多少（最佳化對極小輸入差很敏感）。
# 不比較標籤 y、label_end（標籤本來就是未來資料，只當答案用，不當輸入）。
import shutil, subprocess, sys, json, numpy as np, pandas as pd  # 載入：複製檔案、執行子程式、系統、JSON、數值、表格
from concurrent.futures import ThreadPoolExecutor            # 載入：同時跑幾個截止日
from common import ROOT, CFG, VTAB                           # 載入：專案路徑與設定
from garch_open_returns import opening_returns, read_actions

CUTOFFS = sys.argv[1:] or ["2019-06-28", "2022-09-30", "2024-12-27", "2026-03-27"]  # 截止日（都是交易日的星期五；可在指令後面指定）
STEPS = ["02_build_dataset.py", "03_features_labels.py", "03b_features_v2.py", "features_30.py", "universe.py", "03d_garch_oo.py"]  # 依序重跑的步驟
FEATS = list(CFG["features"]["v7"])                          # V7 的 18 項特徵
EXTRA = ["eligible", "tradable", "is_decision_wk", "adv250", "adv20", "garch_oo5", "rv20"]  # 也要檢查的名單與配置輸入
WORK = ROOT / "outputs" / "v7" / "pit_work"                  # 暫存工作區（每個截止日一份完整副本）
OUT = VTAB / "pit"; OUT.mkdir(parents=True, exist_ok=True)   # 結果資料夾

def make_copy(cut):                                          # 定義：建立「只看得到截止日以前資料」的專案副本
    d = WORK / cut
    if d.exists(): shutil.rmtree(d)                          # 舊的先刪掉
    shutil.copytree(ROOT / "src", d / "src", ignore=shutil.ignore_patterns("__pycache__"))  # 複製程式
    shutil.copy(ROOT / "setup.yaml", d / "setup.yaml")      # 複製設定
    raw = d / "data" / "raw"; raw.mkdir(parents=True)        # 原始資料夾
    shutil.copy(ROOT / "data" / "raw" / "manifest.json", raw / "manifest.json")
    c = pd.Timestamp(cut)
    L = pd.read_csv(ROOT / "data" / "raw" / "listing_dates.csv", dtype={"sid": str}, parse_dates=["listing_date"])
    L[L.listing_date <= c].to_csv(raw / "listing_dates.csv", index=False)  # 截止日時還沒上市的股票，當時不可能知道 → 也刪掉
    not_listed = L.loc[L.listing_date > c, "sid"]
    for f in ("price", "div", "capred"):                     # 行情與公司行動：刪掉截止日之後的列
        x = pd.read_parquet(ROOT / "data" / "raw" / f"{f}.parquet")
        # Removing the listing date alone would admit pre-listing quotes.
        x[(pd.to_datetime(x.date) <= c) & ~x.stock_id.isin(not_listed)].to_parquet(raw / f"{f}.parquet", index=False)
    m = pd.read_csv(ROOT / "data" / "raw" / "manual_splits.csv", dtype={"sid": str}, parse_dates=["date"])
    m[m.date <= c].to_csv(raw / "manual_splits.csv", index=False)  # 手動分割事件也一樣截斷
    return d

def run_pipeline(cut):                                       # 定義：在副本裡從原始資料重跑到特徵
    d = make_copy(cut); log = []
    for s in STEPS:
        r = subprocess.run([sys.executable, s], cwd=d / "src", capture_output=True, text=True)  # 在副本的 src 裡執行
        log.append(f"== {s}: exit {r.returncode}\n{r.stderr[-1500:]}")
        if r.returncode != 0: break                          # 某一步失敗就停
    (OUT / f"log_{cut}.txt").write_text("\n".join(log), encoding="utf-8")
    return d, r.returncode

def load_from(root):                                         # 定義：用副本的 models.load() 讀出合格樣本（與正式流程同一支程式）
    code = ("import sys; sys.path.insert(0, 'src'); from models import load; import pandas as pd; "
            "d = load(); p = pd.read_parquet('data/processed/samples.parquet', columns=['date','sid','eligible']); "
            "d.to_parquet('loaded.parquet', index=False); p.to_parquet('eligible.parquet', index=False)")
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)
    return pd.read_parquet(root / "loaded.parquet"), pd.read_parquet(root / "eligible.parquet")

def compare(full, trunc, cut, cols):                         # 定義：逐欄比較截止日以前的值
    c = pd.Timestamp(cut)
    a = full[full.date <= c].set_index(["date", "sid"]).sort_index()
    b = trunc[trunc.date <= c].set_index(["date", "sid"]).sort_index()
    rows = [{"截止日": cut, "欄位": "（列）", "完整資料列數": len(a), "截斷資料列數": len(b),
             "只在一邊出現的列": len(a.index.symmetric_difference(b.index)), "不一致筆數": np.nan, "最大相對差": np.nan}]
    j = a.index.intersection(b.index)
    for col in cols:
        if col not in a.columns or col not in b.columns: continue
        x, y = a.loc[j, col], b.loc[j, col]
        if x.dtype == bool or y.dtype == bool or x.dtype == object:   # 布林／文字：完全相同才算一致
            bad = ~((x.fillna(-1).astype(str)) == (y.fillna(-1).astype(str))); rel = np.nan
        else:
            x, y = x.astype(float), y.astype(float)
            both_nan = x.isna() & y.isna()
            ad = (x - y).abs().where(~both_nan, 0.0)          # 絕對差（兩邊都空值算一致）
            rd = (ad / np.maximum(x.abs(), 1e-12))            # 相對差
            bad = (ad > 1e-9 + 1e-6 * x.abs()) | (x.isna() != y.isna()); rel = float(rd[x.abs() > 1e-6].max()) if (x.abs() > 1e-6).any() else 0.0
        rows.append({"截止日": cut, "欄位": col, "完整資料列數": len(x), "截斷資料列數": len(y), "只在一邊出現的列": np.nan,
                     "不一致筆數": int(bad.sum()), "最大相對差": rel})
        if bad.any():                                        # 把不一致的例子存下來，方便追查
            ex = pd.DataFrame({"full": x[bad], "trunc": y[bad]}); ex["absdiff"] = (ex.full - ex.trunc).abs()
            ex.sort_values("absdiff", ascending=False).head(30).to_csv(OUT / f"diff_{cut}_{col}.csv")  # 差最大的 30 筆
    return rows

def garch_check(root, cut):                                  # 定義：GARCH 的輸入與參數比較
    c = pd.Timestamp(cut); stk = CFG["universe"]["stocks"]
    def rets(r):                                             # 開盤到開盤報酬（%），與 03d 相同的算法
        p = pd.read_parquet(r / "data" / "processed" / "panel.parquet")
        returns = opening_returns(p[p.sid.isin(stk)], read_actions(r))
        return returns[returns.date <= c].set_index(["date", "sid"]).r
    a, b = rets(ROOT), rets(root); j = a.index.intersection(b.index)
    fa = pd.read_csv(ROOT / "outputs" / "v7" / "tables" / "garch_oo_fits.csv", parse_dates=["fit_end"], dtype={"sid": str})
    fb = pd.read_csv(root / "outputs" / "v7" / "tables" / "garch_oo_fits.csv", parse_dates=["fit_end"], dtype={"sid": str})
    m = fa[fa.fit_end <= c].merge(fb, on=["sid", "fit_end"], suffixes=("_f", "_t"))  # 同一檔、同一次估計
    out = {"截止日": cut, "報酬筆數": len(j), "報酬最大絕對差（%）": float((a[j] - b[j]).abs().max()),
           "只在一邊出現的報酬筆數": len(a.index.symmetric_difference(b.index)),
           "估計次數": len(m), "狀態不同次數": int((m.status_f != m.status_t).sum())}
    for k in ("alpha", "beta", "omega"):                     # 參數的最大相對差
        out[f"{k} 最大相對差"] = float(((m[f"{k}_f"] - m[f"{k}_t"]).abs() / m[f"{k}_f"].abs()).max())
    m.assign(rel_beta=(m.beta_f - m.beta_t).abs() / m.beta_f).sort_values("rel_beta", ascending=False).head(15).to_csv(OUT / f"garch_fit_diff_{cut}.csv", index=False)
    return out

if __name__ == "__main__":
    for c in CUTOFFS:                                        # 先清掉這些截止日上一次留下的差異檔
        for f in OUT.glob(f"*_{c}*.csv"): f.unlink()
    full, full_el = load_from(ROOT)                          # 正式（完整資料）版本
    with ThreadPoolExecutor(max_workers=5) as ex:            # 截止日同時跑
        results = list(ex.map(run_pipeline, CUTOFFS))
    rows, G = [], []
    for (d, rc), cut in zip(results, CUTOFFS):
        if rc != 0: rows.append({"截止日": cut, "欄位": "（重跑失敗，見 log）"}); continue
        tr, tr_el = load_from(d); G.append(garch_check(d, cut))
        rows += compare(full, tr, cut, FEATS + [c for c in EXTRA if c not in FEATS])
        rows += [dict(r, 欄位="eligible（samples 全部列）") for r in compare(full_el, tr_el, cut, ["eligible"])[1:]]
    T = pd.DataFrame(rows)
    old = OUT / "pit_summary.csv"                            # 之前跑過的其他截止日結果保留，同一截止日以新結果取代
    if old.exists(): T = pd.concat([pd.read_csv(old).query("截止日 not in @CUTOFFS"), T]).sort_values("截止日", kind="stable")
    T.to_csv(OUT / "pit_summary.csv", index=False, encoding="utf-8-sig")
    print(T.to_string(index=False))
    G = pd.DataFrame(G); og = OUT / "pit_garch.csv"
    if og.exists(): G = pd.concat([pd.read_csv(og).query("截止日 not in @CUTOFFS"), G]).sort_values("截止日")
    G.to_csv(og, index=False, encoding="utf-8-sig"); print(G.to_string(index=False))
    n_bad = int(T["不一致筆數"].fillna(0).sum())
    print("PIT 結論：", "全部一致，沒有偷看未來" if n_bad == 0 else f"有 {n_bad} 筆不一致，見 diff_*.csv")
    for c in CUTOFFS: shutil.rmtree(WORK / c, ignore_errors=True)  # 刪掉暫存副本（結果已存在 outputs/v7/tables/pit）
    for f in ("loaded.parquet", "eligible.parquet"): (ROOT / f).unlink(missing_ok=True)  # 刪掉讀取用的暫存檔
