# models.py — 模型定義與評估工具（驗證期與封存測試共用，確保兩者程式完全相同）
import subprocess, datetime as dt, json                      # 載入：執行 git、時間、JSON
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from sklearn.linear_model import Ridge, Lasso, ElasticNet   # 載入：Ridge、Lasso、ElasticNet 線性模型
from lightgbm import LGBMRegressor                           # 載入：LightGBM 樹模型
from common import ROOT, CFG, PROC, TAB, nw_mean_ci          # 載入：路徑、設定、HAC 工具

STK = CFG["universe"]["stocks"]                              # 15 檔股票
SEED = CFG["protocol"]["seed"]                               # 隨機種子 42
FLOOR = 0.01                                                 # 預測波動下限（年化 1%），負值截斷用
BASE = ["hv5", "hv10", "hv20", "ret1", "ret5", "ret20", "hl", "oc", "ma5_dist", "ma20_dist",
        "vol_ratio", "vol_cv", "amt_ratio", "rsi", "atr_p"]  # 20 日版 15 項特徵的基本名稱

HOLDOUT_START = pd.Timestamp(CFG["protocol"]["holdout"]["test"][0])  # 封存期起點

def feat_cols(L=20, fset="all15"):                           # 定義：依回看期與特徵組合取得欄位名稱
    if fset in ("v2", "v3", "v4", "v7"): return list(CFG["features"][fset])  # v7：V7 說明書版本  # 第二、三版特徵（setup.yaml 的清單）
    if fset == "v1": fset = "all15"                          # v1 即文件原始 15 項
    a, b = max(2, round(L / 4)), round(L / 2)                # 短、中窗
    names = [f"hv{a}", f"hv{b}", f"hv{L}", "ret1", f"ret{a}", f"ret{L}", "hl", "oc", f"ma{a}_dist", f"ma{L}_dist",
             "vol_ratio", "vol_cv", "amt_ratio", "rsi", "atr_p"]  # 對應 15 項特徵
    if fset == "hv3": names = names[:3]                      # 只用歷史波動 3 項（H2 對照組）
    return [f"{n}_L{L}" for n in names]                      # 加上後綴回傳

def load():                                                  # 定義：讀入 15 檔股票的合格樣本（含第二版特徵）
    df = pd.read_parquet(PROC/"samples.parquet")             # 讀樣本（標籤、v1 特徵、合格標記）
    f2 = PROC/"features_v2.parquet"                          # 第二版特徵檔（src/03b 產生）
    if f2.exists(): df = df.merge(pd.read_parquet(f2, columns=["date", "sid"] + CFG["features"]["v2"]), on=["date", "sid"], how="left")  # 合併 v2 特徵
    f30 = PROC/"features_30.parquet"                         # 組員提出的 30 項特徵（src/features_30.py 產生）
    if f30.exists():                                         # 有檔案才合併
        x30 = pd.read_parquet(f30); keep = [c for c in x30.columns if c not in df.columns or c in ("date", "sid")]  # 只取尚未存在的欄位
        df = df.merge(x30[keep], on=["date", "sid"], how="left")  # 合併
    fg = PROC/"features_garch_oo.parquet"                    # V7：開盤到開盤 GARCH（src/03d 產生）
    if fg.exists(): df = df.merge(pd.read_parquet(fg, columns=["date", "sid", "garch_oo5"]), on=["date", "sid"], how="left")  # 合併
    f3 = PROC/"universe_v3.parquet"                          # v3 股票池與每週決策日（src/universe.py 產生）
    if f3.exists(): df = df.merge(pd.read_parquet(f3, columns=["date", "sid", "adv250", "tradable", "is_decision_wk"]), on=["date", "sid"], how="left")  # 合併
    return df[df.sid.isin(STK) & df.eligible].copy()         # 只留股票與合格列（股票池規則與 v1 相同，v2 缺值由模型處理）

def split(df, train_end, test_start, test_end, label_cap=None, dec_col="is_decision"):  # 定義：切出訓練集（含 purging）與測試集
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)   # 測試起訖
    tr = df[(df.date >= pd.Timestamp(CFG["data"]["study_start"])) & df.y.notna()
            & (df.label_end < ts) & (df.date <= pd.Timestamp(train_end))]  # 訓練：標籤必須在測試開始前結束（purge）
    te_ = df[df[dec_col].fillna(False).astype(bool) & df.y.notna() & (df.date >= ts) & (df.date <= te)]  # 測試：只評估決策日（v3 用每週最後交易日）
    if ts < HOLDOUT_START: te_ = te_[te_.label_end < HOLDOUT_START]  # 驗證期：標籤必須在封存期前結束（依標籤結束日封住，範本規則）
    if label_cap is not None: te_ = te_[te_.label_end <= pd.Timestamp(label_cap)]  # 標籤結束日不得晚於研究終點
    return tr, te_                                           # 回傳兩個集合

class RidgeModel:                                            # 定義：線性模型（Ridge／Lasso／ElasticNet；含只用訓練資料的補值、截尾與標準化）
    def __init__(self, alpha=None, kind="ridge", alpha_frac=None, l1_ratio=1.0):  # 記住設定
        self.alpha, self.kind, self.alpha_frac, self.l1_ratio = alpha, kind, alpha_frac, l1_ratio  # 存起來
    def fit(self, X, y):                                     # 訓練
        self.med = X.median(); X = X.fillna(self.med)        # 缺值用訓練期中位數補（v1 特徵在合格列沒有缺值，結果不變）
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)  # 訓練資料的 1%、99% 分位數（截尾界線）
        Z = X.clip(self.lo, self.hi, axis=1)                 # 截尾
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)   # 訓練資料的平均與標準差
        Zs = (Z - self.mu) / self.sd                         # 標準化
        if self.kind == "ridge": self.m = Ridge(alpha=self.alpha).fit(Zs, y)  # Ridge
        else:                                                # Lasso／ElasticNet：懲罰＝這個 fold 的 alpha_max × 比例（範本 06）
            a_max = np.abs(Zs.values.T @ (y.values - y.mean())).max() / (len(y) * self.l1_ratio)  # 讓所有係數為 0 的最小懲罰
            est = Lasso if self.kind == "lasso" else ElasticNet  # 選估計器
            kw = {} if self.kind == "lasso" else {"l1_ratio": self.l1_ratio}  # ElasticNet 需要 L1 比例
            self.m = est(alpha=a_max * self.alpha_frac, max_iter=5000, **kw).fit(Zs, y)  # 訓練
        return self                                          # 回傳自己
    def predict(self, X):                                    # 預測
        Z = X.fillna(self.med).clip(self.lo, self.hi, axis=1)  # 用訓練中位數補值、訓練界線截尾
        return self.m.predict((Z - self.mu) / self.sd)       # 用訓練平均與標準差標準化後預測

def make_model(kind, params):                                # 定義：依名稱建立模型
    if kind == "ridge": return RidgeModel(params["alpha"])   # Ridge
    if kind in ("lasso", "enet"): return RidgeModel(kind=kind, alpha_frac=params["alpha_frac"], l1_ratio=params.get("l1_ratio", 1.0))  # Lasso／ElasticNet
    kw = {"alpha": params["huber_delta"]} if params.get("objective") == "huber" else {}  # Huber 的轉折點
    return LGBMRegressor(objective=params.get("objective", "l1"), learning_rate=0.05, random_state=SEED, verbose=-1, n_jobs=4,
                         num_leaves=params["num_leaves"], min_child_samples=params["min_child_samples"],
                         n_estimators=params["n_estimators"], **kw)  # LightGBM（v1/v2 預設 MAE 損失；v3 另有 l2、huber）

def fit_predict(kind, params, tr, te, cols):                 # 定義：訓練並預測，回傳預測值與負值次數
    if kind == "hv":                                         # 歷史波動基準：直接用回看期波動
        p = te[cols[2]].values; return p, 0                  # cols[2] 是 20 日波動（v1：hv20；v2：rv20）
    if kind == "train_mean":                                 # 訓練期平均：常數預測
        return np.full(len(te), tr.y.mean()), 0              # 每筆都預測訓練平均
    m = make_model(kind, params).fit(tr[cols], tr.y)         # 訓練模型
    p = np.asarray(m.predict(te[cols]), float)               # 預測
    neg = int((p < FLOOR).sum())                             # 低於下限的次數
    return np.maximum(p, FLOOR), neg                         # 截到下限並回傳

def metrics(te, p):                                          # 定義：計算 MAE、RMSE、IC 等
    t = te[["date", "sid", "y"]].assign(p=p)                 # 合併實際與預測
    e = t.p - t.y                                            # 誤差
    ic = t.groupby("date").apply(lambda g: g.p.rank().corr(g.y.rank()) if len(g) >= 5 else np.nan, include_groups=False).dropna()  # 每日排名相關
    m, lo, hi, _ = nw_mean_ci(ic.values)                     # HAC 平均與區間
    cs = t.groupby("date")[["p", "y"]].mean()                # 每日平均預測與實際
    return {"MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "IC": m, "IC_lo": lo, "IC_hi": hi,
            "ICIR": ic.mean() / ic.std(), "IC_pos_share": (ic > 0).mean(),
            "TS_corr_level": np.corrcoef(cs.p, cs.y)[0, 1], "n": len(t), "n_dates": t.date.nunique()}  # 回傳字典

def git_commit():                                            # 定義：取得目前程式版本
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()  # git 版本號
    except Exception: return "nogit"                         # 沒有 git 時

def data_snapshot():                                         # 定義：取得資料版本（下載時間）
    return json.load(open(ROOT/"data"/"raw"/"manifest.json"))["downloaded_at"]  # 讀 manifest

def log_runs(rows):                                          # 定義：把執行紀錄附加到 run_log.csv
    f = ROOT/"run_log.csv"                                   # 紀錄檔路徑
    df = pd.DataFrame(rows)                                  # 轉成表格
    df.insert(0, "logged_at", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))  # 記錄時間
    df.insert(1, "git_commit", git_commit()); df.insert(2, "data_snapshot", data_snapshot())  # 程式與資料版本
    if f.exists(): df = pd.concat([pd.read_csv(f, encoding="utf-8-sig"), df], ignore_index=True)  # 與既有紀錄合併（欄位依名稱對齊，避免錯位）
    df.to_csv(f, index=False, encoding="utf-8-sig")          # 整檔寫回（只增加列，不改舊列）
