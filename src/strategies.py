# strategies.py — 產生每個決策日的預測、風險門檻與目標權重（驗證與封存測試共用）
# v3 新增（預設值不變，v1／v2 結果與原本逐位元相同）：
#   ・parse 認得 Lasso／ElasticNet／三種損失的 LightGBM 名稱
#   ・predict_period 可用每週決策日、可只留可交易股票
#   ・make_target 可用書面提案的 R_t 規則、(股票, 黃金) 三級權重、反波動率＋20% 上限＋超出部分重新分配
import re, numpy as np, pandas as pd                         # 載入：文字比對、數值運算、表格
from common import CFG                                       # 載入：設定
from models import make_model, feat_cols, split, FLOOR, STK  # 載入：模型工具

PF = CFG["portfolio"]; GOLD, BM = CFG["universe"]["gold_etf"], CFG["universe"]["passive_benchmark"]  # 配置設定與代號

def parse(cfg):                                              # 定義：把設定名稱轉回（模型種類, 參數）
    if cfg.startswith("ridge"): return "ridge", {"alpha": float(cfg.split("alpha")[1])}  # Ridge（v1 與 v3 名稱相同）
    if cfg.startswith("lasso"): return "lasso", {"alpha_frac": float(cfg.split("frac")[1])}  # v3：Lasso
    if cfg.startswith("enet"): return "enet", {"alpha_frac": float(cfg.split("frac")[1]), "l1_ratio": CFG["v3"]["models"]["enet_l1_ratio"]}  # v3：ElasticNet
    m = re.match(r"lgbm_(l1|l2|huber)_L(\d+)_m(\d+)_t(\d+)$", cfg)  # v3：lgbm_損失_L葉數_m葉樣本_t樹數
    if m:                                                    # 是 v3 名稱
        return "lgbm", {"objective": m[1], "num_leaves": int(m[2]), "min_child_samples": int(m[3]), "n_estimators": int(m[4])}  # 只訓練 t 棵樹＝checkpoint t 的預測（同種子、逐棵相同）
    v = [int(x) for x in pd.Series([cfg]).str.extract(r"num_leaves(\d+)_min_child_samples(\d+)_n_estimators(\d+)").iloc[0]]  # v1 名稱：取出數字
    return "lgbm", dict(zip(["num_leaves", "min_child_samples", "n_estimators"], v))  # v1 LightGBM（MAE 損失）

def predict_period(df, model_cfg, train_end, start, end, fset="all15", dec_col="is_decision", tradable_only=False):  # 定義：用截至 train_end 的資料訓練，預測 start~end 所有決策日（不需標籤）
    cols = feat_cols(20, fset)                               # 特徵欄位（預設 v1 的 15 項）
    tr, _ = split(df, train_end, start, end, dec_col=dec_col)  # 訓練集（含 purge；決策日欄位只影響測試集）
    te = df[df[dec_col].fillna(False).astype(bool) & (df.date >= pd.Timestamp(start)) & (df.date <= pd.Timestamp(end))]  # 所有決策日列（不看未來標籤）
    if tradable_only: te = te[te.tradable.fillna(False).astype(bool)]  # v3：只交易通過流動性門檻的股票（模型仍用全部合格股票訓練）
    if model_cfg == "hv": p = te[cols[2]].values             # 歷史波動基準（cols[2]＝20 日波動）
    else:                                                    # 機器學習模型
        kind, prm = parse(model_cfg)                         # 解析設定
        if prm.get("objective") == "huber": prm["huber_delta"] = CFG["v3"]["models"]["lgbm"]["huber_delta_frac"] * tr.y.std()  # Huber 轉折點（與驗證相同算法）
        p = np.maximum(np.asarray(make_model(kind, prm).fit(tr[cols], tr.y).predict(te[cols]), float), FLOOR)  # 訓練、預測、截下限
    keep = ["date", "sid", "y", "label_end"] + [c for c in ("rv5",) if c in te.columns]  # 輸出欄（v3 的 R_t 需要 5 日歷史波動）
    return te[keep].assign(p=p)                              # 回傳預測表

def thresholds(df, train_end, valid_start, q=(33, 67), dec_col="is_decision", tradable_only=False):  # 定義：訓練期「每決策日平均實現波動」的分位數門檻
    tr = df[df[dec_col].fillna(False).astype(bool) & df.y.notna() & (df.date >= pd.Timestamp(CFG["data"]["study_start"]))
            & (df.date <= pd.Timestamp(train_end)) & (df.label_end < pd.Timestamp(valid_start))]  # 只用訓練期
    if tradable_only: tr = tr[tr.tradable.fillna(False).astype(bool)]  # v3：和交易的股票同一群
    m = tr.groupby("date").y.mean()                          # 每日平均實現波動
    return np.percentile(m, q[0]), np.percentile(m, q[1])    # 回傳低、高門檻

def cap_redistribute(iw, cap):                               # 定義：權重上限＋超出部分依原比例分給未達上限者（書面提案第一層）
    w = iw / iw.sum()                                        # 先正規化成總和 1
    for _ in range(len(w)):                                  # 最多重複檔數次就一定收斂
        over = w > cap + 1e-12                               # 超過上限的股票
        if not over.any(): break                             # 沒有超過：完成
        extra = (w[over] - cap).sum(); w[over] = cap         # 砍到上限，記下多出來的
        free = w < cap - 1e-12                               # 還沒到上限的股票
        if not free.any(): break                             # 全部都到上限（檔數 < 1/上限）：多的留現金
        w[free] += extra * w[free] / w[free].sum()           # 依原本比例分出去
    return w                                                 # 回傳（總和 ≤ 1；不足 5 檔時其餘為現金）

def make_target(pred, thr_by_date, tiers=(0.7, 0.5, 0.3), gold=PF["gold_weight"], alloc="ew", static=None, rule="level", ratio=(0.9, 1.1), cap=None):  # 定義：產生目標權重函數
    P = pred.set_index(["date", "sid"])                      # 以（日期, 代號）查預測
    eps = PF["epsilon_vol"]                                  # 反波動率分母下限
    log = []                                                 # 記錄每日風險等級
    def mix(k):                                              # 定義：第 k 級的（股票, 黃金）比例
        t = tiers[k]                                         # 該級設定
        return (t[0], t[1]) if isinstance(t, (tuple, list)) else (t, gold)  # v3：(股票, 黃金)；v1：只有股票，黃金固定
    def f(d):                                                # 每個決策日呼叫
        w = pd.Series(0.0, index=STK + [GOLD, BM])           # 先全部設 0
        try: q = P.loc[d]                                    # 當日各股預測
        except KeyError: q = P.iloc[0:0].droplevel(0)        # 沒有預測就空
        q = q[q.index.isin(STK) & q.p.notna()]               # 只留股票
        p = q.p                                              # 預測波動
        if static == "bm": w[BM] = 1.0; return w             # 被動基準：全部 0050
        if len(p) == 0: return w                             # 沒有可交易股票：全部現金
        if static == "ew100": sw, g = 1.0, 0.0               # 100% 股票等權
        elif static == "fixed": sw, g = mix(1)               # 固定中風險配置（v1：50/20/30；v3：50/20/30）
        else:                                                # 動態配置
            avg = p.mean()                                   # 平均預測波動
            if rule == "ratio":                              # 書面提案：R_t＝平均預測 ÷ 平均 5 日歷史波動
                lo, hi = ratio; x = avg / q.rv5.mean()       # 門檻與訊號
            else: lo, hi = thr_by_date(d); x = avg           # 期中簡報：平均預測波動 vs 訓練期分位數
            if rule == "ratio": tier = 0 if x < lo else (1 if x <= hi else 2)  # R_t：0.9 ≤ R ≤ 1.1 為中風險（書面提案寫法）
            else: tier = 0 if x < lo else (1 if x < hi else 2)  # 水準規則：與 v1 相同的寫法（結果不變）
            sw, g = mix(tier)                                # 股票與黃金比例
            log.append({"date": d, "rule": rule, "signal": x, "avg_pred": avg, "lo": lo, "hi": hi, "tier": ["低", "中", "高"][tier]})  # 記錄
        if alloc in ("iv", "iv_cap"): iw = 1 / np.maximum(p, eps)  # 反波動率
        else: iw = pd.Series(1.0, index=p.index)             # 等權
        if alloc == "iv_cap": w[p.index] = sw * cap_redistribute(iw, cap or PF["per_name_cap"])  # v3：股票總比例 × 上限＋重新分配後權重（回測時 run(cap=False)）
        else: w[p.index] = sw * iw / iw.sum()                # v1：股票總比例 × 股內權重（與原本同一個算式；上限由回測引擎套用，超出留現金）
        w[GOLD] = g                                          # 黃金
        return w                                             # 回傳目標權重（其餘為現金）
    f.log = log                                              # 讓外部讀取風險等級紀錄
    return f                                                 # 回傳函數
