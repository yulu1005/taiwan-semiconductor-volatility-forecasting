# bt.py — 事件驅動回測引擎：t 收盤決策 → t+1 開盤整張成交 → 每日收盤評價（驗證與封存測試共用）
import numpy as np, pandas as pd                             # 載入：數值運算、表格
from common import CFG, PROC                                 # 載入：設定與路徑

STK, GOLD, BM = CFG["universe"]["stocks"], CFG["universe"]["gold_etf"], CFG["universe"]["passive_benchmark"]  # 股票、黃金、基準代號
C = CFG["costs"]; X = CFG["execution"]; PF = CFG["portfolio"]  # 成本、執行、配置設定
LOT = X["lot_size"]                                          # 一張 = 1000 股

def tick_slip(o, base):                                      # 定義：v3 滑價＝max(固定比例, 半個升降單位 ÷ 成交價)（範本 01 B.3）
    T = CFG["v3"]["costs"]                                   # 升降單位表
    def tick(px, table):                                     # 定義：依價格查升降單位
        conds = [px < float(up) for up, _ in table]               # 「股價 < 上限」的條件（由小到大）
        return np.select(conds, [float(t) for _, t in table], default=float(table[-1][1]))  # 第一個成立的級距
    etf = o.index.isin([GOLD, BM])                           # ETF 用另一張表
    t = pd.Series(np.where(etf, tick(o.values, T["tick_table_etf"]), tick(o.values, T["tick_table_stock"])), index=o.index)  # 每檔升降單位
    return np.maximum(base, 0.5 * t / o).fillna(base)        # 每邊滑價（無開盤價者用固定比例，反正不會成交）

class Market:                                                # 定義：把面板資料整理成「日期 × 代號」矩陣，方便查詢
    def __init__(self):                                      # 初始化
        p = pd.read_parquet(PROC/"panel.parquet")            # 讀面板
        s = pd.read_parquet(PROC/"samples.parquet")          # 讀樣本（含 adv20、合格標記）
        piv = lambda d, c: d.pivot(index="date", columns="sid", values=c).sort_index()  # 定義：轉成矩陣
        self.open, self.close = piv(p, "open"), piv(p, "close")  # 原始開盤、收盤（無效日為空值）
        self.high, self.low = piv(p, "high"), piv(p, "low")  # 原始最高、最低
        self.adj, self.aclose, self.aopen = piv(p, "adj"), piv(p, "aclose"), piv(p, "aopen")  # 還原因子與還原價
        self.adv = piv(s, "adv20").reindex(self.open.index)  # 過去 20 日平均成交金額
        self.elig = piv(s.assign(e=s.eligible.astype(float)), "e").reindex(self.open.index).fillna(0) > 0  # 合格標記
        self.dates = self.open.index                         # 交易日曆
        self.mark = self.close.ffill()                       # 評價用：最後有效收盤價（只用來算市值，不用來成交）
        prev = self.aclose.ffill().shift(1)                  # 前一日還原收盤
        lock = (self.high == self.low)                       # 一價到底
        self.lock_up = lock & (self.aopen / prev - 1 >= 0.095)    # 開盤鎖漲停：不能買
        self.lock_dn = lock & (self.aopen / prev - 1 <= -0.095)   # 開盤鎖跌停：不能賣
        self.ca = (self.adj / self.adj.shift(1)).fillna(1.0) # 公司行動：持股數每日調整倍數

def run(mk, decisions, target_fn, start_cash=None, cost_mult=1.0, slip_mult=1.0, adv_cap=None, cap=None,
        overlay=False, end=None, gross=False, slip_model="fixed"):  # 定義：執行一次回測（slip_model="tick" 為 v3；預設 fixed 與 v1 相同）
    cash = float(start_cash or X["capital_twd"])             # 起始現金
    adv_cap = X["adv_cap"] if adv_cap is None else adv_cap   # 成交量限制（None 表示用設定值）
    comm = 0 if gross else C["commission_per_side"] * cost_mult  # 手續費率
    slip = 0 if gross else C["slippage_per_side"] * cost_mult * slip_mult  # 滑價率
    sh = pd.Series(0.0, index=mk.open.columns)               # 持股數
    dates = mk.dates[(mk.dates >= decisions[0]) & (mk.dates <= (end or mk.dates[-1]))]  # 回測期間
    exe = {mk.dates[mk.dates.get_loc(t) + 1]: t for t in decisions if mk.dates.get_loc(t) + 1 < len(mk.dates)}  # 成交日 → 決策日
    rec, trades, peak, pending = [], [], None, None          # 紀錄、交易明細、淨值高點、待執行目標
    for i, d in enumerate(dates):                            # 逐日
        if i > 0: sh *= mk.ca.loc[d]                         # 開盤前套用除權息／減資（含息再投入）
        if d in exe and pending is not None:                 # 今天是成交日
            t = exe[d]; w = pending                          # 對應的決策日與目標權重
            o = mk.open.loc[d]                               # 今日開盤價
            V = cash + (sh * o.fillna(mk.mark.loc[mk.dates[mk.dates.get_loc(d) - 1]])).sum()  # 開盤時組合市值
            tgt = (w * V / (o * LOT)).fillna(0).apply(np.floor) * LOT  # 目標股數（整張，無開盤價則為 0）
            delta = tgt - sh                                 # 需要買賣的股數
            ok = o.notna()                                   # 有有效開盤價才可成交
            delta[~ok] = 0                                   # 停牌不成交
            delta[(delta > 0) & mk.lock_up.loc[d].fillna(False)] = 0   # 鎖漲停不買
            delta[(delta < 0) & mk.lock_dn.loc[d].fillna(False)] = 0   # 鎖跌停不賣
            small = (delta.abs() < LOT) & (tgt > 0)          # 不到一張的零股差額（多半來自配股）不交易
            delta[small] = 0                                 # 略過
            lim = (adv_cap * mk.adv.loc[t] / o).fillna(np.inf) if adv_cap else pd.Series(np.inf, index=o.index)  # 單筆上限股數
            clipped = delta.abs() > lim                      # 超過容量限制
            delta[clipped] = np.sign(delta[clipped]) * np.floor(lim[clipped] / LOT) * LOT  # 縮到上限（整張）
            unfilled = float(((tgt - sh - delta).abs() * o).fillna(0)[ok].sum())  # 未成交金額
            if slip_model == "tick" and not gross: slip = tick_slip(o, C["slippage_per_side"]) * cost_mult * slip_mult  # v3：每檔、每天依成交價計算滑價（成本倍數同樣放大）
            tax = pd.Series(C["sell_tax_stock"], index=o.index); tax[[GOLD, BM]] = C["sell_tax_etf"]  # 證交稅率
            if gross: tax[:] = 0                             # 毛報酬：不計稅
            sells = delta.clip(upper=0).abs()                # 賣出股數
            sv = (sells * o * (1 - slip)).fillna(0)          # 賣出金額（扣滑價）
            cost_s = (sells * o * slip).fillna(0).sum() + (sv * comm).sum() + (sv * tax).sum()  # 賣出成本
            cash += (sv * (1 - comm - tax)).sum(); sh -= sells   # 先賣：入帳並減少持股
            buys = delta.clip(lower=0)                       # 買進股數
            bv = (buys * o * (1 + slip) * (1 + comm)).fillna(0)  # 買進所需現金
            if bv.sum() > cash:                              # 現金不足
                scale = cash / bv.sum()                      # 縮小比例
                buys = (buys * scale / LOT).apply(np.floor) * LOT  # 依比例縮單（整張）
                bv = (buys * o * (1 + slip) * (1 + comm)).fillna(0)  # 重算
            cost_b = (buys * o * slip).fillna(0).sum() + (buys * o * (1 + slip) * comm).fillna(0).sum()  # 買進成本
            cash -= bv.sum(); sh += buys                     # 再買：扣款並增加持股
            trades.append({"date": d, "decision": t, "traded": float((sells * o).fillna(0).sum() + (buys * o).fillna(0).sum()),
                           "cost": cost_s + cost_b, "unfilled": unfilled, "n_clipped": int(clipped.sum())})  # 記錄交易
            pending = None                                   # 清除待執行目標
        px = mk.mark.loc[d]                                  # 今日評價價格
        hold = (sh * px).fillna(0)                           # 各資產市值
        V = cash + hold.sum()                                # 收盤總市值
        peak = V if peak is None else max(peak, V)           # 更新淨值高點
        rec.append({"date": d, "value": V, "cash": cash, "stock_w": hold[hold.index.isin(STK)].sum() / V,
                    "gold_w": hold.get(GOLD, 0) / V, "bm_w": hold.get(BM, 0) / V, "dd": V / peak - 1})  # 記錄每日狀態
        if d in decisions and d != dates[-1]:                # 今天收盤是決策日
            cur_sw = hold[hold.index.isin(STK)].sum() / V    # 目前股票權重
            w = target_fn(d)                                 # 依策略計算目標權重
            if cap is not False:                             # 個股上限
                c = PF["per_name_cap"] if cap is None else cap   # 上限
                sw = w[w.index.isin(STK)].sum()              # 股票總比例
                if sw > 0: w[w.index.isin(STK)] = np.minimum(w[w.index.isin(STK)], c * sw)  # 每檔 ≤ 股票部位 20%，多的留現金
            if overlay and rec[-1]["dd"] <= -CFG["risk"]["kill_switch"]["halt"]:  # 停損：回撤達 15%
                sw = w[w.index.isin(STK)].sum()              # 目標股票比例
                if sw > cur_sw and sw > 0: w[w.index.isin(STK)] *= cur_sw / sw  # 不增加股票曝險
            pending = w.reindex(mk.open.columns).fillna(0)   # 存成待執行目標
    eq = pd.DataFrame(rec).set_index("date"); tr = pd.DataFrame(trades)  # 轉成表格
    return eq, tr                                            # 回傳淨值與交易

def perf(eq, tr, start_cash=None):                           # 定義：計算績效指標
    v = eq.value; r = v.pct_change().dropna(); n = len(r); yrs = n / 252  # 淨值、日報酬、天數、年數
    sr_d = r.mean() / r.std() if r.std() > 0 else np.nan     # 日 Sharpe
    sr = sr_d * np.sqrt(252); se = np.sqrt((1 + sr_d ** 2 / 2) / n) * np.sqrt(252)  # 年化 Sharpe 與標準誤
    dd = v / v.cummax() - 1; trough = dd.idxmin(); pk = v.loc[:trough].idxmax()  # 回撤、谷底、前高
    rec_ = v.loc[trough:][v.loc[trough:] >= v.loc[pk]]       # 回到前高的日子
    recov = (eq.index.get_loc(rec_.index[0]) - eq.index.get_loc(trough)) if len(rec_) else np.nan  # 恢復天數
    traded = tr.traded.sum() if len(tr) else 0; cost = tr.cost.sum() if len(tr) else 0  # 總成交額與總成本
    to = traded / v.mean() / yrs                             # 年化雙邊換手率
    return {"CAGR": (v.iloc[-1] / v.iloc[0]) ** (1 / yrs) - 1, "cum_ret": v.iloc[-1] / v.iloc[0] - 1, "ann_vol": r.std() * np.sqrt(252),
            "Sharpe": sr, "Sharpe_lo": sr - 1.96 * se, "Sharpe_hi": sr + 1.96 * se, "MDD": dd.min(),
            "MDD_days": eq.index.get_loc(trough) - eq.index.get_loc(pk), "recovery_days": recov,
            "turnover_2way": to, "avg_hold_years": 2 / to if to > 0 else np.nan, "cost_pct_capital": cost / v.iloc[0],
            "cost_drag_ann": cost / v.mean() / yrs, "avg_stock_w": eq.stock_w.mean(), "avg_gold_w": eq.gold_w.mean(),
            "avg_cash_w": (eq.cash / v).mean(), "unfilled_avg": tr.unfilled.mean() if len(tr) else 0,
            "skew": r.skew(), "kurt": r.kurt() + 3, "n_days": n}  # 回傳指標字典

def block_boot_sharpe_diff(ra, rb, block=5, B=2000, seed=42):  # 定義：週區塊重抽樣估計 Sharpe 差異區間
    rng = np.random.default_rng(seed)                        # 隨機數產生器
    a, b = ra.values, rb.values; n = len(a); nb = int(np.ceil(n / block))  # 序列與區塊數
    out = []                                                 # 收集
    for _ in range(B):                                       # 重複 B 次
        st = rng.integers(0, n - block, nb)                  # 隨機區塊起點
        idx = (st[:, None] + np.arange(block)).ravel()[:n]   # 展開成索引
        x, y = a[idx], b[idx]                                # 抽出的兩組報酬
        out.append((x.mean() / x.std() - y.mean() / y.std()) * np.sqrt(252))  # 年化 Sharpe 差
    d = (ra.mean() / ra.std() - rb.mean() / rb.std()) * np.sqrt(252)  # 實際差異
    return d, np.percentile(out, 2.5), np.percentile(out, 97.5)  # 回傳差異與 95% 區間
