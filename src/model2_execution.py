import numpy as np
import pandas as pd


def execute_plan(plan, cal, O, Cl, ADV, start, end, *, capital, deposit_rate, slip, comm, tax):
    nxt = pd.Series(cal[1:], index=cal[:-1])
    days = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    targets = {nxt[p["date"]]: p for p in plan if p["date"] in nxt.index and nxt[p["date"]] in days}  # 成交日 → 配置
    if not targets: return None
    first = min(targets); days = days[days >= first]
    cash, hold = capital, pd.Series(dtype=float)              # 現金、持股（還原股數）
    last_close = Cl.ffill()                                   # 停牌日用最後收盤價評價
    nav, expo, trades, wk, prev = [], [], [], [], None
    for d in days:
        if prev is not None:                                  # 現金計息（日曆天）
            cash *= 1 + deposit_rate(d) * (d - prev).days / 365
        if d in targets:                                      # 成交日：開盤調整
            p = targets[d]; o = O.loc[d]; px_val = o.combine_first(last_close.loc[prev] if prev is not None else o)
            V = cash + float((hold * px_val.reindex(hold.index)).sum())  # 開盤前總資產
            tgt_w = pd.Series(p["w"], index=p["sids"])
            names = sorted(set(hold.index) | set(tgt_w.index))
            can = {s: not np.isnan(o.get(s, np.nan)) for s in names}  # 開盤有成交價才可交易
            scale = 1.0                                       # 若全額投入，成本會讓現金變負 → 把股票目標等比例縮小
            for _ in range(20):
                new = {s: (tgt_w.get(s, 0.0) * V * scale / o[s]) if can[s] else hold.get(s, 0.0) for s in names}
                dv = {s: (new[s] - hold.get(s, 0.0)) * o[s] for s in names if can[s]}
                buys = sum(v for v in dv.values() if v > 0); sells = -sum(v for v in dv.values() if v < 0)
                cost = (buys + sells) * (comm + slip) + sells * tax   # 手續費＋滑價（雙邊）＋證交稅（賣方）
                cash_after = cash + sells - buys - cost
                if cash_after >= -1e-6: break
                scale *= 1 + 1.01 * cash_after / max(tgt_w.sum() * V * scale, 1.0)  # 不足的金額從股票目標扣掉
            hold = pd.Series(new); hold = hold[hold.abs() > 1e-12]; cash = cash_after
            for s, v in dv.items():
                if abs(v) > 1e-6: trades.append(dict(exec=d, sid=s, value=v, adv20=ADV.get((p["date"], s), np.nan)))  # 成交額 ÷ 決策日 20 日均額
            wk.append(dict(date=p["date"], exec=d, n=len(p["sids"]), a=p["a"], sigma_p=p["sigma_p"], method=p["method"], exception=p["exception"],
                           rb_dev=float(np.abs(p["rc"] - p["b"]).max()), lam=p["lam"], est_fail=p["est_fail"], cost=cost, turnover=buys + sells, V=V))
        sv = float((hold * last_close.loc[d].reindex(hold.index)).sum())  # 收盤持股市值
        nav.append((d, cash + sv)); expo.append(sv / (cash + sv)); prev = d
    N = pd.DataFrame(nav, columns=["date", "nav"]).set_index("date"); N["expo"] = expo
    return N, pd.DataFrame(trades), pd.DataFrame(wk)
