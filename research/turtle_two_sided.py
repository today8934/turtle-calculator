"""Intraday long/short stock simulation with 100% cash collateral on direct shorts."""
import heapq
import math

import numpy as np
import pandas as pd

from kospi_system1 import CAPITAL, OUT, TICK_REFORM, executable, room


def simulate_two_sided(index, m, last, order, names, priority, eligible, phases, events,
                       label, correlations, start_date, short_eligible=None):
    """Apply coupled OHLC paths. Cash book contains locked short proceeds.

    Spendable collateral = equity minus total marked absolute exposure. Existing
    positions can drift above 100% exposure, but new risk is then rejected.
    The alternative short mask restricts opening or adding to short positions.
    """
    start = index.index.get_loc(start_date)
    cash_book = equity = risk_base = float(CAPITAL)
    risk_notional = risk_base
    next_cut = risk_base * .90
    holdings, last_marks = {}, {}
    daily, fills = [], []
    pnl = {}
    buy_fee, sell_fee = .0005, .0025
    daily.append((index.index[start], equity, cash_book, equity, 0, 0, 0))
    stopped_by_short_restriction = 0
    shorts_max_deficit = 0.0
    serial = 0
    for i in range(start + 1, len(index)):
        date = index.index[i]
        old_tick = date < TICK_REFORM
        if date.year != index.index[i-1].year:
            risk_base = risk_notional = equity
            next_cut = risk_base * .90
        if equity >= risk_base:
            risk_notional = risk_base
            next_cut = risk_base * .90
        while equity <= next_cut and risk_notional >= 1:
            risk_notional *= .8
            next_cut -= risk_notional * .1

        def account_fill(k, side, opening, pos_qty, price, why, fraction, phase):
            nonlocal cash_book
            is_buy = (side == 1) == opening
            delta = (-1-buy_fee if is_buy else 1-sell_fee) * pos_qty * price
            cash_book += delta
            pnl[(k,side)] = pnl.get((k,side),0.0) + delta
            factor = m["factor"][i,k]
            if not np.isfinite(factor):
                factor = 1.0
            fills.append((date, phase, fraction, order[k], side, "buy" if is_buy else "sell",
                          why, price / factor, pos_qty * factor, cash_book))

        for k in list(holdings):
            if last[k] < i:
                pos = holdings.pop(k)
                account_fill(k, pos["side"], False, pos["qty"], 0, "delist", 0, -1)
        pending = {}
        for stage, field in enumerate(phases):
            end = m[field][i]
            begin = m[phases[stage-1]][i] if stage else end
            queue = []

            def mark(k, t):
                if not np.isfinite(end[k]):
                    return last_marks[k]
                return begin[k] + (end[k]-begin[k])*t

            def schedule(t, kind, k, side, price, level, identifier):
                nonlocal serial
                serial += 1
                # Quantize fraction solely for deterministic tie-breaking.
                strength_order = side*priority[i,k] if priority[i,k] < 1e29 else 1e30
                heapq.heappush(queue, (round(t,12), int(kind!="exit"), strength_order,
                                       order[k], serial, kind, k, side, price, level, identifier))

            def crossing(k, side, level, after=0):
                if not np.isfinite(level) or not np.isfinite(end[k]):
                    return None
                if stage == 0:
                    return (0., end[k]) if side*(end[k]-level) >= -1e-7 else None
                slope = end[k]-begin[k]
                if side*slope > 0:
                    time = (level-begin[k])/slope
                    if after - 1e-12 <= time <= 1 + 1e-12:
                        return min(1., max(after, time)), level
                return None

            def add_order(k, pos, after):
                if len(pos["lots"]) >= 4 or pos["units_bought"] >= 4:
                    return
                side = pos["side"]
                level = executable(pos["latest"] + side*.5*pos["n"], m["factor"][i,k], side==1, old_tick)
                hit = crossing(k, side, level, after)
                if hit is not None:
                    schedule(hit[0], "add", k, side, hit[1], level, pos["generation"])

            def exit_orders(k, pos, after):
                side = pos["side"]
                channel = m["d10" if side==1 else "u10"][i,k]
                for lot_id, (stop, qty) in pos["lots"].items():
                    live_stop = executable(stop,m["factor"][i,k],side==-1,old_tick)
                    level = max(channel,live_stop) if side==1 else min(channel,live_stop)
                    hit = crossing(k,-side,level,after)
                    if hit is not None:
                        schedule(hit[0], "exit", k, side, hit[1], level, lot_id)

            for k, pos in holdings.items():
                if not np.isfinite(end[k]):
                    continue
                exit_orders(k, pos, 0)
                add_order(k, pos, 0)
            for side, grid in ((1,events[0]),(-1,events[1])):
                active = np.flatnonzero(np.isfinite(grid[i,stage]) & eligible[i])
                for k in active:
                    level = grid[i,stage,k]
                    hit = crossing(k,side,level)
                    if hit is not None:
                        schedule(hit[0], "entry", k, side, hit[1], level, None)

            def release_pending(t):
                for key, (kind,k,side,level,identifier) in list(pending.items()):
                    if not np.isfinite(end[k]) or side*(mark(k,t)-level) < -1e-7:
                        del pending[key]
                        continue
                    fill = executable(mark(k,t),m["factor"][i,k],side==1,old_tick)
                    schedule(t,kind,k,side,fill,level,identifier)
                    del pending[key]

            while queue:
                t, _, _, _, _, kind, k, side, price, level, identifier = heapq.heappop(queue)
                pos = holdings.get(k)
                if kind == "exit":
                    if not pos or pos["side"]!=side or identifier not in pos["lots"]:
                        continue
                    stop, qty = pos["lots"][identifier]
                    live_stop = executable(stop,m["factor"][i,k],side==-1,old_tick)
                    channel = m["d10" if side==1 else "u10"][i,k]
                    current_level = max(channel,live_stop) if side==1 else min(channel,live_stop)
                    if abs(current_level-level)>1e-6:
                        continue
                    del pos["lots"][identifier]
                    account_fill(k,side,False,qty,price,"2N" if abs(live_stop-level)<1e-6 else "10D",t,stage)
                    pos["qty"] -= qty
                    if not pos["lots"]:
                        del holdings[k]
                    release_pending(t)
                    continue
                if kind == "entry" and pos is not None:
                    continue
                if kind == "add" and (not pos or pos["generation"]!=identifier or pos["units_bought"]>=4):
                    continue
                if side==-1 and short_eligible is not None and not short_eligible[i,k]:
                    stopped_by_short_restriction += 1
                    continue
                factor = m["factor"][i,k]
                n = m["n"][i,k] if kind=="entry" else pos["n"]
                shares = (math.floor(risk_notional*.01/(n/factor)) if kind=="entry"
                          else math.floor(pos["unit"]*factor + 1e-8))
                if shares<=0:
                    continue
                qty = shares/factor
                used_for_shorts = sum(2*p["qty"]*mark(j,t) for j,p in holdings.items() if p["side"]==-1)
                free = cash_book-used_for_shorts
                same_direction = {j:p for j,p in holdings.items() if p["side"]==side}
                risk_cost = price*qty*(1+(buy_fee if side==1 else sell_fee))
                if risk_cost > free + 1e-6 or not room(k,same_direction,correlations[i]):
                    pending[(kind,k,side)] = (kind,k,side,level,identifier)
                    continue
                if kind=="entry":
                    pos = dict(side=side,n=n,lots={},qty=0.,unit=qty,units_bought=0)
                    holdings[k] = pos
                else:
                    pos["lots"] = {key:(executable(stop+side*.5*n,factor,side==-1,old_tick),q)
                                   for key,(stop,q) in pos["lots"].items()}
                serial += 1
                pos["lots"][serial] = (executable(price-side*2*n,factor,side==-1,old_tick),qty)
                pos["generation"] = serial
                pos["units_bought"] += 1
                pos["latest"] = price
                pos["qty"] += qty
                last_marks[k] = price
                pending.pop((kind,k,side),None)
                account_fill(k,side,True,qty,price,kind,t,stage)
                exit_orders(k,pos,t)
                add_order(k,pos,t)
            for key, (_,k,side,level,_) in list(pending.items()):
                if not np.isfinite(end[k]) or side*(end[k]-level) < -1e-7:
                    del pending[key]

        for k in holdings:
            if np.isfinite(m["c"][i,k]):
                last_marks[k] = m["c"][i,k]
        long_value = sum(p["qty"]*last_marks[k] for k,p in holdings.items() if p["side"]==1)
        short_value = sum(p["qty"]*last_marks[k] for k,p in holdings.items() if p["side"]==-1)
        equity = cash_book + long_value - short_value
        free = cash_book - 2*short_value
        shorts_max_deficit = min(shorts_max_deficit,free)
        assert equity>0 and all(len(p["lots"])<=4 for p in holdings.values())
        daily.append((date,equity,cash_book,free,long_value,short_value,len(holdings)))

    for k, pos in holdings.items():
        pnl[(k,pos["side"])] += pos["side"]*pos["qty"]*last_marks[k]
    assert abs(CAPITAL + sum(pnl.values()) - equity) < .01
    curve = pd.DataFrame(daily,columns=["date","equity","cash_book","free_collateral","long_value","short_value","held"])
    trades = pd.DataFrame(fills,columns=["date","phase","fraction","code","side","action","reason","raw_price","raw_shares","cash_book_after"])
    stock_pnl = pd.DataFrame([(order[k],s,v) for (k,s),v in pnl.items()],columns=["code","side","profit_krw"])
    curve.to_csv(OUT/f"{label}_daily.csv",index=False)
    trades.to_csv(OUT/f"{label}_fills.csv",index=False)
    stock_pnl.to_csv(OUT/f"{label}_stock_profit.csv",index=False)
    long_pnl = sum(v for (k,s),v in pnl.items() if s==1)
    short_pnl = sum(v for (k,s),v in pnl.items() if s==-1)
    entries = trades.reason.eq("entry")
    return dict(label=label,initial=CAPITAL,final=round(equity),ret=round(100*(equity/CAPITAL-1),4),
                mdd=round(100*(curve.equity/curve.equity.cummax()-1).min(),4),
                entries=int(entries.sum()),adds=int(trades.reason.eq("add").sum()),
                exits=int(entries.sum())-len(holdings),held=len(holdings),fills=len(trades),
                delist=int(trades.reason.eq("delist").sum()),long_entries=int((entries & trades.side.eq(1)).sum()),
                short_entries=int((entries & trades.side.eq(-1)).sum()),long_pnl=round(long_pnl),
                short_pnl=round(short_pnl),min_free_collateral=round(shorts_max_deficit),
                short_orders_restricted=stopped_by_short_restriction)
