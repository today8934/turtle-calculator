"""Cash-funded long accounts on a chosen path through each day's four quotes."""
from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
import math

import numpy as np
import pandas as pd

from kospi_system1 import CAPITAL, OUT, TICK_REFORM, executable, room


BUY_FEE = 0.0005
SELL_FEE = 0.0025


def simulate_path(local_index, m, last, order, names, priority, eligible, phases, events,
                  label, correlations, start_date):
    history = local_index.index
    start = history.get_loc(start_date)
    cash = equity = base = risk_base = float(CAPITAL)
    next_cut = 0.9 * base
    positions = {}
    marks = {}
    serial = count()
    ledger = []
    curve = [[start_date.date(), equity, cash, 0, local_index.iloc[start]]]
    pnl = defaultdict(float)
    skipped = 0
    retries = 0

    for i in range(start+1, len(history)):
        day = history[i]
        old_tick = day < TICK_REFORM
        if day.year != history[i-1].year or equity >= base:
            if day.year != history[i-1].year:
                base = equity
            risk_base = base
            next_cut = 0.9 * base
        while equity <= next_cut and risk_base >= 1:
            risk_base *= 0.8
            next_cut -= 0.1 * risk_base

        def record(k, action, fill, quantity, cause, line, factor):
            raw_price = fill / factor
            raw_quantity = quantity * factor
            flow = fill * quantity * (1-SELL_FEE if action == "sell" else -(1+BUY_FEE))
            pnl[k] += flow
            ledger.append([day.date(), stage, at, order[k], names[order[k]], action,
                           raw_price, raw_quantity, cause, cash, line/factor])

        stage, at = -1, 0.0
        pending = {}
        for k in list(positions):
            if last[k] < i:
                for stop, shares in positions.pop(k)["lots"]:
                    record(k, "sell", 0.0, shares, "delisting_zero_recovery", 0., 1.)

        for stage, field in enumerate(phases):
            right = m[field][i]
            left = m[phases[max(0,stage-1)]][i]
            opening = stage == 0
            queue = []

            def push(at, k, kind, lvl, version=None, lot=None, retry=False):
                heappush(queue, (round(at, 12), kind != "sell", priority[i,k], order[k],
                                 next(serial), k, kind, lvl, version, lot, retry))

            def crossing(k, level, rising, after=0.0):
                begin = left[k]
                end = right[k]
                if not np.isfinite(begin) or not np.isfinite(end) or not np.isfinite(level):
                    return None
                if opening:
                    reached = end >= level if rising else end <= level
                    return 0.0 if reached else None
                if (rising and end <= begin) or (not rising and end >= begin):
                    return None
                frac = (level - begin) / (end - begin)
                if after - 1e-12 <= frac <= 1 + 1e-12:
                    return min(1., max(after, frac))
                return None

            def add_order(k, after=0.0):
                position = positions[k]
                if position["buys"] >= 4:
                    return
                next_price = executable(position["latest"] + 0.5*position["n"],
                                        m["factor"][i,k], True, old_tick)
                when = crossing(k, next_price, True, after)
                if when is not None:
                    push(when, k, "add", next_price, position["version"])

            def sell_orders(k, after=0.0):
                channel = m["d10"][i,k]
                for lot in positions[k]["lots"]:
                    stop, quantity = lot
                    # A corporate action may put yesterday's stop off today's grid.
                    live_stop = executable(stop, m["factor"][i,k], False, old_tick)
                    level = max(live_stop, channel)
                    when = crossing(k, level, False, after)
                    if when is not None:
                        cause = "2N" if live_stop >= channel else "10D"
                        push(when, k, "sell", level, lot=(lot,cause))

            # A pending order qualifies only while the breakout remains in force.
            for key, (line, version) in list(pending.items()):
                if left[key[0]] < line - 1e-7:
                    del pending[key]

            for k, position in positions.items():
                if not np.isfinite(right[k]):
                    continue
                sell_orders(k)
                add_order(k)
            for k in np.flatnonzero(eligible[i] & np.isfinite(events[i,stage])):
                line = events[i,stage,k]
                when = crossing(k, line, True)
                if when is not None:
                    push(when, k, "entry", line)

            while queue:
                at, _, _, _, _, k, kind, line, version, payload, is_retry = heappop(queue)
                fac = m["factor"][i,k]
                observed = left[k] + at * (right[k] - left[k])
                hold = positions.get(k)
                if kind == "sell":
                    lot, reason = payload
                    if hold is None or lot not in hold["lots"]:
                        continue
                    fill = right[k] if opening else line
                    shares = lot[1]
                    cash += fill * shares * (1-SELL_FEE)
                    record(k, "sell", fill, shares, reason, line, fac)
                    hold["lots"].remove(lot)
                    if not hold["lots"]:
                        del positions[k]
                        pending.pop((k,"add"), None)
                    # Sale proceeds become spendable here. Reprice a still-active
                    # missed order at the buyer's simultaneous quote.
                    for (j, kind2), (trigger, revision) in list(pending.items()):
                        quote = left[j] + at*(right[j]-left[j])
                        if quote + 1e-7 >= trigger:
                            push(at, j, kind2, trigger, revision, retry=True)
                        elif quote < trigger - 1e-7:
                            del pending[(j,kind2)]
                    continue

                if kind == "entry" and hold is not None:
                    continue
                if kind == "add" and (hold is None or version != hold["version"] or hold["buys"] >= 4):
                    continue
                key = (k,kind)
                if is_retry and pending.get(key) != (line,version):
                    continue
                if is_retry:
                    fill = executable(observed, fac, True, old_tick)
                else:
                    fill = right[k] if opening else line
                if kind == "entry":
                    n = m["n"][i,k]
                    shares = math.floor(risk_base*0.01*fac/n)
                else:
                    n = hold["n"]
                    shares = math.floor(hold["unit"]*fac + 1e-8)
                quantity = shares/fac
                cost = quantity*fill*(1+BUY_FEE)
                if shares <= 0:
                    continue
                if cost > cash + 1e-7 or not room(k, positions, correlations[i]):
                    pending[key] = (line, version)
                    skipped += not is_retry
                    continue
                if hold is None:
                    hold = {"unit":quantity, "n":n, "buys":0, "lots":[]}
                    positions[k] = hold
                else:
                    hold["lots"] = [(executable(stop + 0.5*n, fac, False, old_tick), q)
                                    for stop,q in hold["lots"]]
                hold["lots"].append((executable(fill - 2*n, fac, False, old_tick), quantity))
                hold["buys"] += 1
                hold["version"] = next(serial)
                hold["latest"] = fill
                cash -= cost
                record(k, "buy", fill, quantity, kind, line, fac)
                retries += is_retry
                pending.pop(key, None)
                # The next order must be 0.5N from the actual fill, even after an opening gap.
                add_order(k, at)
                if not opening and right[k] < left[k]:
                    # A previously blocked buy could just have filled on a falling leg.
                    sell_orders(k, at)
            # Ignore a blocked order once that excursion ended, even if the
            # value happens to climb back above its trigger on a later leg.
            for key, (line, _) in list(pending.items()):
                if right[key[0]] < line - 1e-7:
                    del pending[key]

        for k in positions:
            if np.isfinite(m["c"][i,k]):
                marks[k] = m["c"][i,k]
        equity = cash + sum(sum(q for _,q in h["lots"]) * marks[k] for k,h in positions.items())
        assert cash > -.01 and equity >= 0
        assert sum(len(h["lots"]) for h in positions.values()) <= 12
        curve.append([day.date(), equity, cash, len(positions), local_index.iloc[i]])

    trades = pd.DataFrame(ledger, columns=["date","phase","fraction","code","name","action","raw_price",
                                           "raw_shares","reason","cash_after","trigger"])
    daily = pd.DataFrame(curve, columns=["date","equity","cash","holdings","index"])
    flows = trades.raw_price * trades.raw_shares * np.where(trades.action.eq("sell"), 1-SELL_FEE, -1-BUY_FEE)
    assert abs(CAPITAL + flows.sum() - cash) < .02
    if len(trades):
        np.testing.assert_allclose(CAPITAL + flows.cumsum(), trades.cash_after, atol=.02, rtol=0)
        fills = trades.loc[trades.phase.ge(0)]
        ix = history.get_indexer(pd.to_datetime(fills.date))
        cols = pd.Index(order).get_indexer(fills.code)
        adjusted = fills.raw_price.to_numpy()*m["factor"][ix,cols]
        assert (adjusted >= m["l"][ix,cols]-1e-5).all() and (adjusted <= m["h"][ix,cols]+1e-5).all()
        at_open = fills.phase.eq(0)
        np.testing.assert_allclose(adjusted[at_open], m["o"][ix[at_open], cols[at_open]], rtol=0, atol=1e-5)
    for k, position in positions.items():
        pnl[k] += sum(q for _,q in position["lots"]) * marks[k]
    assert abs(CAPITAL + sum(pnl.values()) - equity) < .02
    attribution = pd.Series({order[k]:p for k,p in pnl.items()}, name="profit_krw", dtype=float)
    attribution.rename_axis("code").sort_values(ascending=False).to_csv(OUT / f"{label}_stock_profit.csv")
    trades.to_csv(OUT / f"{label}_fills.csv", index=False)
    daily.to_csv(OUT / f"{label}_daily.csv", index=False)
    entries = int(trades.reason.eq("entry").sum())
    return dict(label=label, initial=CAPITAL, final=round(equity), ret=round((equity/CAPITAL-1)*100,4),
                mdd=round((daily.equity/daily.equity.cummax()-1).min()*100,4),
                entries=entries, adds=int(trades.reason.eq("add").sum()), exits=entries-len(positions),
                delist=int(trades.reason.eq("delisting_zero_recovery").sum()), held=len(positions),
                fills=len(trades), opening_fills=int(trades.phase.eq(0).sum()), skipped=skipped, retries=retries)
