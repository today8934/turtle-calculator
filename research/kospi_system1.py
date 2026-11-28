"""Cash-only Turtle System 1 study. Run from the repository root with market/{year}.{parquet,csv}."""
import json
import math
import re
import argparse
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2025-11-28")
END = pd.Timestamp("2026-11-27")
CAPITAL = 100_000_000
TOP = 30
YEARS = range(2024, 2027)
OUT = Path("output")
OUT.mkdir(exist_ok=True)


def data():
    indexes = [pd.read_csv(f"market/{y}.csv", parse_dates=["Date"]).set_index("Date")["Close"] for y in YEARS]
    index = pd.concat(indexes).sort_index().loc[:END]
    assert index.index.is_unique and index.at[START] == 3926.59 and index.at[END] == 4915.68
    frames = []
    for y in YEARS:
        d = pd.read_parquet(f"market/{y}.parquet")
        if "Date" not in d:
            d = d.reset_index()
        d = d.loc[d.Market.eq("KOSPI")].copy()
        d["Date"] = pd.to_datetime(d["Date"].astype(str), format="mixed")
        d["Code"] = d.Code.astype(str).str.zfill(6)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    for day in index.loc[START:].index.difference(pd.DatetimeIndex(d.Date.unique())):
        snapshot = ("https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/"
                    "259713bf397e5e28fb0da531a19b78a3ca15766b/data/listing/krx/"
                    f"{day.date()}.csv")
        daily = pd.read_csv(snapshot, dtype={"Code":str}).rename(columns={"ChagesRatio":"ChangesRatio"})
        daily = daily.loc[daily.Market.eq("KOSPI")].assign(Date=day)
        print("Supplemented KRX snapshot", day.date(), len(daily), flush=True)
        d = pd.concat([d, daily], ignore_index=True)
    d = d.loc[d.Date.isin(index.index)].sort_values(["Code", "Date"])
    assert not d.duplicated(["Code", "Date"]).any()
    counts = d.groupby("Date").size().reindex(index.index)
    assert counts.loc[START:].min() >= 750 and not counts.loc[START:].isna().any(), counts.nsmallest(10)
    exposure = d.pivot(index="Date", columns="Code", values="Amount").reindex(index.index)
    listed = exposure.notna()
    rolling = exposure.rolling(20, min_periods=20).mean().where(listed).shift()
    ranks = rolling.rank(axis=1, ascending=False, method="first")
    names = d.groupby("Code").Name.last().to_dict()
    year = exposure.loc[exposure.index > START].sum().sort_values(ascending=False)
    test = ranks.loc[ranks.index > START]
    for size in (TOP, 100):
        annual = year.head(size).index
        daily = test.le(size)
        common = daily[annual].sum(axis=1)
        assert daily.sum(axis=1).eq(size).all() and len(common) == 243
        uses = daily.sum()
        stats = dict(size=size, days=len(common), mean=common.mean(), median=common.median(),
                     min=int(common.min()), max=int(common.max()), first=int(common.iloc[0]), last=int(common.iloc[-1]),
                     min_date=str(common.idxmin().date()), max_date=str(common.idxmax().date()),
                     distinct=int((uses > 0).sum()), ever_annual=int((uses[annual]>0).sum()),
                     always=int(uses[annual].eq(len(common)).sum()),
                     at_least_25=int(common.ge(25).sum()), top10_avg=float(daily[annual[:10]].sum(axis=1).mean()),
                     last10_avg=float(daily[annual[-10:]].sum(axis=1).mean()))
        print("OVERLAP", json.dumps(stats), flush=True)
        if size == TOP:
            detail = pd.DataFrame({"annual_rank":year.rank(ascending=False, method="first").astype(int),
                                   "annual_krw":year, "days_in_rolling_top30":uses})
            detail["name"] = detail.index.map(names)
            detail.loc[detail.days_in_rolling_top30.gt(0) | detail.annual_rank.le(TOP)].to_csv(OUT/"top30_members.csv")
            print("CORE", [(names[c], int(uses[c])) for c in annual[:10]], flush=True)
            print("MARGINAL", [(names[c], int(uses[c])) for c in uses[annual].nsmallest(5).index], flush=True)
            print("OUTSIDERS", [(names[c], int(uses[c])) for c in uses.drop(annual).nlargest(5).index], flush=True)
            first_date = common.idxmin()
            selected = set(daily.columns[daily.loc[first_date]])
            print("MIN_EXAMPLES", sorted(names[c] for c in set(annual)-selected), sorted(names[c] for c in selected-set(annual)), flush=True)
            overlap = pd.DataFrame({"common":common, "different_slots":size-common})
            overlap["rolling_codes"] = daily.apply(lambda row: " ".join(row.index[row]), axis=1)
            overlap.to_csv(OUT/"top30_overlap_by_date.csv")
    final = set(year.head(TOP).index)
    order = sorted(set(test.columns[test.le(TOP).any()]).union(final))
    return index, d.loc[d.Code.isin(order)], order, names, ranks[order], final


def ticks(price):
    return np.select([price < 2000, price < 5000, price < 20000, price < 50000, price < 200000, price < 500000],
                     [1, 5, 10, 50, 100, 500], default=1000)


def trigger(adjusted, factor, up):
    raw = adjusted / factor
    # A strict downward break at an exact tick-band boundary uses the lower band's increment.
    tick = ticks(raw + (1e-7 if up else -1e-7))
    units = raw / tick
    return ((np.floor(units + 1e-9) + 1 if up else np.ceil(units - 1e-9) - 1) * tick) * factor


def executable(price, factor, up):
    raw = price / factor
    tick = float(ticks(raw + (1e-7 if up else -1e-7)))
    units = raw / tick
    return (math.ceil(units - 1e-9) if up else math.floor(units + 1e-9)) * tick * factor


def panels(index, data, order, use_closes_for_channels=False):
    fields = "o h l c factor n u20 d20 u55 d10 u10".split()
    mats = {k: np.full((len(index), len(order)), np.nan) for k in fields}
    last = np.zeros(len(order), dtype=int)
    mapping = {code: i for i, code in enumerate(order)}
    adjustments = 0
    for code, d in data.groupby("Code"):
        k = mapping[code]
        last[k] = index.index.get_loc(d.Date.max())
        d = d.loc[(d[["Open", "High", "Low", "Close"]].min(axis=1) > 0) & (d.Volume > 0)].set_index("Date")
        if d.empty:
            continue
        raw = d.Close.astype(float)
        diff = d.Changes.astype(float)
        ref = raw - diff
        ratio = (raw.shift() / ref.where(ref > 0)).fillna(1)
        adjustments += int((abs(ratio - 1) > .001).sum())
        factor = ratio.cumprod()
        o, h, l, c = [d[col] * factor for col in ("Open", "High", "Low", "Close")]
        tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        seeded = tr.copy()
        seeded.iloc[:19] = np.nan
        if len(tr) >= 20:
            seeded.iloc[19] = tr.iloc[:20].mean()
        atr = seeded.ewm(alpha=1 / 20, adjust=False).mean()
        week = d.index.to_period("W-FRI")
        weekly = atr.groupby(week).last().reindex(pd.period_range(week.min(), week.max(), freq="W-FRI")).ffill()
        n = pd.Series(weekly.reindex(week - 1).to_numpy(), index=d.index)
        pos = index.index.get_indexer(d.index)
        values = dict(o=o, h=h, l=l, c=c, factor=factor, n=n)
        for key, bounds, length, up in [("u20", h, 20, True), ("d20", l, 20, False),
                                        ("u55", h, 55, True), ("d10", l, 10, False), ("u10", h, 10, True)]:
            if use_closes_for_channels:
                bounds = c
            channel = bounds.rolling(length).max() if up else bounds.rolling(length).min()
            values[key] = trigger(channel.shift(), factor, up)
        for field, arr in values.items():
            mats[field][pos, k] = arr
    return mats, last, adjustments


def breakout_events(m, reverse):
    # Track every instrument before the test begins. The previous hypothetical trade may be long or short.
    rows, cols = m["o"].shape
    if reverse == "near":
        high_is_nearer = (m["h"] - m["o"]) < (m["o"] - m["l"])
        m["first"] = np.where(high_is_nearer, m["h"], m["l"])
        m["second"] = np.where(high_is_nearer, m["l"], m["h"])
        phases = ["o", "first", "second", "c"]
    else:
        phases = ["o", "h", "l", "c"] if reverse else ["o", "l", "h", "c"]
    events = np.full((rows, 4, cols), np.nan)
    for k in range(cols):
        side = 0
        won = False
        pending55 = False
        fill = stop = np.nan
        for i in np.where(np.isfinite(m["n"][:, k]) & (m["n"][:, k] > 0))[0]:
            n, f, u, down, long_exit, short_exit, far = [m[a][i, k] for a in ("n","factor","u20","d20","d10","u10","u55")]
            if not np.isfinite(u):
                continue
            prev = None
            entered_long = entered_short = False
            for phase, col in enumerate(phases):
                p = m[col][i, k]
                gap = phase == 0
                rising = gap or p > prev
                falling = gap or p < prev
                # At the open orders crossed by a gap execute at the opening print.
                if side == 1 and falling and p <= max(stop, long_exit):
                    x = p if gap else max(stop, long_exit)
                    won = x > fill + 1e-7
                    side, pending55 = 0, False
                elif side == -1 and rising and p >= min(stop, short_exit):
                    x = p if gap else min(stop, short_exit)
                    won = x < fill - 1e-7
                    side, pending55 = 0, False
                if not side and rising and p >= u and (gap or prev < u) and not entered_long:
                    fill = p if gap else u
                    side, stop = 1, executable(fill - 2*n, f, False)
                    pending55, entered_long = won, True
                    if not won:
                        events[i, phase, k] = fill
                elif not side and falling and p <= down and (gap or prev > down) and not entered_short:
                    fill = p if gap else down
                    side, stop = -1, executable(fill + 2*n, f, True)
                    pending55, entered_short = False, True
                if side == 1 and pending55 and rising and p >= far and (gap or prev < far):
                    events[i, phase, k] = p if gap else far
                    pending55 = False
                prev = p
    return phases, events


def correlation_limits(index, m):
    closes = pd.DataFrame(m["c"], index=index.index)
    returns = closes.pct_change(fill_method=None)
    # No later price enters either the tie-breaker or the correlation estimate.
    strength = (closes.shift(1) - closes.shift(64)) / m["n"]
    priority = -strength.fillna(-1e30).to_numpy()
    all_corr = {}
    for i in range(index.index.get_loc(START)+1, len(index)):
        corr = returns.iloc[i-60:i].corr(min_periods=40).fillna(1.).to_numpy()
        np.fill_diagonal(corr, 1.)
        all_corr[i] = corr
    return priority, all_corr


def room(k, pos, corr):
    # Check the candidate and all held names as group anchors. A sector is not required.
    active = sorted(set(pos) | {k})
    units = np.array([len(pos[s]["lots"]) if s in pos else 0 for s in active])
    units[active.index(k)] += 1
    if units.sum() > 12 or units[active.index(k)] > 4:
        return False
    pairs = corr[np.ix_(active, active)]
    return bool(((pairs >= .70) @ units <= 6).all() and ((pairs >= .40) @ units <= 10).all())


def simulate(index, m, last, order, names, priority, eligible, phases, events, label, correlations,
             buy_fee=.0005, sell_fee=.0025, retry=False, align_dec19=None, closing_auction=False):
    initial = CAPITAL
    cash = equity = float(initial)
    annual_base = notional = float(initial)
    next_cut = annual_base * .90
    pos, marks, ledger, curve = {}, {}, [], []
    history = index.index
    start = history.get_loc(START)
    curve.append([START.date(), equity, cash, 0, index.at[START]])
    def record(k, action, price, qty, cause):
        ledger.append([history[i].date(), stage, order[k], names[order[k]], action,
                       price / fac if fac else 0, qty * fac if fac else qty, cause, cash])

    def buy_lot(k, fill, qty, n, held):
        nonlocal cash
        cash -= qty * fill * (1 + buy_fee)
        if held is None:
            held = dict(qty=0.,units=0,unit=qty,n=n,latest=fill,lots=[])
            pos[k] = held
        else:
            held["lots"] = [(executable(s + .5*n, fac, False), q) for s,q in held["lots"]]
        held["qty"] += qty
        held["units"] += 1
        held["latest"] = fill
        held["lots"].append((executable(fill - 2*n, fac, False), qty))
        record(k, "buy", fill, qty, "entry" if held["units"]==1 else "add")
        waiting.pop(k, None)
        return held

    for i in range(start + 1, len(index)):
        waiting = {}
        sales = 0
        if history[i].year != history[i-1].year:
            annual_base = notional = equity
            next_cut = annual_base * .90
        if equity >= annual_base:
            notional = annual_base
            next_cut = annual_base * .90
        while equity <= next_cut and notional >= 1:
            notional *= .8
            next_cut -= notional * .10
        eligible_today = eligible[i]
        for k in list(pos):
            if last[k] < i:
                stage, fac = -1, 1.
                record(k, "sell", 0, pos[k]["qty"], "delisting_zero_recovery")
                del pos[k]
        align_today = align_dec19 is not None and history[i]==pd.Timestamp("2025-12-19")
        for stage, field in enumerate(phases):
            signal = np.flatnonzero(np.isfinite(events[i, stage]) & eligible_today)
            active = sorted(set(pos).union(signal), key=lambda k: (priority[i,k], order[k]))
            watch = label.startswith("rolling_") and history[i] == pd.Timestamp("2026-04-08")
            if watch:
                watch_k = order.index("009150")
                print("WATCH_PHASE", label, stage, "bar", *[m[x][i, watch_k]/m["factor"][i,watch_k] for x in ["o","h","l","c"]],
                      "price", events[i,stage,watch_k]/m["factor"][i,watch_k], "eligible", eligible_today[watch_k],
                      "holdings", [(names[order[j]],len(p["lots"])) for j,p in pos.items()], "cash", cash, flush=True)
            for k in active:
                target, fac = m[field][i, k], m["factor"][i,k]
                if not np.isfinite(target):
                    continue
                gap = stage == 0 or closing_auction
                previous = target if gap else m[phases[stage - 1]][i,k]
                if closing_auction:
                    # Two accounting passes over the same closing print: release all sale proceeds, then allocate buys.
                    rising, falling = stage == 1, stage == 0
                else:
                    rising, falling = gap or target > previous, gap or target < previous
                hold = pos.get(k)
                if hold is not None and falling:
                    channel = m["d10"][i,k]
                    remaining = []
                    for stop, qty in sorted(hold["lots"], reverse=True):
                        level = max(stop, channel)
                        if target > level:
                            remaining.append((stop, qty))
                            continue
                        px = target if gap else min(previous, level)
                        cash += qty * px * (1 - sell_fee)
                        sales += 1
                        waiting.pop(k, None)
                        record(k, "sell", px, qty, "2N" if stop >= channel else "10D")
                    hold["lots"] = remaining
                    hold["qty"] = sum(q for _,q in remaining)
                    if not remaining:
                        del pos[k]
                        hold = None
                price = events[i, stage, k]
                if hold is None and rising and eligible_today[k] and np.isfinite(price) and room(k, pos, correlations[i]):
                    n = m["n"][i,k]
                    desired = math.floor(notional * .01 / (n / fac))
                    shares = min(desired, math.floor(cash / (price / fac * (1 + buy_fee))))
                    if watch and order[k]=="009150":
                        print("WATCH_ORDER",label,stage,"price",price/fac,"desired",desired,"affordable",shares,"cash",cash,flush=True)
                    if shares > 0 and shares == desired:
                        hold = buy_lot(k, price, shares / fac, n, None)
                    elif retry and desired > 0:
                        waiting.setdefault(k, (None, desired, n, sales))
                if hold is None or not rising:
                    continue
                while hold["units"] < 4:
                    if align_today and order[k] == "005930":
                        break
                    level = executable(hold["latest"] + .5*hold["n"], fac, True)
                    if target < level or not room(k, pos, correlations[i]):
                        break
                    px = target if gap else max(previous, level)
                    desired = math.floor(hold["unit"]*fac + 1e-8)
                    shares = min(desired, math.floor(cash / (px / fac * (1 + buy_fee))))
                    if shares < 1 or shares < desired:
                        if retry and desired > 0:
                            waiting.setdefault(k, (hold, desired, hold["n"], sales))
                        break
                    buy_lot(k, px, shares / fac, hold["n"], hold)
        if align_today:
            k = order.index("005930")
            fac = m["factor"][i,k]
            held = pos[k]
            assert abs(held["qty"]*fac-265)<1e-6 and room(k,pos,correlations[i])
            assert 265*align_dec19*(1+buy_fee) < cash
            stage = 4
            buy_lot(k,align_dec19*fac,265/fac,held["n"],held)
            print("ALIGN_DEC19",label,"price",align_dec19,"cash",cash,"holdings",
                  [(order[k], round(v["qty"]*m["factor"][i,k],2)) for k,v in pos.items()],flush=True)
        if retry:
            # A fixed, independently observable execution time, with no use of a fabricated quote at the sale timestamp.
            stage = 4
            for k in sorted(list(waiting), key=lambda j: (priority[i,j],order[j])):
                original, desired, n, sale_count_at_miss = waiting[k]
                if sales <= sale_count_at_miss or pos.get(k) is not original or not room(k,pos,correlations[i]):
                    continue
                fill, fac = m["c"][i,k], m["factor"][i,k]
                if fill <= m["d10"][i,k] or desired * fill / fac * (1 + buy_fee) > cash:
                    continue
                buy_lot(k, fill, desired/fac, n, original)
                print("RETRY_FILL", label, history[i].date(), order[k], names[order[k]], desired, round(fill/fac,2),flush=True)
        for k in pos:
            if np.isfinite(m["c"][i,k]):
                marks[k] = m["c"][i,k]
        equity = cash + sum(p["qty"] * marks[k] for k,p in pos.items())
        assert cash > -1e-4 and equity >= 0 and sum(len(p["lots"]) for p in pos.values()) <= 12
        curve.append([history[i].date(), equity, cash, len(pos), index.iloc[i]])
    curve = pd.DataFrame(curve, columns=["date","equity","cash","holdings","index"])
    trades = pd.DataFrame(ledger, columns=["date","phase","code","name","action","raw_price","raw_shares","reason","cash_after"])
    if closing_auction:
        executions = trades.loc[trades.reason.ne("delisting_zero_recovery")]
        trade_dates = history.get_indexer(pd.to_datetime(executions.date))
        trade_codes = pd.Index(order).get_indexer(executions.code)
        assert (trade_dates >= 0).all() and (trade_codes >= 0).all()
        assert np.allclose(executions.raw_price, (m["c"]/m["factor"])[trade_dates,trade_codes], atol=1e-6, rtol=0)
        assert executions.loc[executions.action.eq("sell"),"phase"].eq(0).all()
        buys = executions.loc[executions.action.eq("buy")]
        assert buys.phase.eq(1).all() and not buys.duplicated(["date","code"]).any()
    cash_changes = trades.raw_price * trades.raw_shares * np.where(trades.action.eq("sell"), 1-sell_fee, -1-buy_fee)
    assert np.isclose(initial + cash_changes.sum(), cash, atol=.01, rtol=0)
    attribution = cash_changes.groupby(trades.code).sum()
    for k,p in pos.items():
        attribution.loc[order[k]] += p["qty"] * marks[k]
    assert np.isclose(initial + attribution.sum(), equity, atol=.01, rtol=0)
    attribution.rename("profit_krw").sort_values(ascending=False).to_csv(OUT / f"{label}_stock_profit.csv")
    curve.to_csv(OUT / f"{label}_daily.csv", index=False)
    trades.to_csv(OUT / f"{label}_fills.csv", index=False)
    summary = dict(label=label, initial=initial, final=round(equity), ret=round((equity / initial - 1)*100, 4),
                   mdd=round((curve.equity / curve.equity.cummax() - 1).min()*100, 4),
                   entries=int(trades.reason.eq("entry").sum()), adds=int(trades.reason.eq("add").sum()),
                   exits=int(trades.reason.eq("entry").sum())-len(pos), held=len(pos),
                   delist=int(trades.reason.eq("delisting_zero_recovery").sum()),
                   exposure=round((1 - curve.cash.iloc[1:]/curve.equity.iloc[1:]).mean()*100, 2),
                   candidates=int(eligible[start + 1:].any(axis=0).sum()), traded=int(trades.code.nunique()),
                   max_names=int(curve.holdings.max()))
    pit = (curve.equity / curve.equity.cummax()).idxmin()
    peak = curve.equity.loc[:pit].idxmax()
    summary["peak_date"], summary["trough_date"] = str(curve.date.at[peak]), str(curve.date.at[pit])
    summary["peak_equity"], summary["trough_equity"] = round(curve.equity.at[peak]), round(curve.equity.at[pit])
    if label == "base":
        print("ATTRIBUTION", [(names[c], round(v)) for c,v in pd.concat(
            [attribution.nlargest(3), attribution.nsmallest(3)]).items()], flush=True)
    print("RESULT", json.dumps(summary), flush=True)
    return summary


def close_signals(m):
    """At each closing print, finish the shadow trade before checking for a fresh breakout."""
    dates, codes = m["c"].shape
    events = np.full((dates, 2, codes), np.nan)
    for k in range(codes):
        side, previous_won, waiting_for_55 = 0, False, False
        fill = stop = np.nan
        for i in np.flatnonzero((m["n"][:,k] > 0) & np.isfinite(m["u20"][:,k])):
            c, f, n = m["c"][i,k], m["factor"][i,k], m["n"][i,k]
            if side == 1 and c <= max(stop,m["d10"][i,k]):
                previous_won = c > fill + 1e-7
                side, waiting_for_55 = 0, False
            elif side == -1 and c >= min(stop,m["u10"][i,k]):
                previous_won = c < fill - 1e-7
                side, waiting_for_55 = 0, False
            if side == 0:
                if c >= m["u20"][i,k]:
                    side, fill, stop = 1, c, executable(c - 2*n,f,False)
                    waiting_for_55 = previous_won
                    if not previous_won:
                        events[i,1,k] = c
                elif c <= m["d20"][i,k]:
                    side, fill, stop = -1, c, executable(c + 2*n,f,True)
            if side == 1 and waiting_for_55 and c >= m["u55"][i,k]:
                events[i,1,k] = c
                waiting_for_55 = False
    return events


def close_study():
    index, quotes, order, names, ranks, annual = data()
    rolling = ranks.le(TOP).to_numpy()
    hindsight = np.broadcast_to([c in annual for c in order], rolling.shape)
    reference = index.loc[START:END]
    print("CLOSE_BENCHMARK", len(reference)-1, reference.iloc[0], reference.iloc[-1],
          (reference.iloc[-1]/reference.iloc[0]-1)*100, (reference/reference.cummax()-1).min()*100, flush=True)
    results = []
    for use_closes, basis in [(False,"prior_high_low"), (True,"prior_closes")]:
        m, last, _ = panels(index, quotes, order, use_closes_for_channels=use_closes)
        priority, correlations = correlation_limits(index, m)
        signals = close_signals(m)
        for selection, eligible in [("rolling",rolling), ("annual",hindsight)]:
            label = f"close_{basis}_{selection}"
            results.append(simulate(index,m,last,order,names,priority,eligible,("c","c"),signals,
                                    label,correlations,closing_auction=True))
    pd.DataFrame(results).to_csv(OUT/"summary.csv",index=False)


def main():
    observed = check_december_prices()
    index, data_, order, names, ranks, final = data()
    m, last, adj = panels(index, data_, order)
    ref = index.loc[START:]
    print("BENCH", len(ref)-1, ref.iloc[0], ref.iloc[-1], (ref.iloc[-1]/ref.iloc[0]-1)*100,
          (ref/ref.cummax()-1).min()*100, "source adjustments", adj, flush=True)
    priority, correlations = correlation_limits(index, m)
    rolling = ranks.le(TOP).to_numpy()
    post = np.broadcast_to([c in final for c in order], rolling.shape)
    results = []
    for reverse, tag in [(False,"low_first"),(True,"high_first"),("near","near_first")]:
        path, events = breakout_events(m, reverse)
        for name, eligible in [("rolling",rolling),("annual",post)]:
            results.append(simulate(index,m,last,order,names,priority,eligible,path,events,
                                    f"{name}_{tag}",correlations))
        results.append(simulate(index,m,last,order,names,priority,rolling,path,events,
                                f"retry_{tag}",correlations,retry=True))
        results.append(simulate(index,m,last,order,names,priority,rolling,path,events,
                                f"aligned_{tag}",correlations,align_dec19=observed))
        if tag == "low_first":
            compare_entries()
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)
    explain_price_order(index, m, order)
    for tag in ["low_first","high_first","near_first"]:
        p = pd.read_csv(OUT/f"retry_{tag}_stock_profit.csv", dtype={"code":str}).set_index("code")
        print("RETRY_SAMSUNG_ELECTRO",tag,p.profit_krw.get("009150",0.),flush=True)


def check_december_prices():
    day = pd.Timestamp("2025-12-19",tz="Asia/Seoul")
    minute = {}
    for code in ["005930", "005380"]:
        try:
            params = {"count":400,"session":"regular","investMode":"krx","useAdjustedRate":"false",
                      "to":"2025-12-19T15:31:00+09:00"}
            url = f"https://wts-info-api.tossinvest.com/api/v1/c-chart/kr-s/A{code}/min:1?"+urlencode(params)
            with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0"}),timeout=8) as response:
                obj = json.load(response)
            candles = obj.get("result",obj).get("candles",[])
            if candles:
                print("MINUTE_SAMPLE",code,candles[0],candles[-1],flush=True)
                c = pd.DataFrame(candles)
                stamp = next(k for k in ["dt","dateTime","timestamp","time"] if k in c)
                dt = pd.to_datetime(c[stamp],utc=True).dt.tz_convert("Asia/Seoul")
                c = c.set_index(dt).sort_index()
                c = c.loc[(c.index.date==day.date()) & (c.index.time>=day.replace(hour=9).time())]
                if len(c) >= 300:
                    minute[code] = c
                    c.to_csv(OUT/f"dec19_minute_{code}.csv")
        except (OSError,ValueError,KeyError,TypeError,StopIteration) as exc:
            print("MINUTE unavailable",code,repr(exc),flush=True)
        try:
            url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.KS"
                   f"?interval=60m&period1={int(day.timestamp())}&period2={int((day+pd.Timedelta(days=1)).timestamp())}")
            request = Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urlopen(request,timeout=10) as response:
                result = json.load(response)["chart"]["result"][0]
            frame = pd.DataFrame(result["indicators"]["quote"][0],
                                 index=pd.to_datetime(result["timestamp"],unit="s",utc=True).tz_convert("Asia/Seoul"))
            frame = frame.loc[frame.index.date==day.date()]
            frame.to_csv(OUT/f"dec19_hourly_{code}.csv")
            print("HOURLY",code,frame.to_string().replace("\n"," | "),flush=True)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            print("HOURLY unavailable",code,repr(exc),flush=True)
        try:
            url = f"https://finance.naver.com/item/sise_time.naver?code={code}&thistime=20251219153000&page=1"
            with urlopen(Request(url,headers={"User-Agent":"Mozilla/5.0"}),timeout=8) as response:
                html = response.read().decode("euc-kr",errors="replace")
            rows = []
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>",html,re.S|re.I):
                cells = [re.sub(r"<.*?>","",c,flags=re.S).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>",tr,re.S|re.I)]
                if cells and re.fullmatch(r"\d\d:\d\d",cells[0]):
                    rows.append(cells[:2])
            print("NAVER sample",code,rows[:3],flush=True)
        except (OSError, ValueError) as exc:
            print("NAVER unavailable",code,repr(exc),flush=True)
    if len(minute)==2:
        car, sam = minute["005380"], minute["005930"]
        if "low" in car and "high" in sam and car.low.le(279500).any():
            sold = car.index[car.low.le(279500)][0]
            if sold not in sam.index:
                return 106300.
            print("OBSERVED_MINUTE", str(sold), sam.loc[sold].to_dict(),
                  sam.loc[sam.index>sold,"high"].max(), sam.loc[sam.high.ge(108000)].index.max(), flush=True)
            if "close" in sam and 106200 <= sam.at[sold,"close"] <= 107100:
                return float(sam.at[sold,"close"])
    return 106300.  # Official closing auction, after the observed Hyundai 10:00-10:59 sale window.


def explain_price_order(index, m, order):
    logs, curves, pnl = {}, {}, {}
    for path in ["low_first", "high_first"]:
        stem = f"rolling_{path}"
        logs[path] = pd.read_csv(OUT/f"{stem}_fills.csv", dtype={"code":str})
        curves[path] = pd.read_csv(OUT/f"{stem}_daily.csv").set_index("date")
        pnl[path] = pd.read_csv(OUT/f"{stem}_stock_profit.csv", dtype={"code":str}).set_index("code").profit_krw
    gap = curves["high_first"].equity - curves["low_first"].equity
    def emit(tag, record):
        print(tag, json.dumps(record, ensure_ascii=False, default=str), flush=True)
    first = gap.index[gap.abs()>1][0]
    emit("FIRST_EQUITY_DATE", first)
    first_trades = set()
    for path,log in logs.items():
        sample = log.loc[log.date <= first, ["date","name","action","raw_price","raw_shares","reason","cash_after"]]
        emit("FIRST_EQUITY_"+path, sample.tail(12).round(2).to_dict("records"))
        for _,row in log.loc[log.date == first].iterrows():
            first_trades.add(row.code)
        emit("FIRST_MARK_"+path, curves[path].loc[first].to_dict())
    i = index.index.get_loc(pd.Timestamp(first))
    for code in sorted(first_trades):
        k = order.index(code)
        emit("OHLC_"+code, {field:round(float(m[field][i,k]/m["factor"][i,k]),2) for field in ["o","h","l","c","n","u20","d10","u55"]})
    # An explicit same-day buy/stop instance where the two paths treated the same security differently.
    for path, log in logs.items():
        other = logs["high_first" if path=="low_first" else "low_first"]
        count = 0
        for (date,code), rows in log.groupby(["date","code"],sort=True):
            if not (rows.reason.eq("entry").any() and rows.action.eq("sell").any()):
                continue
            comp = other.loc[(other.date==date)&(other.code==code)]
            if not comp.reason.eq("entry").any() or comp.action.eq("sell").any():
                continue
            emit("ENTRY_STOP_"+path, [date,code,rows[["name","reason","raw_price","raw_shares"]].round(2).values.tolist(),comp[["reason","raw_price","raw_shares"]].round(2).values.tolist()])
            count += 1
            if count==2:
                break
    p = pd.DataFrame(pnl).fillna(0)
    names = pd.concat(logs.values()).drop_duplicates("code").set_index("code").name
    p["delta"] = p.high_first-p.low_first
    p["name"] = p.index.map(names)
    assert abs(p.delta.sum()-gap.iloc[-1])<1
    emit("PROFIT_DIFFERENCE", pd.concat([p.nlargest(6,"delta"),p.nsmallest(3,"delta")]).round(0).to_dict("records"))
    for code in p.nlargest(2,"delta").index:
        for path, log in logs.items():
            emit("TOP_TRADES_"+path, log.loc[log.code==code,["date","name","reason","raw_price","raw_shares"]].round(2).values.tolist())
    entries = [set(zip(t.date,t.code)) for log in logs.values() for t in [log.loc[log.reason=="entry"]]]
    emit("ENTRY_OVERLAP_PATHS", [len(x) for x in entries]+[len(entries[0]&entries[1])])


def compare_entries():
    calendars = {}
    for mode in ["rolling", "annual"]:
        label = f"{mode}_low_first"
        trades = pd.read_csv(OUT / f"{label}_fills.csv",dtype={"code":str})
        trades = trades.loc[trades.reason.eq("entry")]
        calendars[mode] = set(zip(trades.date, trades.code))
    common = calendars["rolling"] & calendars["annual"]
    print("SAME_ENTRY", len(common), len(calendars["rolling"]), len(calendars["annual"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--close-only", action="store_true")
    if parser.parse_args().close_only:
        close_study()
    else:
        main()

