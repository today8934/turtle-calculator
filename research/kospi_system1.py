"""Cash-only Turtle System 1 study. Run from the repository root with market/{year}.{parquet,csv}."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


START = pd.Timestamp("2025-11-28")
END = pd.Timestamp("2026-11-27")
CAPITAL = 10_000_000
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
    rolling = exposure.fillna(0).rolling(20).mean().where(listed).shift()
    ranks = rolling.rank(axis=1, ascending=False, method="first")
    ranks1 = exposure.rank(axis=1, ascending=False, method="first").shift()
    fixed = ranks.loc[ranks.index > START].iloc[0].nsmallest(100).index.tolist()
    final = exposure.loc[(exposure.index > START)].sum().nlargest(100).index.tolist()
    members = set(fixed + final)
    for rank in [ranks, ranks1]:
        members.update(rank.columns[rank.loc[rank.index > START].le(100).any()])
    d = d.loc[d.Code.isin(members)]
    order = sorted(members)
    names = d.groupby("Code").Name.last().to_dict()
    return index, d, order, names, ranks[order], ranks1[order], set(fixed), set(final)


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


def panels(index, data, order):
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


def simulate(index, m, last, order, names, priority, eligible, phases, events, label,
             initial=CAPITAL, cap=4, buy_fee=.0005, sell_fee=.0025, partial=False):
    cash = equity = float(initial)
    pos, marks, ledger, curve = {}, {}, [], []
    history = index.index
    start = history.get_loc(START)
    curve.append([START.date(), equity, cash, 0, index.at[START]])
    def record(k, action, price, qty, cause):
        ledger.append([history[i].date(), stage, order[k], names[order[k]], action,
                       price / fac if fac else 0, qty * fac if fac else qty, cause, cash])

    for i in range(start + 1, len(index)):
        opening_equity = equity
        eligible_today = eligible[i]
        for k in list(pos):
            if last[k] < i:
                stage, fac = -1, 1.
                record(k, "sell", 0, pos[k]["qty"], "delisting_zero_recovery")
                del pos[k]
        for stage, field in enumerate(phases):
            signal = np.flatnonzero(np.isfinite(events[i, stage]) & eligible_today)
            active = sorted(set(pos).union(signal), key=lambda k: (priority[i,k], order[k]))
            for k in active:
                target, fac = m[field][i, k], m["factor"][i,k]
                if not np.isfinite(target):
                    continue
                gap = stage == 0
                previous = target if gap else m[phases[stage - 1]][i,k]
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
                        record(k, "sell", px, qty, "2N" if stop >= channel else "10D")
                    hold["lots"] = remaining
                    hold["qty"] = sum(q for _,q in remaining)
                    if not remaining:
                        del pos[k]
                        hold = None
                price = events[i, stage, k]
                units_in_account = sum(len(p["lots"]) for p in pos.values())
                if hold is None and rising and eligible_today[k] and np.isfinite(price) and len(pos) < cap and units_in_account < 12:
                    n = m["n"][i,k]
                    desired = math.floor(opening_equity * .01 / (n / fac))
                    shares = min(desired, math.floor(cash / (price / fac * (1 + buy_fee))))
                    if shares > 0 and (partial or shares == desired):
                        qty = shares / fac
                        cash -= qty * price * (1 + buy_fee)
                        hold = dict(qty=qty, units=1, unit=qty, n=n, latest=price,
                                    lots=[(executable(price - 2*n, fac, False), qty)])
                        pos[k] = hold
                        record(k, "buy", price, qty, "entry")
                if hold is None or not rising:
                    continue
                while hold["units"] < 4 and sum(len(p["lots"]) for p in pos.values()) < 12:
                    level = executable(hold["latest"] + .5*hold["n"], fac, True)
                    if target < level:
                        break
                    px = target if gap else max(previous, level)
                    desired = math.floor(hold["unit"]*fac + 1e-8)
                    shares = min(desired, math.floor(cash / (px / fac * (1 + buy_fee))))
                    if shares < 1 or (not partial and shares < desired):
                        break
                    qty = shares / fac
                    cash -= qty * px * (1 + buy_fee)
                    hold["qty"] += qty
                    hold["units"] += 1
                    hold["latest"] = px
                    hold["lots"] = [(executable(s + .5*hold["n"],fac,False), q) for s,q in hold["lots"]]
                    hold["lots"].append((executable(px - 2*hold["n"],fac,False), qty))
                    record(k, "buy", px, qty, "add")
        for k in pos:
            if np.isfinite(m["c"][i,k]):
                marks[k] = m["c"][i,k]
        equity = cash + sum(p["qty"] * marks[k] for k,p in pos.items())
        assert cash > -1e-4 and equity >= 0 and len(pos) <= cap
        curve.append([history[i].date(), equity, cash, len(pos), index.iloc[i]])
    curve = pd.DataFrame(curve, columns=["date","equity","cash","holdings","index"])
    trades = pd.DataFrame(ledger, columns=["date","phase","code","name","action","raw_price","raw_shares","reason","cash_after"])
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
                   candidates=int(eligible[start + 1:].any(axis=0).sum()))
    pit = (curve.equity / curve.equity.cummax()).idxmin()
    peak = curve.equity.loc[:pit].idxmax()
    summary["peak_date"], summary["trough_date"] = str(curve.date.at[peak]), str(curve.date.at[pit])
    summary["peak_equity"], summary["trough_equity"] = round(curve.equity.at[peak]), round(curve.equity.at[pit])
    print("RESULT", json.dumps(summary), flush=True)
    return summary


def main():
    index, data_, order, names, ranks, ranks1, first, final = data()
    m, last, adj = panels(index, data_, order)
    ref = index.loc[START:]
    print("BENCH", len(ref)-1, ref.iloc[0], ref.iloc[-1], (ref.iloc[-1]/ref.iloc[0]-1)*100,
          (ref/ref.cummax()-1).min()*100, "source adjustments", adj, flush=True)
    primary = ranks.fillna(1e9).to_numpy()
    daily = ranks1.fillna(1e9).to_numpy()
    pre = np.broadcast_to([c in first for c in order], primary.shape)
    post = np.broadcast_to([c in final for c in order], primary.shape)
    results = []
    for reverse in [False, True, "near"]:
        path, events = breakout_events(m, reverse)
        def run(name, priority, eligible, **kw):
            results.append(simulate(index,m,last,order,names,priority,eligible,path,events,name,**kw))
        if reverse == "near":
            run("near_extreme_first", primary, primary <= 100)
            run("near_extreme_first_4m", primary, primary <= 100, initial=4_000_000)
        elif reverse:
            run("high_first", primary, primary <= 100)
            run("high_first_4m", primary, primary <= 100, initial=4_000_000)
        else:
            run("base", primary, primary <= 100)
            run("prior_day_turnover", daily, daily <= 100)
            run("fixed_at_start", primary, pre)
            run("full_period_top100_hindsight", primary, post)
            run("no_four_name_cap", primary, primary <= 100, cap=100)
            run("gross", primary, primary <= 100, buy_fee=0., sell_fee=0.)
            run("capital_4m", primary, primary <= 100, initial=4_000_000)
            run("allow_partial_units", primary, primary <= 100, partial=True)
    pd.DataFrame(results).to_csv(OUT / "summary.csv", index=False)


if __name__ == "__main__":
    main()

