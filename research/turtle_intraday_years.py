"""Calendar-year Turtle S1 on KOSPI shares, without inverse instruments."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kospi_system1 import market_data, panels, correlation_limits
from turtle_path_account import simulate_path
from turtle_shadow_path import shadow_events
from turtle_two_sided import simulate_two_sided


SIZES = (10, 20, 30)
PATHS = {"high_first": ("o","h","l","c"), "low_first": ("o","l","h","c"),
         "near_first": ("o","first","second","c")}
PRE_2013_FINANCIALS = {
    "000810", "004940", "006800", "010050", "016360", "032830", "053000", "055550",
    "086790", "088350", "105560",
}
OUT = Path("output")


def short_mask(dates, codes):
    all_ban = (dates >= "2011-08-10") & (dates <= "2011-11-09")
    sector_ban = np.outer(dates < "2013-11-14", pd.Index(codes).isin(PRE_2013_FINANCIALS))
    return ~all_ban[:,None] & ~sector_ban


def verify_fills(label, raw, start, dates, codes, restrict, two_sided):
    fills = pd.read_csv(OUT / f"{label}_fills.csv", dtype={"code":str}, parse_dates=["date"])
    daily = pd.read_csv(OUT / f"{label}_daily.csv", parse_dates=["date"])
    assert daily.date.is_monotonic_increasing and daily.equity.iloc[0] == 100_000_000
    if len(fills) == 0:
        return 0
    assert fills.date.gt(start).all() and fills.raw_shares.gt(0).all()
    order = pd.MultiIndex.from_frame(fills[["date","phase","fraction"]])
    assert order.is_monotonic_increasing, label
    ordinary = fills.loc[fills.phase.ge(0)]
    bars = raw.reindex(pd.MultiIndex.from_frame(ordinary[["date","code"]]))
    price = ordinary.raw_price.to_numpy()
    assert np.all(price >= bars.Low.to_numpy() - .0001)
    assert np.all(price <= bars.High.to_numpy() + .0001)
    opening = ordinary.phase.eq(0).to_numpy()
    np.testing.assert_allclose(price[opening], bars.Open.to_numpy()[opening], rtol=0, atol=.0001)
    if two_sided:
        shorts = ordinary.loc[ordinary.side.eq(-1) & ordinary.reason.isin(["entry","add"])]
        assert restrict[dates.get_indexer(shorts.date),pd.Index(codes).get_indexer(shorts.code)].all()
    cash_field = "cash_book_after" if two_sided else "cash_after"
    cashflow = (fills.raw_price * fills.raw_shares * np.where(fills.action.eq("buy"), -1.0005, .9975))
    np.testing.assert_allclose(1e8+cashflow.cumsum(), fills[cash_field], atol=.02, rtol=0)
    buying = ordinary.action.eq("buy").to_numpy()
    moved = bars.Changes.to_numpy()
    locked = bars.High.eq(bars.Low).to_numpy() & ((buying & (moved>0)) | (~buying & (moved<0)))
    return int(locked.sum())


def run(path):
    initial, end = pd.Timestamp("2009-12-30"), pd.Timestamp("2019-12-30")
    index, quotes, ranks, names, _ = market_data(range(2008,2020), initial, end)
    include = ranks.loc["2010":].le(max(SIZES)).any()
    order = include[include].index.to_list()
    assert not set(order).intersection({"114800","251340"})
    used = quotes.loc[quotes.Code.isin(order)]
    raw = used.set_index(["Date","Code"])[["Open","High","Low","Close","Changes"]]
    m, last, _ = panels(index, used, order)
    if path == "near_first":
        high_near = m["h"] - m["o"] < m["o"] - m["l"]
        m["first"] = np.where(high_near, m["h"], m["l"])
        m["second"] = np.where(high_near, m["l"], m["h"])
    phases = PATHS[path]
    long_events, short_events = shadow_events(m, index.index, phases, return_short=True)
    stats = []
    for year in range(2010,2020):
        days = np.flatnonzero(index.index.year == year)
        previous = index.index[days[0]-1]
        begin, stop = days[0]-65, days[-1]+1
        local = index.iloc[begin:stop]
        annual_ranks = ranks.iloc[days].loc[:,order]
        cols = np.flatnonzero(annual_ranks.le(max(SIZES)).any().to_numpy())
        codes = [order[k] for k in cols]
        values = {key: arr[begin:stop,cols] for key,arr in m.items()}
        events = (long_events[begin:stop][:,:,cols],short_events[begin:stop][:,:,cols])
        priority, corr = correlation_limits(local, values, start_date=previous)
        rank = ranks.iloc[begin:stop].loc[:,codes]
        restrict = short_mask(local.index, codes)
        benchmark = (local.iloc[-1]/local.at[previous]-1)*100
        for size in SIZES:
            eligible = rank.le(size).to_numpy()
            assert (eligible[local.index>previous].sum(axis=1)==size).all()
            args = local, values, last[cols]-begin, codes, names, priority, eligible, phases
            for strategy in ("long", "long_short"):
                label = f"{path}_{strategy}_{year}_{size}"
                if strategy == "long":
                    row = simulate_path(*args,events[0],label,corr,previous)
                else:
                    row = simulate_two_sided(*args,events,label,corr,previous,short_eligible=restrict)
                row["one_price_directional_fills"] = verify_fills(label,raw,previous,local.index,codes,restrict,strategy=="long_short")
                stats.append(dict(year=year,size=size,path=path,strategy=strategy,kospi=benchmark,**row))
        print("FINISHED_YEAR",year,path,flush=True)
    result = pd.DataFrame(stats)
    assert len(result)==60
    result.to_csv(OUT / f"calendar_{path}.csv",index=False)
    print("PATH_SUMMARY",path,result.groupby(["strategy","size"]).ret.mean().to_string(),flush=True)


def combine():
    results = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("calendar_*first.csv"))],ignore_index=True)
    assert len(results)==180 and not results.duplicated(["year","size","path","strategy"]).any()
    assert results.groupby("year").kospi.nunique().eq(1).all()
    results.to_csv(OUT/"intraday_comparison.csv",index=False)
    for (strategy,path), g in results.groupby(["strategy","path"]):
        print("SCENARIO_DATA",strategy,path,g[["year","size","ret","mdd","kospi","fills","delist","one_price_directional_fills"]].round(4).to_json(orient="values"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice",choices=[*PATHS,"combine"])
    choice = parser.parse_args().choice
    if choice == "combine":
        combine()
    else:
        run(choice)
