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
MERGER_RECALL = {pd.Timestamp("2015-08-26"): {"000830"}}


def short_mask(dates, codes):
    all_ban = (dates >= "2011-08-10") & (dates <= "2011-11-09")
    sector_ban = np.outer(dates < "2013-11-14", pd.Index(codes).isin(PRE_2013_FINANCIALS))
    return ~all_ban[:,None] & ~sector_ban


def verify_fills(label, raw, start, dates, codes, restrict, two_sided, gate=None, flatten=False):
    fills = pd.read_csv(OUT / f"{label}_fills.csv", dtype={"code":str}, parse_dates=["date"])
    daily = pd.read_csv(OUT / f"{label}_daily.csv", parse_dates=["date"])
    assert daily.date.is_monotonic_increasing and daily.equity.iloc[0] == 100_000_000
    if len(fills) == 0:
        return dict(one_price_directional_fills=0, fees_pct=0, completed=0, losing=0, winning=0,
                    closed_same_day=0, avg_win_pct=None, avg_loss_pct=None, forced_exits=0, off_day_exits=0)
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
    risk_order = ordinary.reason.isin(["entry","add"]).to_numpy()
    off_day = (~gate[dates.get_indexer(ordinary.date)] if gate is not None
               else np.zeros(len(ordinary), dtype=bool))
    assert not (risk_order & off_day).any(), label
    forced = ordinary.reason.eq("index_exit").to_numpy()
    assert not forced.any() or (flatten and ordinary.phase.to_numpy()[forced].max()==0), label
    if two_sided:
        shorts = ordinary.loc[ordinary.side.eq(-1) & ordinary.reason.isin(["entry","add"])]
        assert restrict[dates.get_indexer(shorts.date),pd.Index(codes).get_indexer(shorts.code)].all()
        recalled = ordinary[ordinary.reason.eq("merger_recall")]
        if len(recalled):
            assert recalled.date.eq("2015-08-26").all() and recalled.code.eq("000830").all()
            np.testing.assert_allclose(recalled.raw_price,48100,atol=.0001,rtol=0)
    cash_field = "cash_book_after" if two_sided else "cash_after"
    cashflow = (fills.raw_price * fills.raw_shares * np.where(fills.action.eq("buy"), -1.0005, .9975))
    np.testing.assert_allclose(1e8+cashflow.cumsum(), fills[cash_field], atol=.02, rtol=0)
    buying = ordinary.action.eq("buy").to_numpy()
    moved = bars.Changes.to_numpy()
    locked = bars.High.eq(bars.Low).to_numpy() & ((buying & (moved>0)) | (~buying & (moved<0)))
    fee = fills.raw_price * fills.raw_shares * np.where(fills.action.eq("buy"),.0005,.0025)
    assert ordinary.shape[0] == fills.shape[0], "Unresolved delisted holding"
    campaigns = {}
    completed = losing = same_day = 0
    win_returns, loss_returns = [], []
    for record,flow in zip(fills.itertuples(),cashflow):
        if record.reason == "entry":
            assert record.code not in campaigns
            campaigns[record.code] = [0, 0., record.date, 0.]
        c = campaigns[record.code]
        c[0] += 1 if record.reason in ("entry","add") else -1
        c[1] += flow
        if record.reason in ("entry","add"):
            c[3] += record.raw_price * record.raw_shares
        if record.reason == "merger_recall":
            c[0] = 0
        if c[0] == 0:
            completed += 1
            losing += c[1] < 0
            rate = 100*c[1]/c[3]
            if c[1] > 0:
                win_returns.append(rate)
            elif c[1] < 0:
                loss_returns.append(rate)
            same_day += c[2] == record.date
            del campaigns[record.code]
    return dict(one_price_directional_fills=int(locked.sum()), fees_pct=fee.sum()/1e6,
                completed=completed, losing=losing, winning=len(win_returns), closed_same_day=same_day,
                avg_win_pct=np.mean(win_returns) if win_returns else None,
                avg_loss_pct=np.mean(loss_returns) if loss_returns else None,
                forced_exits=int(forced.sum()), off_day_exits=int((~risk_order & off_day).sum()))


def run(path, sma100=False):
    initial, end = pd.Timestamp("2009-12-30"), pd.Timestamp("2019-12-30")
    index, quotes, ranks, names, _ = market_data(range(2008,2020), initial, end)
    if sma100:
        cents = (index*100).round().astype("int64")
        np.testing.assert_allclose(cents/100,index,rtol=0,atol=.00001)
        prior = cents.shift(1)
        trailing_sum = cents.rolling(100,min_periods=100).sum().shift(1)
        gate = prior.mul(100).gt(trailing_sum)
        assert trailing_sum.loc["2010":].notna().all()
        expected = [204,130,164,143,119,133,159,236,47,137]
        assert gate.loc["2010":].groupby(gate.loc["2010":].index.year).sum().tolist()==expected
        if path=="near_first":
            pd.DataFrame({"previous_close":prior/100,"previous_SMA100":trailing_sum/10000,
                          "permit_new_positions":gate}).loc["2010":].to_csv(OUT/"sma100_days.csv",index_label="trade_date")
        prior_results = pd.read_csv("research/results/kospi_system1_intraday_2010_2019.csv").set_index(
            ["year","turnover_top","intraday_path","position_policy"])
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
                for mode in (("original","add_gate","flatten") if sma100 else ("original",)):
                    label = f"{path}_{strategy}_{year}_{size}"
                    this_gate = None
                    if mode!="original":
                        label += f"_{mode}"
                        this_gate = gate.loc[local.index].to_numpy(dtype=bool)
                    rules = dict(market_gate=this_gate, force_flat_on_gate=mode=="flatten")
                    if strategy == "long":
                        row = simulate_path(*args,events[0],label,corr,previous,**rules)
                    else:
                        row = simulate_two_sided(*args,events,label,corr,previous,short_eligible=restrict,
                                                 scheduled_recall=MERGER_RECALL,**rules)
                    row.update(verify_fills(label,raw,previous,local.index,codes,restrict,strategy=="long_short",
                                            this_gate,mode=="flatten"))
                    if not sma100:
                        stats.append(dict(year=year,size=size,path=path,strategy=strategy,kospi=benchmark,**row))
                    elif mode=="original":
                        base = row
                        old = prior_results.loc[year,size,path,strategy]
                        np.testing.assert_allclose([row["ret"],row["mdd"],row["completed"],row["losing"],row["fills"]],
                            old[["intraday_return_pct","intraday_mdd_pct","fully_closed_positions","losing_closed_positions",
                                 "fills"]].to_numpy(dtype=float),rtol=0,atol=.00001)
                    else:
                        stats.append(dict(year=year,size=size,path=path,strategy=strategy,filter=mode,
                            days=len(days),allowed=int(gate.iloc[days].sum()),kospi=benchmark,base_ret=base["ret"],
                            ret=row["ret"],delta=round(row["ret"]-base["ret"],4),base_mdd=base["mdd"],mdd=row["mdd"],
                            base_closed=base["completed"],base_wins=base["winning"],closed=row["completed"],
                            wins=row["winning"],losses=row["losing"],flat=row["completed"]-row["winning"]-row["losing"],
                            avg_win_pct=row["avg_win_pct"],avg_loss_pct=row["avg_loss_pct"],entries=row["entries"],
                            adds=row["adds"],forced_exits=row["forced_exits"],off_day_exits=row["off_day_exits"]))
        print("FINISHED_YEAR",year,path,flush=True)
    result = pd.DataFrame(stats)
    assert len(result)==(120 if sma100 else 60)
    result.to_csv(OUT / f"{'sma100' if sma100 else 'calendar'}_{path}.csv",index=False)
    print("PATH_SUMMARY",path,result.groupby(["strategy","size"]).ret.mean().to_string(),flush=True)


def combine(sma100=False):
    if sma100:
        result = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("sma100_*first.csv"))],ignore_index=True)
        assert len(result)==360 and not result.duplicated(["year","size","path","strategy","filter"]).any()
        result = result.sort_values(["year","path","filter","strategy","size"]).round(6)
        result.to_csv(OUT/"sma100_results.csv",index=False)
        import json
        print("SMA_FIELDS",",".join(result.columns))
        for row in result.to_numpy().tolist():
            print("SMA_ROW",json.dumps([None if pd.isna(x) else x for x in row],separators=(",",":")))
        return
    results = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("calendar_*first.csv"))],ignore_index=True)
    assert len(results)==180 and not results.duplicated(["year","size","path","strategy"]).any()
    assert results.groupby("year").kospi.nunique().eq(1).all()
    results.to_csv(OUT/"intraday_comparison.csv",index=False)
    for (strategy,path), g in results.groupby(["strategy","path"]):
        keys = ["year","size","ret","mdd","kospi","fills","delist","one_price_directional_fills",
                "fees_pct","completed","losing","closed_same_day","merger_recalls"]
        print("SCENARIO_DATA",strategy,path,g[keys].round(4).to_json(orient="values"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("choice",choices=[*PATHS,"combine"])
    parser.add_argument("--sma100",action="store_true")
    args = parser.parse_args()
    if args.choice == "combine":
        combine(args.sma100)
    else:
        run(args.choice,args.sma100)
