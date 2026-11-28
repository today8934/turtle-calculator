"""Compare close-only cash long stock trades with and without the KOSPI SMA100 gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kospi_system1 import CAPITAL, close_signals, correlation_limits, market_data, panels, simulate


OUT = Path("output")
SIZES = (10, 20, 30)


def payoff_and_audit(label, quotes, gate, mode, expected):
    trades = pd.read_csv(OUT/f"{label}_fills.csv", dtype={"code":str}, parse_dates=["date"])
    curve = pd.read_csv(OUT/f"{label}_daily.csv").set_index("date")
    assert curve.equity.iloc[0] == CAPITAL and abs(curve.equity.iloc[-1]-expected["final"]) <= .51
    assert not trades.reason.eq("delisting_zero_recovery").any(), (label, "unsettled corporate action")
    print_dates = pd.MultiIndex.from_frame(trades[["date","code"]])
    np.testing.assert_allclose(trades.raw_price, quotes.reindex(print_dates), rtol=0, atol=.00001)
    buys = trades.action.eq("buy").to_numpy()
    if gate is not None and mode!="original":
        assert gate.reindex(trades.date[buys]).all()
    assert trades.loc[buys,"phase"].eq(1).all() and trades.loc[~buys,"phase"].eq(0).all()
    assert not trades.loc[buys].duplicated(["code","date"]).any()
    assert mode=="flatten" or not trades.reason.eq("index_exit").any()
    principal = trades.raw_price * trades.raw_shares
    flow = principal * np.where(buys, -1.0005, .9975)
    np.testing.assert_allclose(CAPITAL+flow.cumsum(),trades.cash_after,rtol=0,atol=.02)
    live, finished = {},[]
    for f,amount,cash_flow in zip(trades.itertuples(),principal,flow):
        if f.reason=="entry":
            assert f.code not in live
            live[f.code] = [0,0.,0.]
        p = live[f.code]
        buying = f.action=="buy"
        p[0] += 1 if buying else -1
        p[1] += amount if buying else 0
        p[2] += cash_flow
        assert 0<=p[0]<=4
        if p[0]==0:
            finished.append((p[2],100*p[2]/p[1]))
            del live[f.code]
    assert len(live)==expected["held"]
    open_profit = curve.equity.iloc[-1]-curve.cash.iloc[-1]+sum(p[2] for p in live.values())
    assert abs(CAPITAL + sum(v for v,_ in finished)+open_profit-curve.equity.iloc[-1])<.02
    wins = [r for v,r in finished if v>.00001]
    losses = [r for v,r in finished if v<-.00001]
    return dict(closed=len(finished),wins=len(wins),losses=len(losses),win_rate=(100*len(wins)/len(finished) if finished else None),
                avg_win_pct=np.mean(wins) if wins else None,avg_loss_pct=np.mean(losses) if losses else None,
                index_exit_fills=int(trades.reason.eq("index_exit").sum()))


def stock_filter(index, universe, matrices):
    close = pd.DataFrame(matrices["c"],index=index.index,columns=universe)
    avg = pd.DataFrame(np.nan,index=index.index,columns=universe)
    threshold = avg.copy()
    for code in universe:
        valid = close[code].dropna()
        avg.loc[valid.index,code] = valid.rolling(100,min_periods=100).mean()
        threshold.loc[valid.index,code] = valid.shift().rolling(99,min_periods=99).mean()
    # close[t] > SMA100[t] iff close[t] exceeds the mean of the previous 99 closes.
    # Guard equality against floating-point error in corporate-action-adjusted prices.
    tol = 1e-9 * pd.DataFrame(matrices["factor"],index=index.index,columns=universe)
    eligible = (close-threshold).gt(tol)
    np.testing.assert_array_equal(eligible, (close-avg).gt(tol*0.99))
    return eligible,avg


def main(stock=False):
    index, market, ranks, names, _ = market_data(range(2008,2020),pd.Timestamp("2009-12-30"),pd.Timestamp("2019-12-30"))
    cents = (100*index).round().astype("int64")
    np.testing.assert_allclose(cents/100,index,atol=.00001,rtol=0)
    prior = cents.shift(1)
    total = cents.rolling(100,min_periods=100).sum().shift(1)
    gate = prior.mul(100).gt(total)
    assert total.loc["2010":].notna().all()
    by_year = gate.loc["2010":].groupby(gate.loc["2010":].index.year).sum()
    assert by_year.tolist()==[204,130,164,143,119,133,159,236,47,137]
    full = ranks.loc["2010":].le(max(SIZES)).any()
    universe = full[full].index.tolist()
    assert not set(universe).intersection({"114800","251340"})
    quotes = market[market.Code.isin(universe)]
    raw = quotes.set_index(["Date","Code"]).Close
    m,last,_ = panels(index,quotes,universe)
    if stock:
        name_eligible,stock_sma = stock_filter(index,universe,m)
        avg_lookup = stock_sma.stack()
        adj_lookup = pd.DataFrame(m["c"],index=index.index,columns=universe).stack()
        allowed_lookup = name_eligible.stack()
        prior_market = pd.read_csv("research/results/kospi_close_sma100_2010_2019.csv").set_index(["year","top","mode"])
    shadow = close_signals(m,index.index)
    old = pd.read_csv("reference/calendar_summary.csv").set_index(["year","size"])
    stocks_only = pd.read_csv("research/results/kospi_system1_intraday_2010_2019.csv").groupby(["year","turnover_top"]).prior_close_long_return_pct.first()
    rows = []
    for year in range(2010,2020):
        days = np.flatnonzero(index.index.year==year)
        start = index.index[days[0]-1]
        begin,end = days[0]-65,days[-1]+1
        local = index.iloc[begin:end]
        rank = ranks.iloc[begin:end].loc[:,universe]
        cols = np.flatnonzero(rank.loc[local.index>start].le(max(SIZES)).any().to_numpy())
        codes = [universe[k] for k in cols]
        values = {key: v[begin:end,cols] for key,v in m.items()}
        events = shadow[begin:end][:,:,cols]
        priority,corr = correlation_limits(local,values,start_date=start)
        can_trade = gate.loc[local.index].to_numpy(dtype=bool)
        benchmark = (local.iloc[-1]/local.at[start]-1)*100
        for size in SIZES:
            eligible = rank.loc[:,codes].le(size).to_numpy()
            assert (eligible[local.index>start].sum(axis=1)==size).all()
            if stock:
                tradable = name_eligible.loc[local.index,codes].to_numpy()
                candidate = np.isfinite(events[:,1,:]) & eligible & (local.index>start)[:,None]
                has_history = stock_sma.loc[local.index,codes].notna().to_numpy()
                signal_days = int(candidate.sum())
                passed = int((candidate & tradable).sum())
                insufficient_history = int((candidate & ~has_history).sum())
            modes = ("original","stock_entry","stock_entry_add") if stock else ("original","entry_gate","flatten")
            for mode in modes:
                label = f"close_{mode}_{year}_top{size}"
                row = simulate(local,values,last[cols]-begin,codes,names,priority,eligible,("c","c"),
                               events,label,corr,closing_auction=True,start_date=start,
                               market_gate=can_trade if not stock and mode!="original" else None,
                               force_flat_on_gate=mode=="flatten",
                               stock_entry_gate=tradable if stock and mode!="original" else None,
                               gate_adds=mode!="stock_entry")
                metrics = payoff_and_audit(label,raw,None if stock else gate,mode,row)
                if stock and mode!="original":
                    fills = pd.read_csv(OUT/f"{label}_fills.csv",dtype={"code":str},parse_dates=["date"])
                    checked = fills.loc[fills.reason.isin(["entry","add"] if mode=="stock_entry_add" else ["entry"])]
                    keys = pd.MultiIndex.from_frame(checked[["date","code"]])
                    assert allowed_lookup.reindex(keys).all()
                    assert (adj_lookup.reindex(keys).to_numpy()>avg_lookup.reindex(keys).to_numpy()).all()
                    if mode=="stock_entry":
                        keys = pd.MultiIndex.from_frame(fills.loc[fills.reason.eq("add"),["date","code"]])
                        entry_only_adds_below = int((~allowed_lookup.reindex(keys)).sum())
                if mode=="original":
                    baseline = row["ret"]
                    keys = ["ret","mdd","final","entries","adds","exits","traded","held"]
                    np.testing.assert_allclose([row[k] for k in keys],old.loc[(year,size),keys].astype(float),rtol=0,atol=.0001)
                    assert abs(stocks_only.at[(year,size)]-baseline)<.0001
                data = dict(year=year,top=size,mode=mode,eligible_days=int(by_year.at[year]),sessions=len(days),
                            kospi_pct=benchmark,return_pct=row["ret"],improvement_pp=row["ret"]-baseline,
                            mdd_pct=row["mdd"],entries=row["entries"],adds=row["adds"],exposure_pct=row["exposure"],**metrics)
                if stock:
                    del data["eligible_days"]
                    data.update(signal_candidates=signal_days,signals_passing_stock_sma=passed,
                                signals_without_sma=insufficient_history,
                                index_gate_return_pct=prior_market.at[(year,size,"entry_gate"),"return_pct"],
                                adds_below_sma_if_ungated=entry_only_adds_below if mode=="stock_entry" else None)
                rows.append(data)
    out = pd.DataFrame(rows).round(6)
    assert len(out)==90 and not out.duplicated(["year","top","mode"]).any()
    out.to_csv(OUT/("calendar_close_stock_sma100.csv" if stock else "calendar_close_sma100.csv"),index=False)
    print("CLOSE_FILTER_FIELDS",",".join(out.columns))
    for row in out.to_numpy().tolist():
        print("CLOSE_FILTER_ROW",json.dumps([None if pd.isna(x) else x for x in row],separators=(",",":")))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--stock-sma100",action="store_true")
    main(args.parse_args().stock_sma100)
