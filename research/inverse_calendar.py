"""Close-only stock portfolios with two cash-purchased inverse ETF trading candidates."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from kospi_system1 import market_data, panels, close_signals, correlation_limits, simulate


OUTPUT = Path("output")
YEARS = range(2010, 2020)
SIZES = (10, 20, 30)
FUNDS = {
    "114800": ("KODEX Inverse", "2009-09-16", 9930, 6350),
    "251340": ("KODEX KOSDAQ150 Futures Inverse", "2016-08-10", 10000, 7610),
}


def load_fund(code, calendar):
    name, first, first_open, last_close = FUNDS[code]
    frame = pd.read_csv(f"market/{code}_D.csv", dtype={"date": str})
    frame = frame.rename(columns=str.title)
    frame["Date"] = pd.to_datetime(frame.Date, format="%Y%m%d")
    frame = frame.loc[frame.Date.le(calendar[-1])].copy().sort_values("Date")
    expected = calendar[calendar >= first]
    assert not frame.Date.duplicated().any()
    assert pd.DatetimeIndex(frame.Date).equals(expected), (code, expected.difference(frame.Date))
    ohlc = frame[["Open", "High", "Low", "Close"]]
    assert ohlc.mod(5).eq(0).all().all()
    assert frame.Volume.gt(0).all() and ohlc.gt(0).all().all()
    assert frame.High.ge(ohlc.max(axis=1)).all() and frame.Low.le(ohlc.min(axis=1)).all()
    assert frame.Open.iloc[0] == first_open and frame.Close.iloc[-1] == last_close
    frame["Name"], frame["Code"] = name, code
    frame["Changes"] = frame.Close.diff().fillna(0)
    return frame[["Code", "Name", "Date", "Open", "High", "Low", "Close", "Volume", "Changes"]]


def main():
    initial_date = pd.Timestamp("2009-12-30")
    end = pd.Timestamp("2019-12-30")
    index, stocks, ranks, names, _ = market_data(range(2008, 2020), initial_date, end)
    funds = pd.concat([load_fund(code, index.index) for code in FUNDS])
    assert not stocks.Code.isin(FUNDS).any()
    inside = ranks.loc["2010":].le(max(SIZES)).any()
    stock_codes = ranks.columns[inside].tolist()
    order = stock_codes + list(FUNDS)
    names.update({code: item[0] for code, item in FUNDS.items()})
    quotes = pd.concat([stocks.loc[stocks.Code.isin(stock_codes)], funds], ignore_index=True)
    matrices, last, _ = panels(index, quotes, order, etf_codes=list(FUNDS))
    signals = close_signals(matrices, index.index)
    raw_closes = quotes.set_index(["Date", "Code"]).Close
    summaries = []
    for year in YEARS:
        year_days = np.flatnonzero(index.index.year == year)
        start, end = year_days[0], year_days[-1] + 1
        previous = index.index[start-1]
        begin = start - 65
        local = index.iloc[begin:end]
        annual_rank = ranks.iloc[start:end][stock_codes]
        selected = np.flatnonzero(annual_rank.le(max(SIZES)).any().to_numpy()).tolist()
        selected.extend(range(len(stock_codes), len(order)))
        ids = [order[k] for k in selected]
        prices = {key: arr[begin:end,selected] for key, arr in matrices.items()}
        events = signals[begin:end][:,:,selected]
        priority, correlation = correlation_limits(local, prices, start_date=previous)
        annual_funds = pd.Index(ids).isin(FUNDS)
        tradable = np.isfinite(prices["c"]) & np.isfinite(prices["n"]) & (prices["n"] > 0)
        stock_ranks = ranks.iloc[begin:end].reindex(columns=ids)
        for size in SIZES:
            stock_eligible = stock_ranks.le(size).to_numpy()
            assert (stock_eligible[local.index > previous].sum(axis=1) == size).all()
            outcomes = []
            for mode in ("stocks", "with_inverse"):
                eligible = stock_eligible.copy()
                if mode == "with_inverse":
                    eligible[:,annual_funds] = tradable[:,annual_funds]
                label = f"{mode}_{year}_top{size}"
                outcome = simulate(local, prices, last[selected]-begin, ids, names, priority,
                                   eligible, ("c","c"), events, label, correlation,
                                   closing_auction=True, start_date=previous)
                ledger = pd.read_csv(OUTPUT / f"{label}_fills.csv", dtype={"code":str}, parse_dates=["date"])
                keys = pd.MultiIndex.from_frame(ledger[["date","code"]])
                np.testing.assert_allclose(ledger.raw_price, raw_closes.reindex(keys), atol=1e-6, rtol=0)
                assert outcome["delist"] == 0
                by_code = pd.read_csv(OUTPUT / f"{label}_stock_profit.csv", dtype={"code":str}).set_index("code").profit_krw
                outcomes.append(outcome)
            fund_entries = ledger.loc[ledger.code.isin(FUNDS) & ledger.reason.eq("entry")]
            fund_profits = by_code.reindex(list(FUNDS), fill_value=0)
            baseline, combined = outcomes
            summaries.append(dict(year=year, size=size, baseline=baseline["ret"], inverse=combined["ret"],
                                  delta=round(combined["ret"]-baseline["ret"],4), kospi=100*(local.iloc[-1]/local.at[previous]-1),
                                  original_mdd=baseline["mdd"], inverse_mdd=combined["mdd"],
                                  kospi_etf_entries=int(fund_entries.code.eq("114800").sum()),
                                  kosdaq_etf_entries=int(fund_entries.code.eq("251340").sum()),
                                  kospi_etf_pct=fund_profits["114800"]/1e6,
                                  kosdaq_etf_pct=fund_profits["251340"]/1e6))
    table = pd.DataFrame(summaries)
    baseline = pd.read_csv(OUTPUT / "calendar_summary.csv").set_index(["year","size"]).sort_index()
    check = table.set_index(["year","size"]).sort_index()
    np.testing.assert_allclose(check.baseline, baseline.ret, atol=.00001, rtol=0)
    np.testing.assert_allclose(check.original_mdd, baseline.mdd, atol=.00001, rtol=0)
    table.to_csv(OUTPUT / "inverse_comparison.csv", index=False)
    print("INVERSE_RESULTS", table.round(5).to_json(orient="records"), flush=True)


if __name__ == "__main__":
    main()
