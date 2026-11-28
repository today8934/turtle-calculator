"""Rebuild daily equity from raw closes and the recorded trades, without strategy code."""
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT = Path('output')
reports = []
annual = pd.read_csv(OUTPUT / 'calendar_summary.csv')
for year, variants in annual.groupby('year'):
    quotes = pd.read_parquet(f'market/{year}.parquet')
    if 'Date' not in quotes:
        quotes = quotes.reset_index()
    quotes = quotes[quotes.Market.eq('KOSPI') & quotes.Volume.gt(0)
                    & quotes[['Open','High','Low','Close']].min(axis=1).gt(0)].copy()
    quotes['Date'] = pd.to_datetime(quotes.Date.astype(str), format='mixed')
    quotes['Code'] = quotes.Code.astype(str).str.zfill(6)
    raw = quotes.pivot(index='Date', columns='Code', values='Close').sort_index()
    reference = raw - quotes.pivot(index='Date', columns='Code', values='Changes')
    ratio = raw.ffill().shift().div(reference.where(reference.gt(0))).fillna(1.0)
    for row in variants.itertuples():
        trades = pd.read_csv(OUTPUT / f'{row.label}_fills.csv', dtype={'code':str}, parse_dates=['date'])
        daily = pd.read_csv(OUTPUT / f'{row.label}_daily.csv', parse_dates=['date']).set_index('date')
        codes = sorted(set(trades.code))
        fills = raw.stack().reindex(pd.MultiIndex.from_frame(trades[['date','code']])).to_numpy()
        np.testing.assert_allclose(fills, trades.raw_price, atol=.000001, rtol=0)
        buys = trades.action.eq('buy')
        traded = trades.raw_shares * fills
        costs = traded * np.where(buys, .0005, .0025)
        flows = (-traded.where(buys, -traded) - costs).groupby(trades.date).sum()
        cash = 100_000_000 + flows.reindex(daily.index, fill_value=0).cumsum()
        np.testing.assert_allclose(cash, daily.cash, atol=.01, rtol=0)
        sizes = trades.raw_shares.where(buys, -trades.raw_shares)
        changes = sizes.groupby([trades.date, trades.code]).sum().unstack(fill_value=0)
        changes = changes.reindex(index=daily.index, columns=codes, fill_value=0)
        splits = ratio.reindex(index=daily.index, columns=codes).fillna(1).cumprod()
        holdings = (changes / splits).cumsum() * splits
        assert holdings.min().min() >= -.000001
        marks = raw.reindex(index=daily.index, columns=codes).ffill()
        value = cash + (holdings * marks).sum(axis=1)
        error = (value - daily.equity).abs().max()
        assert error < .01, (row.label, error)
        gains = pd.read_csv(OUTPUT / f'{row.label}_stock_profit.csv')
        reports.append(dict(year=year, top=row.size, fills=len(trades), days=len(daily)-1,
                            worst_error_krw=error, net_pct=(value.iloc[-1] / 1e8 - 1) * 100,
                            fees_pct=costs.sum()/1e6, winning_stocks=int(sum(gains.profit_krw > 0)),
                            traded_stocks=len(codes)))

checks = pd.DataFrame(reports)
checks.to_csv(OUTPUT / 'calendar_audit.csv', index=False)
print('LEDGER_TOTAL', checks.fills.sum(), checks.days.sum(), checks.worst_error_krw.max())
print('LEDGER_MEANS', checks.groupby('top')[['net_pct','fees_pct']].mean().round(4).to_json())
print('LEDGER_EXAMPLE', checks.query('year == 2014 and top == 10').to_json(orient='records'))