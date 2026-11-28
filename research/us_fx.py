"""USD/KRW and S&P 500 price-index endpoints for independently funded annual accounts."""

import json
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ECB = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD+KRW.EUR.SP00.A"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC"

# Checked against https://fred.stlouisfed.org/data/DEXKOUS and SP500,
# https://sp500.net/sp500_yearly_returns, and the Nov. 27 ^GSPC historical close.
# The November 27 provisional cross uses ECB KRW per EUR divided by USD per EUR:
# https://www.ecb.europa.eu/stats/shared/pdf/eurofxref.pdf (2026-11-27 release).
# US equities traded Dec 31, 2010/2021, but holiday FX was fixed Dec 30.
ARCHIVE = {
    "2009-12-31": (1115.10, 1163.65),
    "2010-12-31": (1257.64, 1130.60),
    "2011-12-30": (1257.60, 1158.50),
    "2012-12-31": (1426.19, 1063.24),
    "2013-12-31": (1848.36, 1055.25),
    "2014-12-31": (2058.90, 1090.89),
    "2015-12-31": (2043.94, 1169.26),
    "2016-12-30": (2238.83, 1203.73),
    "2017-12-29": (2673.61, 1067.42),
    "2018-12-31": (2506.85, 1112.85),
    "2019-12-31": (3230.78, 1155.46),
    "2020-12-31": (3756.07, 1086.11),
    "2021-12-31": (4766.18, 1188.59),
    "2022-12-30": (3839.50, 1260.18),
    "2023-12-29": (4769.83, 1290.97),
    "2024-12-31": (5881.63, 1477.86),
    "2025-12-31": (6845.50, 1444.55),
    "2026-11-27": (7297.49, 1462.65 / 1.0957),
}
PREVIOUS_FIXING = {"2010-12-31": "2010-12-30", "2021-12-31": "2021-12-30"}


def _fetch(url, **query):
    request = Request(url + "?" + urlencode(query), headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=18) as response:
        return response.read()


def _fred(series, start, end):
    frame = pd.read_csv(BytesIO(_fetch(FRED, id=series, cosd=start, coed=end)))
    values = pd.to_numeric(frame[series], errors="coerce").to_numpy()
    return pd.Series(values, index=pd.DatetimeIndex(frame.iloc[:, 0])).dropna().sort_index()


def _ecb(start, end):
    frame = pd.read_csv(BytesIO(_fetch(ECB, startPeriod=start, endPeriod=end, format="csvdata")))
    dates = frame.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE")
    rates = dates["KRW"] / dates["USD"]
    rates.index = pd.to_datetime(rates.index)
    return rates.dropna().sort_index()


def _yahoo(day):
    begin = day.tz_localize("America/New_York")
    stop = begin + pd.Timedelta(days=1)
    data = json.loads(_fetch(YAHOO, period1=int(begin.timestamp()), period2=int(stop.timestamp()), interval="1d"))
    block = data["chart"]["result"][0]
    for epoch, price in zip(block["timestamp"], block["indicators"]["quote"][0]["close"]):
        local = pd.Timestamp(epoch, unit="s", tz="UTC").tz_convert("America/New_York")
        if local.date() == day.date() and price is not None and price > 0:
            return round(price, 2)
    raise ValueError(f"Missing S&P 500 session {day.date()}")


def _observation(day):
    """No future FX or stale index fallback. Archived values preserve the stated data vintage."""
    date = day.date().isoformat()
    if date in ARCHIVE:
        spx, fx = ARCHIVE[date]
        source = "ECB EUR cross" if date == "2026-11-27" else "FRED DEXKOUS"
        if date == "2026-11-27":
            spx_source = "S&P 500 via Yahoo / FT"
        elif day.year >= 2016:
            spx_source = "S&P DJI via FRED SP500"
        else:
            spx_source = "S&P 500 historical year-end close"
        return spx, fx, PREVIOUS_FIXING.get(date, date), source, spx_source

    week = (day - pd.Timedelta(days=7)).date().isoformat()
    try:
        spx = _fred("SP500", date, date).at[day]
        spx_source = "S&P DJI via FRED SP500"
    except (OSError, ValueError, IndexError, KeyError):
        spx = _yahoo(day)
        spx_source = "S&P 500 via Yahoo"
    try:
        fx = _fred("DEXKOUS", week, date)
    except (OSError, ValueError, IndexError, KeyError):
        fx = pd.Series(dtype=float)
    if day in fx.index:
        return spx, fx.at[day], date, "FRED DEXKOUS", spx_source
    try:
        cross = _ecb(week, date)
    except (OSError, ValueError, IndexError, KeyError):
        cross = pd.Series(dtype=float)
    # If one source is unpublished, use the latest available observation as of date.
    options = [(rates.index[-1], priority, rates.iloc[-1], label)
               for priority, (rates, label) in enumerate([(cross, "ECB EUR cross"), (fx, "FRED DEXKOUS")])
               if len(rates) and rates.index[-1] <= day and (day-rates.index[-1]).days <= 4]
    if not options:
        raise ValueError(f"No verified USD/KRW within four calendar days of {date}")
    fixing, _, value, label = max(options)
    return spx, value, fixing.date().isoformat(), label, spx_source


def year_endpoints(calendar, asof=None, first_year=2010):
    """Return non-indexed table of period endpoints, underlying levels, and percent returns.

    Supply the actual US sessions in the simulated price panel, including the
    preceding December; asof is inclusive. The same FX converts both the strategy
    and price-only benchmark. Using the fixed archive avoids vintage drift.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(calendar)).normalize().unique().sort_values()
    if asof is not None:
        dates = dates[dates <= pd.Timestamp(asof)]
    if dates.empty:
        raise ValueError("Empty trading calendar")
    endpoints = {y: dates[dates.year == y][-1] for y in dates.year.unique()}
    observations = {}
    rows = []
    for year, end in endpoints.items():
        if year < first_year:
            continue
        start = endpoints.get(year-1)
        if start is None or start.month != 12:
            raise ValueError(f"{year} needs the previous December's last US trading session")
        for date in [start, end]:
            if date not in observations:
                observations[date] = _observation(date)
        p0, f0, d0, s0, i0 = observations[start]
        p1, f1, d1, s1, i1 = observations[end]
        rows.append(dict(year=int(year), start=start.date().isoformat(), end=end.date().isoformat(),
                         fx_start=f0, fx_end=f1, fx_start_date=d0, fx_end_date=d1,
                         fx_start_source=s0, fx_end_source=s1, spx_start=p0, spx_end=p1,
                         spx_start_source=i0, spx_end_source=i1,
                         spx_usd_pct=(p1/p0-1)*100, spx_krw_pct=(p1*f1/(p0*f0)-1)*100))
    return pd.DataFrame(rows)
