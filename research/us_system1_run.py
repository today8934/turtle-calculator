"""Stitch early and modern split-only bars, rank point in time, run annual accounts."""
from pathlib import Path
import argparse
import json
import duckdb
import exchange_calendars as xc
import numpy as np
import pandas as pd

from us_turtle_close import prepare, simulate_year
from us_fx import year_endpoints


def assemble(dolt, early, dates, cache):
    db = duckdb.connect(str(cache / "us_rank.duckdb"))
    db.execute("SET memory_limit='4GB'")
    db.execute("SET threads=3")
    db.register("calendar_frame", pd.DataFrame({"date": dates, "i": range(len(dates))}))
    db.execute("CREATE OR REPLACE TEMP VIEW cal AS SELECT date::DATE AS date,i FROM calendar_frame")
    for name, path in (("newer", dolt), ("older", early)):
        db.read_parquet(str(path)).create_view(name)
    # Both vendors supply split-only OHLC. Align the old adjusted basis across the
    # boundary so historical channel/ATR levels remain meaningful. Raw cash is unchanged.
    db.execute("""
        CREATE OR REPLACE TEMP TABLE bridge AS
        WITH pairs AS (
            SELECT o.symbol,o.date,n.close/o.close AS ratio,
                   n.raw_close/o.raw_close AS raw_ratio,
                   row_number() OVER(PARTITION BY o.symbol ORDER BY o.date) AS seq
            FROM older o JOIN newer n ON n.symbol=o.symbol AND n.date=o.date
            WHERE o.close>0 AND n.close>0 AND o.date >= DATE '2011-01-01'
        )
        SELECT symbol, median(ratio) AS ratio, median(raw_ratio) AS raw_ratio
        FROM pairs WHERE seq<=20 GROUP BY symbol
    """)
    print("BRIDGE", db.execute("SELECT count(*), count(*) FILTER (WHERE abs(raw_ratio-1)>.05) FROM bridge").fetchone(),flush=True)
    # A reused ticker or distant predecessor across the boundary is a separate security.
    # Only connect both histories when actual dollar prices corroborate the match.
    db.execute("""
       CREATE OR REPLACE TEMP VIEW all_quotes AS
       SELECT n.date::DATE AS date,n.symbol,n.high,n.low,n.close,n.raw_close,n.turnover,
              0.0::DOUBLE AS dividend,n.eligible FROM newer n WHERE n.date>=DATE '2011-01-01'
       UNION ALL
       SELECT o.date::DATE AS date,
              CASE WHEN b.raw_ratio BETWEEN .95 AND 1.05 THEN o.symbol ELSE o.symbol||'@WIKI' END AS symbol,
              o.high*coalesce(b.ratio,1),o.low*coalesce(b.ratio,1),o.close*coalesce(b.ratio,1),
              o.raw_close,o.turnover,0.0::DOUBLE,o.eligible
       FROM older o LEFT JOIN bridge b USING(symbol) WHERE o.date<DATE '2011-01-01'
    """)
    # Assign one exact exchange-session number to each day; require all 20 prior
    # sessions and no missing turnover. Presence of a security on today's session is
    # needed to fill at the close. Requiring a tradable bar does not reorder yesterday's ranks.
    db.execute("""
        CREATE OR REPLACE TABLE base AS
        SELECT c.i,q.* FROM all_quotes q JOIN cal c USING(date)
        WHERE q.high>0 AND q.low>0 AND q.close>0 AND q.raw_close>0 AND q.turnover>0 AND q.high>=q.close AND q.low<=q.close AND q.eligible
    """)
    dup = db.execute("SELECT symbol,date FROM base GROUP BY ALL HAVING count(*)>1 LIMIT 1").fetchall()
    if dup:
        raise ValueError(f"Duplicate bars {dup}")
    db.execute("""
        CREATE OR REPLACE TEMP TABLE winners AS
        WITH t AS (
            SELECT symbol,i,avg(turnover) OVER w AS amt,count(turnover) OVER w AS cnt,
                   min(i) OVER w AS first_i
            FROM base WINDOW w AS (PARTITION BY symbol ORDER BY i ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        ), ranked AS (
            SELECT symbol,i+1 AS i, row_number() OVER(PARTITION BY i ORDER BY amt DESC,symbol) AS rank
            FROM t WHERE cnt=20 AND first_i=i-19 AND amt>0
        )
        SELECT * FROM ranked WHERE rank<=30
    """)
    checks = db.execute("""
        SELECT year(b.date) AS year,count(DISTINCT b.symbol) AS universe,
               count(DISTINCT b.date) AS sessions,count(*) AS obs
        FROM base b GROUP BY ALL ORDER BY year
    """).fetchdf()
    print("COVERAGE",checks.to_json(orient="records"),flush=True)
    bad=db.execute("""
        SELECT w.symbol,count(*) AS n FROM winners w JOIN base b USING(symbol,i)
        WHERE b.high < b.close-0.000001 OR b.low>b.close+0.000001 GROUP BY 1 LIMIT 8
    """).fetchall()
    if bad:
        print("RANGE_QUALITY",bad,flush=True)
    frame=db.execute("""
        SELECT b.date,b.symbol,b.high,b.low,b.close,b.raw_close,b.turnover,b.dividend,w.rank
        FROM base b JOIN (SELECT DISTINCT symbol FROM winners) s USING(symbol)
        LEFT JOIN winners w ON b.symbol=w.symbol AND b.i=w.i ORDER BY b.symbol,b.date
    """).fetchdf()
    print("RANKED", len(frame), frame.symbol.nunique(), frame.date.min(),frame.date.max(),flush=True)
    frame["move"]=frame.groupby("symbol",sort=False).close.pct_change(fill_method=None)
    extremes=frame.loc[frame["rank"].le(30) & frame.move.abs().gt(.50),["date","symbol","rank","move"]].sort_values("move")
    print("EXTREMES",extremes.to_json(orient="records",date_format="iso"),flush=True)
    extremes.to_csv(cache/"ranked-price-outliers.csv",index=False)
    db.close()
    return frame,checks


def run(args):
    out=Path(args.out)
    out.mkdir(exist_ok=True,parents=True)
    end=pd.Timestamp(args.end)
    dates=xc.get_calendar("XNYS",start="2008-01-01",end=end).sessions.tz_localize(None)
    frame,checks=assemble(Path(args.dolt),Path(args.early),dates,out)
    if frame.date.max()!=end:
        raise ValueError(f"Missing end date: {frame.date.max()}")
    fx=year_endpoints(dates,asof=end,first_year=2010).set_index("year")
    panels=prepare(frame,dates)
    sums, curves, fills=[],[],[]
    for year in range(2010,end.year+1):
        y=fx.loc[year]
        group=[]
        for top in [10,20,30]:
            s,d,f=simulate_year(panels, year,top,100_000_000/y.fx_start,terminal_policy="last_mark")
            s.update(krw_final=s["final"]*y.fx_end,krw_pct=(s["final"]*y.fx_end/100_000_000-1)*100,
                     fx_start=y.fx_start,fx_end=y.fx_end,spx_usd_pct=y.spx_usd_pct,spx_krw_pct=y.spx_krw_pct)
            sums.append(s)
            d["year"],d["top"]=year,top
            f["year"],f["top"]=year,top
            curves.append(d)
            fills.append(f)
            group.append({k:round(float(s[k]),5) for k in
                ("year","top","ret","krw_pct","spx_krw_pct","spx_usd_pct","long_entries","short_entries","mdd","estimated_terminal")})
        print("ANNUAL",json.dumps(group),flush=True)
    summary=pd.DataFrame(sums)
    assert len(summary)==51 and (summary.sessions>=220).all()
    assert (summary.long_entries>0).all() and (summary.short_entries>0).all()
    checks.to_csv(out/"coverage.csv",index=False)
    fx.to_csv(out/"fx-benchmark.csv")
    summary.to_csv(out/"annual.csv",index=False)
    pd.concat(curves).to_csv(out/"daily.csv.gz",index=False)
    traded=pd.concat(fills)
    traded.to_csv(out/"fills.csv.gz",index=False)
    print("TERMINAL",traded.loc[traded.reason.eq("terminal_last_mark"),["date","symbol","year","top"]].drop_duplicates().to_json(orient="records",date_format="iso"),flush=True)
    print("ANNUAL_CSV_BEGIN\n"+summary.to_csv(index=False,float_format="%.8f")+"ANNUAL_CSV_END",flush=True)
    print("FX_CSV_BEGIN\n"+fx.to_csv(float_format="%.8f")+"FX_CSV_END",flush=True)
    print("COVERAGE_CSV_BEGIN\n"+checks.to_csv(index=False)+"COVERAGE_CSV_END",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dolt",required=True)
    ap.add_argument("--early",required=True)
    ap.add_argument("--out",default="results-us")
    ap.add_argument("--end",default="2026-11-27")
    run(ap.parse_args())
