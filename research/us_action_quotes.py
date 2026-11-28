"""Restore three ex-date parent marks present in an independent split-separated stock history."""
from pathlib import Path
import argparse

import duckdb
import pandas as pd


SOURCE = "https://github.com/floyds1995/Auto-Index-Constituents-Tracker/tree/7d8f1236a1652c85bc7c7b76c2e6030fe3ae8d88/us_data/raw_ohlcv/parquet"
# date, symbol, unadjusted high, low, close, actual shares traded
BARS = [
    ("2020-04-03","RTX",53.30,48.05,49.93,18932300),
    ("2021-06-03","MRK",74.00,72.90,73.91,15974400),
    ("2021-11-04","IBM",123.34,119.90,120.85,7208700),
]


def restore(path):
    db=duckdb.connect()
    path=Path(path)
    db.read_parquet(str(path)).create_view("quotes")
    extra=[]
    for day,sym,hi,lo,c,volume in BARS:
        d=pd.Timestamp(day)
        old=db.execute("SELECT raw_close FROM quotes WHERE symbol=? AND date=?", [sym,d.date()]).fetchone()
        if old:
            print("EXDATE_ALREADY_QUOTED",day,sym,old[0],flush=True)
            continue
        ref=db.execute("SELECT close/raw_close FROM quotes WHERE symbol=? AND date>? AND date<=? ORDER BY date LIMIT 1",
                       [sym,d.date(),(d+pd.Timedelta(days=21)).date()]).fetchone()
        if not ref:
            print("EXDATE_NO_PRICE_BASIS",day,sym,flush=True)
            continue
        f=float(ref[0])
        extra.append(dict(date=d.date(),symbol=sym,high=hi*f,low=lo*f,close=c*f,raw_close=c,
                          turnover=c*volume,eligible=True,dividend=0.))
        print("EXDATE_RESTORED",day,sym,c,volume,SOURCE,flush=True)
    if not extra:
        return
    tmp=path.with_suffix(".event-updated.parquet")
    db.register("extra",pd.DataFrame(extra))
    db.sql("SELECT * FROM quotes UNION ALL BY NAME SELECT * FROM extra").write_parquet(str(tmp),compression="zstd")
    db.close()
    tmp.replace(path)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("path")
    restore(p.parse_args().path)
