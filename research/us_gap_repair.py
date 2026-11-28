"""Fill only corroborated interior holes in the liquid US stock tape."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import argparse
from io import StringIO
import json
import shutil

import duckdb
import exchange_calendars as xc
import numpy as np
import pandas as pd

from us_data_pipeline import HEX_BASE, fetch, sqlval, sqlpaths


RECENT = ("https://huggingface.co/datasets/AmirTrader/YahooFinance/resolve/"
          "7beb0b92fe169273848e03b778ffdd111a975c51/data/daily")
SP500 = "https://github.com/Johnbrick123/sp500-data/releases/download/data/prices.parquet"
COLS = ["date","symbol","high","low","close","raw_close","turnover","dividend","eligible"]


def identify(db, sessions):
    db.register("exchange_days", pd.DataFrame({"day":sessions, "i":np.arange(len(sessions))}))
    # Extensive overlap from widely separated companies reusing one ticker is
    # intentionally excluded: trade days must bracket holes at most 10 sessions apart.
    return db.execute("""
        WITH candidates AS (
            SELECT symbol FROM tape GROUP BY symbol HAVING max(turnover)>=250000000
        ), days AS (
            SELECT q.symbol,d.i,q.turnover,lag(d.i) OVER w AS prev,
                   avg(q.turnover) OVER (PARTITION BY q.symbol ORDER BY d.i
                       ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) AS local_adv,
                   max(q.turnover) OVER (PARTITION BY q.symbol ORDER BY d.i
                       ROWS BETWEEN 20 PRECEDING AND 20 FOLLOWING) AS local_peak
            FROM tape q JOIN candidates USING(symbol)
            JOIN exchange_days d ON q.date=d.day
            WHERE q.eligible
            WINDOW w AS (PARTITION BY q.symbol ORDER BY d.i)
        )
        SELECT d.symbol, prev::INTEGER AS left_i, d.i::INTEGER AS right_i,
               d.local_adv,d.local_peak
        FROM days d WHERE d.i-prev BETWEEN 2 AND 11 AND
          (d.local_adv>=60000000 OR d.local_peak>=250000000)
    """).fetchdf()


def normalize_quote(h, left, right, sleft, sright, context, raw_donor=False):
    """Return native OHLC/raw/turnover, or a diagnostic string."""
    if h is None or sleft is None or sright is None:
        return "missing_source_or_anchor"
    for x in (h,sleft,sright):
        if not all(np.isfinite(x.get(k,np.nan)) and x[k]>0 for k in ("close","high","low","volume")):
            return "invalid_source"
        if x["low"]>x["close"]+1e-6 or x["high"]<x["close"]-1e-6:
            return "invalid_source_range"
    raw = np.array([left.raw_close/sleft["close"],right.raw_close/sright["close"]])
    adj = np.array([left.close/sleft["close"],right.close/sright["close"]])
    liquid = np.array([left.turnover/(sleft["close"]*sleft["volume"]),
                       right.turnover/(sright["close"]*sright["volume"])])
    if not np.all(np.isfinite(np.r_[raw,adj,liquid])) or min(*raw,*adj,*liquid)<=0:
        return "invalid_native_anchor"
    if abs(raw[0]/raw[1]-1)>.006 or abs(adj[0]/adj[1]-1)>.006:
        return "price_basis_or_identity_disagreement"
    if raw_donor:
        # These sources explicitly report historical USD/share and share count.
        # One partial-tape volume outlier on an anchor can be tolerated only
        # when surrounding actual-volume observations establish the same units.
        neighboring_ratio = [q.turnover/(s["close"]*s["volume"]) for q,s in context
                             if s is not None and s["close"]>0 and s["volume"]>0]
        if (np.max(abs(raw-1))>.006 or len(neighboring_ratio)<4
                or np.median(abs(np.asarray(neighboring_ratio)-1))>.25
                or np.median(abs(liquid-1))>.5):
            return "raw_usd_or_volume_units_disagree"
    elif np.max(np.abs(liquid-1))>.35:
        return "volume_basis_disagreement"
    stable = [native.raw_close/src["close"] for native,src in context
              if src is not None and np.isfinite(src["close"]) and src["close"]>0]
    if len(stable)<3 or np.median(np.abs(np.asarray(stable)/np.median(raw)-1))>.006:
        return "additional_anchors_disagree"
    # Exact split-only basis is constant between surrounding native dates.
    # Median calibrates absolute units across vendors; round the historical
    # raw close to a tradable quote increment, not the adjusted indicators.
    rawpx = float(h["close"] * (1. if raw_donor else np.median(raw)))
    rawpx = round(rawpx,2 if rawpx>=1 else 4)
    native_factor = np.median([left.close/left.raw_close,right.close/right.raw_close])
    price = rawpx*native_factor
    factor = native_factor if raw_donor else np.median(adj)
    hi = max(float(h["high"]*factor),price)
    lo = min(float(h["low"]*factor),price)
    return (hi,lo,price,rawpx,float(h["close"]*h["volume"]))


def read_source(db, path, kind):
    if kind=="hex":
        relation=f"read_parquet({sqlpaths(path)})"
        name="upper(replace(q.symbol,'.','-'))"
    else:
        relation=f"read_parquet({sqlval(Path(path).as_posix())})"
        name="upper(replace(q.ticker,'.','-'))"
    return db.execute(f"""
        SELECT a.symbol,a.date,q.high,q.low,q.close,q.volume
        FROM {relation} q JOIN needed_index a ON {name}=a.symbol AND q.date::DATE=a.date
    """).fetchdf()


def recent_symbol(cache, sym):
    import requests
    path=cache/"recent"/(sym+".parquet")
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        try:
            resp=requests.get(f"{RECENT}/{quote(sym,safe='')}.parquet",timeout=(5,12))
            if resp.status_code!=200 or resp.content[:4]!=b"PAR1":
                return None
            path.write_bytes(resp.content)
        except (requests.RequestException,OSError):
            return None
    try:
        frame=pd.read_parquet(path,columns=["date","high","low","close","volume"])
    except (ValueError,OSError):
        return None
    frame["symbol"]=sym
    frame["date"]=pd.to_datetime(frame.date)
    return frame


def wiki_raw(db, source):
    from us_early import _canonical
    path=Path(source)
    if path.is_dir():
        paths=sorted(p for p in path.glob("wiki-price-table.*") if p.suffix.lower() in
                     (".parquet",".csv",".pq"))
        if len(paths)!=1:
            raise ValueError(f"Expected one extracted WIKI price table: {paths}")
        path=paths[0]
    reader="read_csv_auto" if path.suffix.lower()==".csv" else "read_parquet"
    rel=f"{reader}({sqlval(path.as_posix())})"
    raw={_canonical(c[0]):'q."'+c[0].replace('"','""')+'"' for c in db.execute(
        f"DESCRIBE SELECT * FROM {rel}").fetchall()}
    cols=", ".join(f"{raw[c]}::DOUBLE AS {c}" for c in ("high","low","close","volume"))
    symbol=f"upper(replace(replace(replace({raw['symbol']}, 'WIKI/',''),'.','-'),'_','-'))"
    symbol=f"CASE {symbol} WHEN 'FB' THEN 'META' WHEN 'BBRY' THEN 'RIMM' ELSE {symbol} END"
    return db.execute(f"""SELECT a.symbol,a.date,{cols} FROM {rel} q JOIN needed_index a
         ON {symbol}=a.symbol AND {raw['date']}::DATE=a.date""").fetchdf()


def kibot_raw(sym, start, end):
    import requests
    args=dict(action="history",symbol=sym,interval="daily",startdate=start.strftime("%m/%d/%Y"),
              enddate=end.strftime("%m/%d/%Y"),unadjusted="1",user="guest",password="guest")
    try:
        resp=requests.get("https://api.kibot.com/",params=args,timeout=(5,12))
        if resp.status_code!=200:
            return None
        data=pd.read_csv(StringIO(resp.text),header=None)
        if data.shape[1] not in (6,7):
            return None
        data.columns=(["date","time"] if data.shape[1]==7 else ["date"]) + ["open","high","low","close","volume"]
        data["date"]=pd.to_datetime(data.date,errors="coerce")
        data["symbol"]=sym
        numeric=["high","low","close","volume"]
        data[numeric]=data[numeric].apply(pd.to_numeric,errors="coerce")
        return data[["symbol","date",*numeric]].dropna()
    except (ValueError,pd.errors.ParserError,requests.RequestException):
        return None


def repair(inp,out,cache,dividends=None,wiki_source=None):
    inp,out,cache=Path(inp),Path(out),Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    db=duckdb.connect()
    db.execute("SET threads=3")
    db.execute("SET memory_limit='3GB'")
    db.execute(f"CREATE VIEW tape AS SELECT * FROM read_parquet({sqlval(inp.as_posix())})")
    lo,hi=db.execute("SELECT min(date),max(date) FROM tape").fetchone()
    sessions=xc.get_calendar("XNYS",start=pd.Timestamp(lo),end=pd.Timestamp(hi)).sessions.tz_localize(None)
    holes=identify(db,sessions)
    need=set()
    tasks=[]
    for row in holes.itertuples(index=False):
        for i in range(max(0,row.left_i-10),min(len(sessions),row.right_i+11)):
            need.add((row.symbol,sessions[i]))
        for i in range(row.left_i+1,row.right_i):
            tasks.append((row.symbol,i,row.left_i,row.right_i,row.local_adv,row.local_peak))
    print(f"Potential short gaps: {len(tasks)} sessions across {holes.symbol.nunique()} liquid symbols",flush=True)
    if not tasks:
        shutil.copyfile(inp,out)
        return
    ni=pd.DataFrame(sorted(need),columns=["symbol","date"])
    db.register("needed_index",ni)
    native=db.execute("SELECT t.* FROM tape t SEMI JOIN needed_index n USING(symbol,date)").fetchdf()
    native["date"]=pd.to_datetime(native.date)
    native={(r.symbol,r.date):r for r in native.itertuples(index=False)}
    added={}
    notes={}
    dividends_by_day={}
    if dividends and Path(dividends).exists():
        ds=pd.read_csv(dividends,usecols=["act_symbol","ex_date","amount"])
        ds["symbol"]=ds.act_symbol.astype(str).str.replace(".","-",regex=False)
        ds["date"]=pd.to_datetime(ds.ex_date)
        # Align this known corporate rename to the primary tape IDs.
        ds.loc[(ds.symbol=="FB") & ds.date.between("2012-05-18","2022-06-08"),"symbol"]="META"
        dividends_by_day=ds.groupby(["symbol","date"]).amount.sum().to_dict()

    def apply(src,label,raw_donor=False):
        if src is None or src.empty:
            return
        src["date"]=pd.to_datetime(src.date)
        dup=src.duplicated(["symbol","date"],keep=False)
        if dup.any():
            src=src.loc[~dup]
        source={(r.symbol,r.date):{"high":r.high,"low":r.low,"close":r.close,"volume":r.volume}
                for r in src.itertuples(index=False)}
        for sym,i,a,b,adv,peak in tasks:
            key=(sym,sessions[i])
            if key in added or key not in source:
                continue
            ka,kb=(sym,sessions[a]),(sym,sessions[b])
            look=10 if raw_donor else 3
            near=[(sym,sessions[j]) for j in range(max(0,a-look),min(len(sessions),b+look+1))]
            ctx=[(native[k],source[k]) for k in near if k in native and k in source]
            ans=normalize_quote(source[key],native[ka],native[kb],source.get(ka),source.get(kb),ctx,raw_donor)
            if isinstance(ans,str):
                notes[key]=label+":"+ans
                continue
            high,low,close,raw_close,turnover=ans
            added[key]=(sessions[i],sym,high,low,close,raw_close,turnover,
                        float(dividends_by_day.get(key,0.)),True)
            notes[key]=label
        print(f"Corroborated {len(added)} total gaps after {label}",flush=True)

    if wiki_source:
        apply(wiki_raw(db,wiki_source),"WIKI-raw",True)
    shards=[f"train-{i:05d}-of-00004.parquet" for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        hex_files=list(pool.map(lambda name:fetch(f"{HEX_BASE}/{name}",cache/name),shards))
    apply(read_source(db,hex_files,"hex"),"HexQuant")
    try:
        sp=fetch(SP500,cache/"sp500.parquet")
        apply(read_source(db,sp,"sp500"),"SP500-archive")
    except (OSError,duckdb.Error) as exc:
        print("Optional S&P archive unavailable",type(exc).__name__,flush=True)
    # This later snapshot improves coverage of IPOs and trading dates after
    # Dec 2025. Fetch unresolved identities only, with a conservative cap.
    unresolved={}
    for sym,i,a,b,adv,peak in tasks:
        if (sym,sessions[i]) not in added and sessions[i]>=pd.Timestamp("2019-11-26"):
            unresolved[sym]=max(unresolved.get(sym,0),peak)
    recent=sorted(unresolved,key=unresolved.get,reverse=True)[:600]
    with ThreadPoolExecutor(max_workers=12) as pool:
        parts=[v for v in pool.map(lambda symbol:recent_symbol(cache,symbol),recent) if v is not None]
    if parts:
        apply(pd.concat(parts,ignore_index=True).merge(ni,on=["symbol","date"]),"AmirTrader")
    # A separate feed with genuine unadjusted prints for the largest remaining
    # holes since WIKI ceased publication. Cap the guest requests explicitly.
    missing=[(sym,sessions[i],adv) for sym,i,a,b,adv,peak in tasks
             if (sym,sessions[i]) not in added and sessions[i]>pd.Timestamp("2018-03-27")]
    if missing:
        latest=pd.DataFrame(missing,columns=["symbol","date","adv"])
        priority=latest.groupby("symbol").adv.max().nlargest(10).index
        ext=[]
        for symbol in priority:
            window=latest.loc[latest.symbol.eq(symbol),"date"]
            part=kibot_raw(symbol,window.min()-pd.Timedelta(days=35),window.max()+pd.Timedelta(days=35))
            if part is not None:
                ext.append(part)
        if ext:
            apply(pd.concat(ext,ignore_index=True).merge(ni,on=["symbol","date"]),"Kibot-raw",True)
    rows=[]
    for sym,i,a,b,adv,peak in tasks:
        key=(sym,sessions[i])
        rows.append(dict(date=sessions[i],symbol=sym,left=sessions[a],right=sessions[b],local_adv=adv,
                         nearby_peak=peak,recovered=key in added,source_or_reason=notes.get(key,"unavailable")))
    audit=pd.DataFrame(rows)
    audit.to_csv(out.with_suffix(".gap-audit.csv"),index=False)
    if not added:
        shutil.copyfile(inp,out)
    else:
        fills=pd.DataFrame(added.values(),columns=COLS)
        db.register("gap_fills",fills)
        if db.execute("SELECT count(*) FROM gap_fills JOIN tape USING(date,symbol)").fetchone()[0]:
            raise ValueError("Repair would overwrite an existing quote")
        db.execute(f"""COPY (SELECT * FROM tape UNION ALL BY NAME SELECT * FROM gap_fills)
                       TO {sqlval(out.as_posix())} (FORMAT PARQUET,COMPRESSION ZSTD)""")
    fn=lambda d: d.to_json(orient="records",date_format="iso")
    print("GAP_AUDIT",json.dumps({"attempted":len(tasks),"repaired":len(added),
          "symbols":len(set(s for s,_ in added)),"missing_before":len(tasks),
          "missing_after":len(tasks)-len(added)}),flush=True)
    print("GME_2021",fn(audit[(audit.symbol=="GME")&(audit.date.dt.year==2021)]),flush=True)
    print("LARGEST_UNVERIFIED",fn(audit[~audit.recovered].nlargest(15,"local_adv")),flush=True)
    db.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--cache",default="cache/us-gap-repair")
    p.add_argument("--dividends",help="Source Dolt dividend CSV")
    p.add_argument("--wiki-source",help="Extracted WIKI raw price table or its directory")
    a=p.parse_args()
    repair(a.input,a.output,a.cache,a.dividends,a.wiki_source)
