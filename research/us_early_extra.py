"""Augment WIKI for 2010 using historical US stock archives and the 2011 tape."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests


SLUG = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
KIBOT = ("https://raw.githubusercontent.com/willhjw/big_movers/"
         "8ccb831aca1cda6a5d01471f7968e11552d88007/collected_stocks/{}.csv")
TIINGO = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
KNOWN = set("VALE PBR MT BHP NOK TEVA TSM ITUB ALU SAP BBL BCS TOT RIMM RIO ABV "
            "BBD ERIC AMX SNY AUY SID HBC POT ECA SU ABX GG CHL PBR-A VALE-P".split())
ALIASES = {"BB": "RIMM", "BBRY": "RIMM", "TTE": "TOT", "HSBC": "HBC"}
FIELDS = ["date", "symbol", "high", "low", "close", "raw_close", "turnover", "dividend", "eligible"]
START, END = pd.Timestamp("2008-01-01"), pd.Timestamp("2011-06-30")
YAHOO_START, YAHOO_END = 1199145600, 1795824000
HEADER = {"User-Agent": "Mozilla/5.0"}


def normalized(s):
    v = s.upper().replace(".", "-").replace("_", "-")
    return ALIASES.get(v, v)


def download(urls, dest):
    if dest.exists():
        return
    temp = dest.with_suffix(".part")
    for url in urls:
        try:
            with requests.get(url, headers=HEADER, stream=True, timeout=(7, 60)) as r:
                r.raise_for_status()
                with temp.open("wb") as out:
                    for block in r.iter_content(1024 * 1024):
                        out.write(block)
            with ZipFile(temp) as archive:
                if not archive.infolist():
                    raise ValueError("empty archive")
            temp.replace(dest)
            return
        except (OSError, ValueError, requests.RequestException) as exc:
            print("Extra archive fetch:", url, exc, flush=True)
            temp.unlink(missing_ok=True)
    raise RuntimeError(f"Download failed: {dest}")


def clean(x):
    x = x.copy()
    x.columns = [c.lower() for c in x.columns]
    x = x.rename(columns={"datetime":"date"})
    x["date"] = pd.to_datetime(x.date, errors="coerce")
    x = x[x.date.between(START, END)].dropna(subset=["high", "low", "close", "volume"])
    x = x[(x.close>0) & (x.low>0) & (x.high>=x.low) & (x.volume>0)]
    x = x.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    x["turnover"] = x.close*x.volume
    return x


def stock_metadata(master):
    frame = pd.read_csv(master, dtype=str, keep_default_na=False)
    frame = frame[(frame.assetType.str.lower()=="stock") &
                  frame.exchange.str.upper().isin(["NYSE","NASDAQ","NYSE AMERICAN","NYSE MKT","AMEX",
                                                     "NYSE ARCA","BATS","CBOE BZX"]) &
                  frame.priceCurrency.eq("USD")]
    lo, hi = pd.to_datetime(frame.startDate,errors="coerce"), pd.to_datetime(frame.endDate,errors="coerce")
    return set(frame.loc[(lo <= "2010-12-31") & (hi.isna()|(hi >= "2009-01-01")), "ticker"].map(normalized))


def archive_candidates(archive, allowed, covered, minimum):
    result = {}
    with ZipFile(archive) as z:
        for member in z.infolist():
            path = Path(member.filename.replace("\\", "/"))
            if not path.name.lower().endswith(".us.txt") or "stocks" not in [p.lower() for p in path.parts]:
                continue
            sym = normalized(path.name[:-7])
            if sym in covered or sym not in allowed or member.file_size<32:
                continue
            with z.open(member) as stream:
                frame = clean(pd.read_csv(stream))
            if len(frame)<20:
                continue
            amt = frame.turnover.rolling(20).mean()
            peak = amt[frame.date.between("2009-11-01","2010-12-31")].max()
            if sym in KNOWN or (np.isfinite(peak) and peak>=minimum):
                result[sym] = frame
    return result


def yahoo_bytes(symbol):
    query = {"period1":YAHOO_START, "period2":YAHOO_END, "interval":"1d", "events":"div,splits"}
    for host in ("query1", "query2"):
        try:
            r=requests.get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}",params=query,
                           headers=HEADER,timeout=(4,10))
            if r.status_code!=200:
                continue
            doc=r.json()["chart"]["result"][0]
            if doc["meta"].get("currency")!="USD":
                continue
            ts=np.asarray(doc["timestamp"])
            dates=pd.to_datetime(ts,unit="s",utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
            frame=pd.DataFrame(doc["indicators"]["quote"][0],index=dates)
            pairs=doc.get("events",{})
            splits=pd.Series(1.,index=dates)
            divs=pd.Series(0.,index=dates)
            for event in pairs.get("splits",{}).values():
                day=pd.Timestamp(event["date"],unit="s",tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
                if day in splits.index:
                    splits.loc[day] *= event["numerator"]/event["denominator"]
            for event in pairs.get("dividends",{}).values():
                day=pd.Timestamp(event["date"],unit="s",tz="UTC").tz_convert("America/New_York").tz_localize(None).normalize()
                if day in divs.index:
                    divs.loc[day] += event["amount"]
            future=splits.iloc[::-1].cumprod().shift(1,fill_value=1.).iloc[::-1]
            frame["raw_close"]=frame.close*future
            frame["dividend"]=divs*future
            frame["date"]=dates
            out=clean(frame.reset_index(drop=True))
            if (out.date.dt.year==2010).sum()>=20:
                return out
        except (ValueError, requests.RequestException, KeyError, TypeError, IndexError, ZeroDivisionError):
            continue
    return None


def past_archives(symbol):
    mapped={"RIMM":["RIMM","BBRY"],"TOT":["TOT","TTE"],"HBC":["HBC","HSBC"]}.get(symbol,[symbol])
    for s in mapped:
        try:
            r=requests.get(KIBOT.format(s),headers=HEADER,timeout=(4,10))
            if r.status_code!=200:
                continue
            d=clean(pd.read_csv(BytesIO(r.content)))
            if (d.date.dt.year==2010).sum()>=20:
                return d
        except (requests.RequestException,ValueError,KeyError):
            continue
    return None


def calibrate(d, ref, sym, source):
    d=d.copy()
    scale, value_scale, spread, corr = 1., 1., np.nan, np.nan
    overlap=0
    if ref is not None:
        pair=d.merge(ref[["date","raw_close","turnover"]],on="date",suffixes=("","_tape"))
        if len(pair)>=12:
            overlap=len(pair)
            raw=pair["raw_close_tape"] if "raw_close_tape" in pair else pair["raw_close"]
            corr=np.log(pair.close).diff().corr(np.log(raw).diff())
            # First sessions anchor the nearest boundary. Within this period a
            # cash dividend can cause a small drift in Stooq's adjusted close.
            anchor=(raw/pair.close).iloc[:15]
            scale=float(anchor.median())
            spread=float(np.median(abs(anchor/scale-1)))
            value_scale=float((pair.turnover_tape/pair.turnover).median())
            if not (.80<=corr<=1.0001 and spread<.025 and .55<=value_scale<=1.8):
                return None, [sym,source,overlap,corr,scale,value_scale,"rejected overlap"]
            if "raw_close" in d and not .95<=np.median(raw/pair["raw_close"])<=1.05:
                return None,[sym,source,overlap,corr,scale,value_scale,"Yahoo raw mismatched tape"]
    if "raw_close" not in d:
        if not overlap and not (source=="frozen" and sym in {"RIMM","ALU"}):
            return None,[sym,source,overlap,corr,scale,value_scale,"missing anchor"]
        d["raw_close"]=d.close*scale
        # Only calibrate systematic differences, not random volume-reporting noise.
        if abs(value_scale-1)>.05:
            d["turnover"]*=value_scale
    if "dividend" not in d:
        d["dividend"]=0.
    d["symbol"]=sym
    d["eligible"]=True
    return d[FIELDS], [sym,source,overlap,corr,scale,value_scale,"added"]


def enrich(early,dolt,out,cache):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    original=pd.read_parquet(early,columns=FIELDS)
    original["date"]=pd.to_datetime(original.date)
    covered=set(original.loc[original.date.between("2009-01-01","2010-12-31"),"symbol"].map(normalized))
    tape=pd.read_parquet(dolt,columns=["symbol","date","close","raw_close","turnover","eligible"],
        filters=[("date",">=",pd.Timestamp("2011-01-01").date()),("date","<=",END.date())])
    tape["date"]=pd.to_datetime(tape.date)
    refs={normalized(s):x for s,x in tape[tape.eligible].groupby("symbol")}
    z=cache/"stooq-2017.zip"
    download([f"https://api.kaggle.com/api/v1/datasets/download/{SLUG}",
              f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}"],z)
    master=cache/"tiingo-supported.zip"
    try:
        download([TIINGO],master)
        allowed=stock_metadata(master)|set(refs)|KNOWN
    except (ValueError,OSError,RuntimeError):
        allowed=set(refs)|KNOWN
    # Any omitted Stooq name below this screening bound would need turnover
    # underreported by >3x on its strongest 20-session window to enter top 30.
    fil=original.sort_values(["symbol","date"]).copy()
    fil["avg"]=fil.groupby("symbol").turnover.transform(lambda x:x.rolling(20).mean())
    threshold=fil[fil.date.dt.year==2010].groupby("date").avg.nlargest(30).groupby(level=0).min().min()/3.
    candidates=archive_candidates(z,allowed,covered,threshold)
    syms=sorted(set(candidates)|(KNOWN-covered))
    print("EXTRA candidates",len(syms),"threshold",threshold,flush=True)
    def one(sym):
        tried=[]
        # Prioritize contemporaneous prices and distributions. Frozen source
        # recovers defunct ALU/RIMM even if the current-symbol feed omits them.
        live={"RIMM":"BB","TOT":"TTE","HBC":"HSBC"}.get(sym,sym)
        series=yahoo_bytes(live)
        for source in ("yahoo","frozen","stooq"):
            if source=="frozen":
                series=past_archives(sym) if sym in KNOWN else None
            if source=="stooq":
                series=candidates.get(sym)
            if series is None:
                continue
            result,diagnostic=calibrate(series,refs.get(sym),sym,source)
            tried.append(diagnostic)
            if result is not None:
                return result,tried
        if not tried:
            tried.append([sym,"none",0,None,None,None,"missing"])
        return None,tried
    with ThreadPoolExecutor(max_workers=8) as pool:
        tested=list(pool.map(one,syms))
    logs=[row for _,record in tested for row in record]
    diag=pd.DataFrame(logs,columns=["symbol","source","overlap_days","return_corr","price_scale","turnover_scale","status"])
    diag.to_csv(out.with_suffix(".sources.csv"),index=False)
    frames=[original]+[x for x,_ in tested if x is not None]
    enriched=pd.concat(frames,ignore_index=True)
    duplicated=enriched.duplicated(["date","symbol"],keep="first")
    if duplicated.any():
        enriched=enriched[~duplicated].copy()
    enriched.to_parquet(out,index=False)
    print("EXTRA kept",len(frames)-1,"names",diag[diag.symbol.isin(KNOWN)][["symbol","source","status"]].to_dict("records"),flush=True)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for dep in ["early","dolt","out","cache"]:
        parser.add_argument("--"+dep,type=Path,required=True)
    args=parser.parse_args()
    enrich(args.early,args.dolt,args.out,args.cache)
