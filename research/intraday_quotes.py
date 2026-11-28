"""Align hourly KST quotes with the regular-session KRX daily bars used for the signals."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SEOUL = ZoneInfo("Asia/Seoul")
FROM = datetime(2025,5,1,tzinfo=SEOUL)
UNTIL = datetime(2026,11,28,tzinfo=SEOUL)
SLOTS = 8   # Local time 09:00 through 16:00, including the delayed exam-day session.


def retrieve(code):
    cache = Path("market/hourly")
    cache.mkdir(exist_ok=True)
    dest = cache / f"{code}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    params = urlencode(dict(interval="60m",period1=int(FROM.timestamp()),period2=int(UNTIL.timestamp())))
    req = Request(f"https://query1.finance.yahoo.com/v8/finance/chart/{code}.KS?{params}",
                  headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    for attempt in range(2):
        try:
            with urlopen(req,timeout=16) as res:
                data = json.load(res)["chart"]["result"][0]
            if data["meta"]["symbol"].upper()!=f"{code}.KS":
                return None
            obj = [data["timestamp"],data["indicators"]["quote"][0]]
            dest.write_text(json.dumps(obj))
            return obj
        except HTTPError as error:
            if error.code in [429,500,502,503,504] and attempt==0:
                time.sleep(min(20, int(error.headers.get("Retry-After","3"))))
                continue
            return None
        except (OSError,ValueError,KeyError,TypeError,IndexError):
            if attempt == 0:
                time.sleep(1)
    return None


def verified_hours(index, m, order, out):
    n, k = m["c"].shape
    bars = np.full((n,k,SLOTS,4), np.nan)
    valid = np.zeros((n,k), dtype=bool)
    columns = ["open","high","low","close"]
    raw = np.stack([m[f] for f in ["o","h","l","c"]],axis=-1)/m["factor"][:,:,None]
    summary = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for j,obj in enumerate(pool.map(retrieve,order)):
            if obj is None:
                continue
            dt = pd.to_datetime(obj[0], unit="s",utc=True).tz_convert("Asia/Seoul")
            values = pd.DataFrame(obj[1]).reindex(columns=columns).to_numpy(float)
            day = index.index.get_indexer(dt.tz_localize(None).normalize())
            slot = dt.hour.to_numpy()-9
            good = (day>=0)&(slot>=0)&(slot<SLOTS)&(dt.minute==0)&np.isfinite(values).all(axis=1)&(values.min(axis=1)>0)
            good &= (values[:,1]>=values[:,0])&(values[:,1]>=values[:,3])&(values[:,2]<=values[:,0])&(values[:,2]<=values[:,3])
            bars[day[good],j,slot[good]] = values[good]
            sampled = np.isfinite(bars[:,j,:,0]).sum(axis=1)
            for i in np.where((sampled>=6) & np.isfinite(raw[:,j,0]))[0]:
                sub = bars[i,j]
                observed = np.flatnonzero(np.isfinite(sub[:,0]))
                first, last = observed[0], observed[-1]
                if first not in (0,1) or not np.array_equal(observed,np.arange(first,last+1)):
                    continue
                if last not in (first+5,first+6):
                    continue
                oficial_open, high, low, close = raw[i,j]
                # Daily extremes must both be traceable to an observed bar or to the KRX close.
                # Reject stale/synthetic prints outside the KRX range or with a different opening price.
                if abs(sub[first,0]-oficial_open)>.05 or abs(max(sub[observed,1].max(),close)-high)>.05 or abs(min(sub[observed,2].min(),close)-low)>.05:
                    continue
                if last == first+5:
                    prev = sub[last,3]
                    sub[last+1] = (prev,max(prev,close),min(prev,close),close)
                else:
                    sub[last,1] = max(sub[last,1],close)
                    sub[last,2] = min(sub[last,2],close)
                    sub[last,3] = close
                valid[i,j] = True
            summary.append((order[j],int(good.sum()),int(valid[:,j].sum())))
    pd.DataFrame(summary,columns=["code","hourly_rows","matched_days_including_warmup"]).to_csv(out/"quote_validation.csv",index=False)
    print("HOURLY_SOURCE",len(summary),len(order),int(valid.sum()),flush=True)
    return bars,valid


def tick(p):
    return np.select([p<2000,p<5000,p<20000,p<50000,p<200000,p<500000],[1,5,10,50,100,500],default=1000)


def hourly_path(index,m,hours,valid,policy):
    factor = m["factor"]
    o,h,l,c = [m[name]/factor for name in ["o","h","l","c"]]
    high_first = (h-o)<(o-l) if policy == "near" else np.full_like(o,policy=="high",dtype=bool)
    a,b = np.where(high_first,h,l),np.where(high_first,l,h)
    def interpolated(t):
        t = np.clip(t/6.5*3,0,3)
        raw = np.where(t<=1,o+(a-o)*t,np.where(t<=2,a+(b-a)*(t-1),b+(c-b)*(t-2)))
        grid = tick(raw)
        return np.round(raw/grid)*grid
    # Estimate regular opening hour for a fallback stock from the other quoted stocks on the same date.
    at_nine = np.isfinite(hours[:,:,0,0]) & valid
    late = ((valid.sum(axis=1)>0) & (at_nine.sum(axis=1)*2<valid.sum(axis=1))).astype(float)[:,None]
    phase_names = []
    for slot in range(SLOTS):
        left = slot-late
        right = np.minimum(left+1,6.5)
        start, end = interpolated(left),interpolated(right)
        top,bot = np.maximum(start,end),np.minimum(start,end)
        for t,v in [(6.5/3,a),(13./3,b)]:
            within = (left<=t)&(t<=right)
            top = np.where(within,np.maximum(top,v),top)
            bot = np.where(within,np.minimum(bot,v),bot)
        fallback_active = (left>=0)&(left<6.5)
        extrema = [np.where(fallback_active,p,np.nan) for p in (start,top,bot,end)]
        u,hi,lo,cl = [np.where(valid,hours[:,:,slot,k],extrema[k]) for k in range(4)]
        # Vendor feeds can contain fractional won/tick quotations. Quantize hourly endpoints
        # to the exchange grid, keeping threshold fills within official daily price bounds.
        quant = lambda p: np.clip(np.round(p/tick(p))*tick(p),l,h)
        u,cl = quant(u),quant(cl)
        hi = np.maximum(np.maximum(u,cl),np.floor(hi/tick(hi))*tick(hi))
        lo = np.minimum(np.minimum(u,cl),np.ceil(lo/tick(lo))*tick(lo))
        hi_first = (hi-u)<(u-lo) if policy=="near" else policy=="high"
        for key,arr in zip(("open","first","second","close"),(u,np.where(hi_first,hi,lo),np.where(hi_first,lo,hi),cl)):
            name = f"hour{slot+9}_{key}"
            m[name] = arr*factor
            phase_names.append(name)
    return phase_names

