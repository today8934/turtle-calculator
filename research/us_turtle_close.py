"""US stock, close-only Turtle S1. Adjusted prices must be split-consistent."""
from dataclasses import dataclass
from math import ceil, floor

import numpy as np
import pandas as pd


@dataclass
class ClosePanels:
    dates: pd.DatetimeIndex
    symbols: tuple
    close: np.ndarray
    factor: np.ndarray
    rank: np.ndarray
    n: np.ndarray
    low10: np.ndarray
    high10: np.ndarray
    signal: np.ndarray
    strength: np.ndarray
    returns: np.ndarray
    last: np.ndarray
    dividend: np.ndarray


def _price(price, factor, up, strict=False):
    """Round a trigger in actual dollars, then return it in adjusted units."""
    raw = price / factor
    tick = .01 if raw >= 1 else .0001
    steps = raw / tick
    if strict:
        result = floor(steps+1e-8)+1 if up else ceil(steps-1e-8)-1
    else:
        result = ceil(steps-1e-8) if up else floor(steps+1e-8)
    return result*tick*factor


def _channel(series, lookback, factor, up):
    rolling = series.rolling(lookback)
    bounds = (rolling.max() if up else rolling.min()).shift().to_numpy()
    raw = bounds/factor
    tick = np.where(raw>=1,.01,.0001)
    step = np.floor(raw/tick+1e-8)+1 if up else np.ceil(raw/tick-1e-8)-1
    return step*tick*factor


def prepare(quotes, calendar=None, max_top=30):
    """date, symbol, high, low, close, turnover; optional rank, raw_close, dividend.

    Adjusted high/low/close use one common split basis per security. raw_close is
    required when this basis differs from traded dollars. Optional rank is the
    already lagged, point-in-time rank in the full eligible universe; never lag
    that field twice. A supplied eligible bool is applied on the order date.
    Missing ranks are ineligible. Rank only limits fresh entries, not exits/adds.
    Dividend is gross cash per raw share on the ex-date; use split-only prices
    when providing it. With no dividend column, returns are price returns.
    """
    q = quotes.copy(deep=False)
    q["date"] = pd.to_datetime(q.date)
    q["symbol"] = q.symbol.astype(str)
    dates = pd.DatetimeIndex(sorted(q.date.unique()) if calendar is None else calendar)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Trading calendar must be unique and increasing")
    q = q[q.date.isin(dates)]
    if q.duplicated(["date","symbol"]).any():
        raise ValueError("Duplicate security-day")
    if "rank" not in q:
        turnover = q.pivot(index="date",columns="symbol",values="turnover").reindex(dates)
        values = turnover.rolling(20,min_periods=20).mean().shift()
        if "eligible" in q:
            eligible = q.pivot(index="date",columns="symbol",values="eligible").reindex(dates)
            values = values.where(eligible.eq(True))
        ranks = values.rank(axis=1,ascending=False,method="first")
        symbols = tuple(ranks.columns[ranks.le(max_top).any()])
        ranked = ranks.loc[:,list(symbols)]
    else:
        if "eligible" in q:
            q["rank"] = q["rank"].where(q.eligible.eq(True))
        symbols = tuple(sorted(q.loc[q["rank"].le(max_top),"symbol"].unique()))
        ranked = q[q.symbol.isin(symbols)].pivot(index="date",columns="symbol",values="rank").reindex(index=dates,columns=symbols)
    shape = (len(dates),len(symbols))
    if not symbols:
        raise ValueError("No securities in the supplied ranking")
    c, fac, n, lo10, hi10, div = [np.full(shape,np.nan) for _ in range(6)]
    side_signal = np.zeros(shape,dtype=np.int8)
    last = np.full(len(symbols),-1,dtype=int)
    sec_index = {s:k for k,s in enumerate(symbols)}
    for symbol, history in q.loc[q.symbol.isin(symbols)].groupby("symbol",sort=False,observed=True):
        k = sec_index[symbol]
        d = history.sort_values("date").set_index("date")
        d = d.loc[d[["high","low","close"]].gt(0).all(axis=1)]
        if d.empty:
            continue
        if (d.high.lt(d.close-1e-6)|d.low.gt(d.close+1e-6)).any():
            raise ValueError(f"{symbol}: closing price outside its range")
        pos = dates.get_indexer(d.index)
        prices = d.close.to_numpy(dtype=float)
        factor = prices/d.raw_close.to_numpy(dtype=float) if "raw_close" in d else np.ones(len(d))
        if not np.all(np.isfinite(factor)&(factor>0)):
            raise ValueError(f"{symbol}: invalid adjusted / raw close ratio")
        c[pos,k],fac[pos,k],last[k] = prices,factor,pos[-1]
        if "dividend" in d:
            div[pos,k] = d.dividend.fillna(0)
        previous = d.close.shift()
        tr = pd.concat([d.high-d.low,(d.high-previous).abs(),(d.low-previous).abs()],axis=1).max(axis=1)
        seeded = tr.copy()
        seeded.iloc[:19] = np.nan
        if len(tr)>=20:
            seeded.iloc[19] = tr.iloc[:20].mean()
        atr = seeded.ewm(alpha=1/20,adjust=False).mean()
        week = d.index.to_period("W-FRI")
        by_week = atr.groupby(week).last().reindex(pd.period_range(week.min(),week.max(),freq="W-FRI")).ffill()
        vola = by_week.reindex(week-1).to_numpy()
        n[pos,k] = vola
        up20,down20 = _channel(d.high,20,factor,True),_channel(d.low,20,factor,False)
        up55,down55 = _channel(d.high,55,factor,True),_channel(d.low,55,factor,False)
        upper,lower = _channel(d.high,10,factor,True),_channel(d.low,10,factor,False)
        hi10[pos,k],lo10[pos,k] = upper,lower
        side, prior_won, wait55 = 0,False,False
        entry = stop = np.nan
        for t in np.flatnonzero((vola>0) & np.isfinite(up20)):
            price = prices[t]
            tol = 1e-8*factor[t]
            if side==1 and price<=max(stop,lower[t])+tol:
                prior_won = price>entry+tol
                side,wait55=0,False
            elif side==-1 and price>=min(stop,upper[t])-tol:
                prior_won = price<entry-tol
                side,wait55=0,False
            if not side:
                sign = 1 if price>=up20[t]-tol else -1 if price<=down20[t]+tol else 0
                if sign:
                    side,entry,wait55 = sign,price,prior_won
                    stop = _price(price-sign*2*vola[t],factor[t],sign<0)
                    if not wait55:
                        side_signal[pos[t],k] = sign
            if wait55 and ((side==1 and price>=up55[t]-tol) or (side==-1 and price<=down55[t]+tol)):
                side_signal[pos[t],k] = side
                wait55 = False
    closes = pd.DataFrame(c,index=dates)
    strength = (closes.shift(1)-closes.shift(64)).to_numpy()/n
    returns = closes.pct_change(fill_method=None).to_numpy()
    return ClosePanels(dates,symbols,c,fac,ranked.to_numpy(dtype=float),n,lo10,hi10,
                       side_signal,strength,returns,last,div)


def _room(panels,i,k,side,positions):
    live = {j:len(p["lots"]) for j,p in positions.items() if p["side"]==side}
    live[k] = live.get(k,0)+1
    units = np.array(list(live.values()))
    if units.sum()>12 or live[k]>4:
        return False
    if units.sum()<=6:
        return True
    codes = list(live)
    corr = pd.DataFrame(panels.returns[max(0,i-60):i,codes]).corr(min_periods=40).fillna(1.).to_numpy()
    np.fill_diagonal(corr,1)
    return bool(((corr>=.70)@units<=6).all() and ((corr>=.40)@units<=10).all())


def simulate_year(panels,year,top,initial_usd,fee_bps=5,borrow_bps=50,settlements=None,terminal_policy="raise"):
    """Annual independent USD cash account; dictionary of settlements uses (date,id)
    -> raw cash per share. Missing, unverified terminal events raise by default;
    terminal_policy='last_mark' explicitly assumes proceeds at the last quote.
    All execution occurs at close with exits funded before entries. New shorts
    require 100% collateral, and their sale proceeds cannot fund new purchases.
    """
    if initial_usd<=0 or terminal_policy not in ("raise","last_mark"):
        raise ValueError("Invalid funding or terminal policy")
    m=panels
    days=np.flatnonzero(m.dates.year==year)
    if not len(days) or not days[0]:
        raise ValueError(f"{year}: missing test dates or prior warmup")
    settle = {(pd.Timestamp(date),str(sym)):float(value)
              for (date,sym),value in (settlements or {}).items()}
    if any(not np.isfinite(v) or v<0 for v in settle.values()):
        raise ValueError("Invalid terminal proceeds")
    terminal_dates = {s:d for (d,s) in sorted(settle)}
    fee_rate,borrow_rate=fee_bps/10000,borrow_bps/10000
    cash=equity=notional=float(initial_usd)
    cut=initial_usd*.9
    pos,ledger,finished={},{},[]
    trades=[]
    marks=np.full(len(m.symbols),np.nan)
    factors=marks.copy()
    daily=[(m.dates[days[0]-1],cash,equity,0.,0.,0.,0)]
    fees=borrow=dividends=0.
    missing_term=0

    def record(k,sign,enter,qty,px,factor,reason):
        nonlocal cash,fees
        value=px*qty
        charge=value*fee_rate
        flow=(-sign if enter else sign)*value-charge
        cash+=flow
        fees+=charge
        ledger[k]["net"]+=flow
        if enter:
            ledger[k]["notional"]+=value
        trades.append((date,m.symbols[k],sign,reason,"buy" if (sign==1)==enter else "sell",
                       px/factor,qty*factor,charge,cash))

    def complete(k):
        p=ledger.pop(k)
        finished.append((p["net"],p["net"]/p["notional"]*100))
        del pos[k]

    for i in days:
        date=m.dates[i]
        if equity>=initial_usd:
            notional=initial_usd
            cut=.9*initial_usd
        while equity<=cut and notional>=.01:
            notional*=.8
            cut-=.1*notional
        elapsed=(date-m.dates[i-1]).days
        for k,p in pos.items():
            if p["side"]==-1:
                debt=p["qty"]*marks[k]*borrow_rate*elapsed/365
                cash-=debt
                ledger[k]["net"]-=debt
                borrow+=debt
        quoted=np.isfinite(m.close[i])
        marks[quoted]=m.close[i,quoted]
        factors[quoted]=m.factor[i,quoted]
        for k in list(pos):
            p=pos[k]
            s=p["side"]
            raw_dist=m.dividend[i,k]
            if np.isfinite(raw_dist) and raw_dist:
                flow=s*p["qty"]*factors[k]*raw_dist
                cash+=flow
                ledger[k]["net"]+=flow
                dividends+=flow
            symbol=m.symbols[k]
            settlement=settle.get((date,symbol))
            reason="settlement"
            if settlement is None and i>m.last[k]:
                if terminal_dates.get(symbol,date)>date:
                    continue
                if terminal_policy=="raise":
                    raise ValueError(f"Unresolved terminal holding: {symbol}, {date.date()}, last close {m.dates[m.last[k]].date()}")
                settlement=marks[k]/factors[k]
                reason="terminal_last_mark"
                missing_term+=1
            if settlement is not None:
                for _,qty in p["lots"]:
                    record(k,s,False,qty,settlement*factors[k],factors[k],reason)
                complete(k)
                continue
            if not quoted[k]:
                continue
            px,f=m.close[i,k],m.factor[i,k]
            bound=m.low10[i,k] if s==1 else m.high10[i,k]
            remaining=[]
            for stop,qty in p["lots"]:
                live_stop=_price(stop,f,s<0)
                hit_stop=s*(px-live_stop)<=1e-8*f
                hit_channel=s*(px-bound)<=1e-8*f
                if hit_stop or hit_channel:
                    record(k,s,False,qty,px,f,"2N" if hit_stop and (not hit_channel or s*(live_stop-bound)>=0) else "10D")
                else:
                    remaining.append((stop,qty))
            p["lots"]=remaining
            p["qty"]=sum(q for _,q in remaining)
            if not remaining:
                complete(k)

        candidates=set(pos)|set(np.flatnonzero((m.rank[i]<=top) & (m.signal[i]!=0)))
        def priority(k):
            side=pos[k]["side"] if k in pos else int(m.signal[i,k])
            value=m.strength[i,k]
            return (-side*value if np.isfinite(value) else np.inf,m.symbols[k])
        for k in sorted(candidates,key=priority):
            if not quoted[k] or (date,m.symbols[k]) in settle:
                continue
            p=pos.get(k)
            px,f=m.close[i,k],m.factor[i,k]
            if p:
                side,n=p["side"],p["n"]
                next_add=_price(p["latest"]+side*.5*n,f,side==1)
                if p["buys"]>=4 or side*(px-next_add)<-1e-8*f:
                    continue
                shares=floor(p["unit"]*f+1e-8)
            else:
                side,n=int(m.signal[i,k]),m.n[i,k]
                if not side or not np.isfinite(n) or n<=0:
                    continue
                shares=floor(notional*.01/(n/f))
            qty=shares/f
            available=cash-2*sum(v["qty"]*marks[j] for j,v in pos.items() if v["side"]<0)
            if shares<=0 or px*qty*(1+fee_rate)>available+1e-8 or not _room(m,i,k,side,pos):
                continue
            if p is None:
                p=dict(lots=[],side=side,qty=0.,unit=qty,buys=0,n=n)
                pos[k]=p
                ledger[k]=dict(net=0.,notional=0.)
                reason="entry"
            else:
                p["lots"]=[(_price(stop+side*.5*n,f,side<0),q) for stop,q in p["lots"]]
                reason="add"
            p["lots"].append((_price(px-side*2*n,f,side<0),qty))
            p["qty"]+=qty
            p["buys"]+=1
            p["latest"]=px
            record(k,side,True,qty,px,f,reason)
        long_value=sum(v["qty"]*marks[k] for k,v in pos.items() if v["side"]==1)
        short_value=sum(v["qty"]*marks[k] for k,v in pos.items() if v["side"]==-1)
        equity=cash+long_value-short_value
        assert np.isfinite(equity) and equity>0,(date,equity)
        daily.append((date,cash,equity,long_value,short_value,cash-2*short_value,len(pos)))

    equity_curve=pd.DataFrame(daily,columns=["date","cash_book","equity","long_value","short_value","free_collateral","held"])
    fills=pd.DataFrame(trades,columns=["date","symbol","side","reason","action","raw_price","raw_shares","fee_usd","cash_book_after"])
    closed_pnl=sum(x[0] for x in finished)
    open_pnl=sum(v["net"] for v in ledger.values())+long_value-short_value
    assert abs(initial_usd+closed_pnl+open_pnl-equity)<.01
    vals=equity_curve.equity
    wins=[r for v,r in finished if v>1e-8]
    losses=[r for v,r in finished if v<-1e-8]
    entry=fills.reason.eq("entry")
    summary=dict(year=year,top=top,initial=initial_usd,final=equity,ret=(equity/initial_usd-1)*100,
                 mdd=(vals/vals.cummax()-1).min()*100,entries=int(entry.sum()),
                 long_entries=int((entry&fills.side.eq(1)).sum()),short_entries=int((entry&fills.side.eq(-1)).sum()),
                 adds=int(fills.reason.eq("add").sum()),closed=len(finished),wins=len(wins),losses=len(losses),
                 win_rate=100*len(wins)/len(finished) if finished else None,
                 avg_win_pct=np.mean(wins) if wins else None,avg_loss_pct=np.mean(losses) if losses else None,
                 fees_usd=fees,borrow_usd=borrow,dividends_usd=dividends,held=len(pos),fills=len(fills),
                 estimated_terminal=missing_term,realized_usd=closed_pnl,open_pnl_usd=open_pnl,
                 first=str(m.dates[days[0]].date()),last=str(m.dates[days[-1]].date()),sessions=len(days))
    return summary,equity_curve,fills
