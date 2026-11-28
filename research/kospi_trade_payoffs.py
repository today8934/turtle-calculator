"""Summarize completed campaigns from the archived fills of run 46114743756."""
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean


SOURCE = Path("transactions")
OUT = Path("payoffs")
OUT.mkdir(exist_ok=True)
BASE = 100_000_000
EPS = .00001


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(name, rows):
    with (OUT / name).open("w",encoding="utf-8-sig",newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def stats(path,policy,top,year,closed,opened,open_pnl,portfolio_pct):
    wins = [x for x in closed if x["net_krw"] > EPS]
    losses = [x for x in closed if x["net_krw"] < -EPS]
    def average(rows,key):
        return mean(r[key] for r in rows) if rows else None
    return dict(path=path,policy=policy,top=top,year=year,closed=len(closed),wins=len(wins),
                losses=len(losses),flat=len(closed)-len(wins)-len(losses),
                win_rate=100*len(wins)/len(closed) if closed else None,
                avg_win_pct=average(wins,"trade_pct"),avg_loss_pct=average(losses,"trade_pct"),
                avg_win_krw=average(wins,"net_krw"),avg_loss_krw=average(losses,"net_krw"),
                closed_pnl_krw=sum(r["net_krw"] for r in closed),open_count=opened,
                open_pnl_krw=open_pnl,portfolio_pct=portfolio_pct)


def summarize(base,key,old):
    path,policy,top,year = key
    fills = read(SOURCE / f"{base}_fills.csv")
    day = read(SOURCE / f"{base}_daily.csv")[-1]
    reported = defaultdict(float)
    for r in read(SOURCE / f"{base}_stock_profit.csv"):
        reported[(r["code"],int(r.get("side",1)))] += float(r["profit_krw"])
    live,completed,settled = {},[],defaultdict(float)
    book = float(BASE)
    for f in fills:
        code, why = f["code"], f["reason"]
        side = int(f.get("side",1))
        opening = why in ("entry","add")
        buy = f["action"]=="buy"
        assert buy == ((side==1)==opening)
        if why == "entry":
            assert code not in live
            live[code] = dict(lots=0,side=side,code=code,notional=0.,net_krw=0.,entry_date=f["date"])
        t = live[code]
        assert t["side"]==side
        amount = float(f["raw_price"])*float(f["raw_shares"])
        assert amount>0
        flow = amount*(-1.0005 if buy else .9975)
        t["net_krw"] += flow
        book += flow
        assert abs(book-float(f.get("cash_book_after") or f["cash_after"]))<.02
        if opening:
            t["lots"] += 1
            t["notional"] += amount
        elif why == "merger_recall":
            t["lots"] = 0
        else:
            assert why in ("2N","10D"), why
            t["lots"] -= 1
        assert 0<=t["lots"]<=4
        if not t["lots"]:
            t["exit_date"] = f["date"]
            t["trade_pct"] = 100*t["net_krw"]/t["notional"]
            settled[(code,side)] += t["net_krw"]
            completed.append(t)
            del live[code]
    if policy=="long":
        market_value = float(day["equity"])-float(day["cash"])
        held = int(day["holdings"])
    else:
        market_value = float(day["long_value"])-float(day["short_value"])
        held = int(day["held"])
    assert len(live)==held
    assert abs(book+market_value-float(day["equity"]))<.02
    open_pnl = market_value + sum(t["net_krw"] for t in live.values())
    live_keys = {(code,t["side"]) for code,t in live.items()}
    for k in (set(reported)|set(settled))-live_keys:
        assert abs(reported[k]-settled[k])<.02
    assert abs(sum(reported[k]-settled[k] for k in live_keys)-open_pnl)<.02
    gain = float(day["equity"])-BASE
    assert abs(sum(settled.values())+open_pnl-gain)<.02
    row = stats(path,policy,top,year,completed,held,open_pnl,gain/1e6)
    assert row["closed"]==int(old["fully_closed_positions"])
    assert row["losses"]==int(old["losing_closed_positions"])
    assert abs(row["portfolio_pct"]-float(old["intraday_return_pct"]))<.00006
    return completed,row


def rounded(r):
    return {k: (round(v,6) if isinstance(v,float) else v) for k,v in r.items()}


previous = read(Path("research/results/kospi_system1_intraday_2010_2019.csv"))
assert len(previous)==180
annual,pooled,all_closed = [],[],[]
groups = defaultdict(list)
for old in previous:
    path,policy = old["intraday_path"],old["position_policy"]
    top,year = int(old["turnover_top"]),int(old["year"])
    base = f"{path}_{policy}_{year}_{top}"
    completed,row = summarize(base,(path,policy,top,year),old)
    annual.append(row)
    groups[(path,policy,top)].extend(completed)
    for t in completed:
        all_closed.append(dict(path=path,policy=policy,top=top,year=year,
                               **{k:v for k,v in t.items() if k!="lots"}))
for (path,policy,top),closed in sorted(groups.items()):
    a=[r for r in annual if (r["path"],r["policy"],r["top"])==(path,policy,top)]
    assert len(a)==10
    pooled.append(stats(path,policy,top,"all",closed,sum(r["open_count"] for r in a),
                        sum(r["open_pnl_krw"] for r in a),mean(r["portfolio_pct"] for r in a)))
for name,rows in (("win_metrics.csv",annual),("win_pooled.csv",pooled),("completed_trades.csv",all_closed)):
    write(name,[rounded(r) for r in rows])
import json
for r in annual+pooled:
    print("PAYOFF_ROW",json.dumps(list(rounded(r).values()),separators=(",",":")))
print("PAYOFF_COLUMNS",",".join(annual[0]))
print("RECONCILED",len(annual),len(pooled),len(all_closed))
