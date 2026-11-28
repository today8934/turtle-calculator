"""Cash-settle verified pro-rata share distributions at the first ex-date close."""
from pathlib import Path

import pandas as pd


# First regular ex-distribution session, continuing parent, distributed security,
# shares per one parent share, issuer or clearing-house notice.
EVENTS = [
    # For MSI the ratio is per post-reverse-split share (7/8 MMI).
    ("2011-01-04","MSI","MMI",7/8,"https://www.motorolasolutions.com/content/dam/msi/investors/doc_downloads/StockOwnership_CompanyActions/msi-and-mmi-shareholder-tax-basis-information-and-irs-form-8937.pdf"),
    ("2011-07-01","MRO","MPC",.5,"https://www.sec.gov/Archives/edgar/data/101778/000119312511151791/dex991.htm"),
    ("2012-05-01","COP","PSX",.5,"https://infomemo.theocc.com/infomemos?number=30483"),
    ("2012-10-02","MDLZ","KRFT",1/3,"https://www.mondelezinternational.com/investors/stock/spin-off-information/"),
    ("2013-01-02","ABT","ABBV",1,"https://infomemo.theocc.com/infomemos?number=31854"),
    ("2014-04-03","GOOGL","GOOG",1,"https://s206.q4cdn.com/479360582/files/doc_downloads/additional/2014-form8937.pdf"),
    ("2014-06-09","TWX","TIME",1/8,"https://www.sec.gov/Archives/edgar/data/1105705/000110570514000031/exhibit991.htm"),
    ("2015-07-01","BAX","BXLT",1,"https://www.sec.gov/Archives/edgar/data/10456/000119312515246136/d57625d8k.htm"),
    ("2015-07-20","EBAY","PYPL",1,"https://www.sec.gov/Archives/edgar/data/1065088/000119312515257121/d93601d8k.htm"),
    ("2015-11-02","HPQ","HPE",1,"https://investor.hp.com/news-events/news/news-details/2015/HP-Board-of-Directors-Approves-Separation/default.aspx"),
    ("2016-11-01","YUM","YUMC",1,"https://www.sec.gov/Archives/edgar/data/1673358/000104746916016421/a2230169z8-k.htm"),
    ("2017-04-03","HPE","DXC",.085904,"https://www.sec.gov/Archives/edgar/data/23082/000119312517113607/d371672d8k.htm"),
    ("2019-04-02","DD","DOW",1/3,"https://investors.dow.com/en/news/news-details/2019/Dow-completes-separation-from-DowDuPont/default.aspx"),
    ("2019-06-03","DD","CTVA",1,"https://www.sec.gov/Archives/edgar/data/1755672/000175567219000014/corteva-630201910xquse.htm"),
    ("2019-02-26","GE","WAB",.005371,"https://infomemo.theocc.com/infomemos?number=44621"),
    ("2020-04-03","RTX","CARR",1,"https://www.sec.gov/Archives/edgar/data/101829/000114036120008397/nc10010681x2_8k.htm"),
    ("2020-04-03","RTX","OTIS",.5,"https://www.sec.gov/Archives/edgar/data/101829/000114036120008397/nc10010681x2_8k.htm"),
    ("2020-11-17","PFE","VTRS",.1240788849,"https://infomemo.theocc.com/infomemos?number=47861"),
    ("2021-06-03","MRK","OGN",.1,"https://www.merck.com/news/merck-announces-completion-of-organon-co-spinoff/"),
    ("2021-08-02","XPO","GXO",1,"https://news.xpo.com/2667/xpo-logistics-spin-off-gxo-logistics-to-ring-nyse-opening-bell/"),
    ("2021-11-02","DELL","VMW",.440626,"https://infomemo.theocc.com/infomemos?number=49493"),
    ("2021-11-04","IBM","KD",.2,"https://newsroom.ibm.com/2021-11-03-IBM-Completes-the-Separation-of-Kyndryl"),
    ("2022-04-11","T","WBD",.241917,"https://investors.att.com/stockholder-services/cost-basis-guide/worksheet/atandt-inc-wbd"),
    ("2022-08-22","AMC","APE",1,"https://investor.amctheatres.com/sec-filings/all-sec-filings/content/0001104659-22-092397/0001104659-22-092397.pdf"),
    ("2022-11-01","XPO","RXO",1,"https://investors.xpo.com/news-releases/news-release-details/xpo-completes-spin-rxo"),
    ("2023-01-04","GE","GEHC",1/3,"https://www.ge.com/sites/default/files/ge-healthcare-distribution-ratio-faqs.pdf"),
    ("2023-08-23","AMC","AMC",1/7.5,"https://infomemo.theocc.com/infomemos?number=53048"),
    ("2024-04-01","MMM","SOLV",.25,"https://investors.3m.com/news-events/press-releases/detail/1835/3m-completes-spin-off-of-solventum"),
    ("2024-04-02","GE","GEV",.25,"https://www.geaerospace.com/ko/node/5116"),
    ("2025-02-24","WDC","SNDK",1/3,"https://www.sec.gov/Archives/edgar/data/106040/000010604025000012/wdc8-khardxspin.htm"),
    ("2025-10-30","HON","SOLS",.25,"https://honeywell.gcs-web.com/news-releases/news-release-details/honeywell-completes-spin-solstice-advanced-materials"),
    ("2025-11-03","DD","Q",.5,"https://www.investors.dupont.com/news-and-media/press-release-details/2025/DuPont-Completes-Separation-of-Qnity-Electronics/default.aspx"),
    ("2026-01-05","CMCSA","VSNT",1/25,"https://www.sec.gov/Archives/edgar/data/1166691/000095010326000079/dp239108_ex9901.htm"),
    ("2026-06-29","HON","HONA",.5,"https://www.sec.gov/Archives/edgar/data/773840/000077384026000084/exhibit991-pressrelease629.htm"),
]


def apply_spinoffs(frame, dolt_parquet, audit_path=None):
    import duckdb
    quotes = frame.copy()
    quotes["date"] = pd.to_datetime(quotes.date)
    # Historical share identifiers were unified before computing the daily rankings.
    requests = pd.DataFrame(EVENTS,columns=["date","parent","child","shares","source"])
    requests["date"] = pd.to_datetime(requests.date)
    db = duckdb.connect()
    db.register("event_request",requests)
    db.read_parquet(str(dolt_parquet)).create_view("price_data")
    child_prices = db.execute("""
        SELECT r.date,r.child,p.raw_close FROM event_request AS r
        LEFT JOIN price_data AS p ON p.symbol=r.child AND p.date=r.date
    """).fetchdf()
    # The raw full tape can include a since-delisted child missing from the
    # security master. Reading the ticker/date is legitimate even if ineligible for entry.
    missing = child_prices[child_prices.raw_close.isna()].copy()
    raw_file = Path(dolt_parquet).parent / "ohlcv.csv"
    if not missing.empty and raw_file.is_file():
        db.register("missing",missing[["date","child"]])
        raw = db.read_csv(str(raw_file), all_varchar=True)
        raw.create_view("raw_tape")
        extra = db.execute("""
          SELECT r.date,r.child,try_cast(p.close AS DOUBLE) AS raw_close
          FROM missing AS r JOIN raw_tape AS p
            ON p.act_symbol=r.child AND try_cast(p.date AS DATE)=r.date
        """).fetchdf()
        child_prices=pd.concat([child_prices.dropna(subset=["raw_close"]),extra],ignore_index=True)
    db.close()
    values = {(pd.Timestamp(d),s):float(v) for d,s,v in child_prices.itertuples(index=False)
              if pd.notna(v) and v>0}
    # Independently reported unadjusted first regular closes for these two
    # securities, including the delisted MMI no longer present in some masters.
    values.setdefault((pd.Timestamp("2011-01-04"),"MMI"),33.12)
    values.setdefault((pd.Timestamp("2014-04-03"),"GOOG"),569.74)
    values.setdefault((pd.Timestamp("2022-08-22"),"APE"),6.00)
    return settle_distributions(quotes,requests,values,audit_path)


def settle_distributions(quotes,events,child_prices,audit_path=None):
    quotes["spin_cash"] = 0.
    quotes["spin_ratio"] = 1.
    ledger=[]
    for (date,parent), group in events.groupby(["date","parent"],sort=True):
        mask = quotes.symbol.eq(parent)
        before = mask & quotes.date.lt(date)
        event = mask & quotes.date.eq(date)
        status = "applied"
        if not mask.any():
            status="outside_ranked_symbol_history"
        elif event.sum()!=1 or not before.any():
            status="no_continuous_parent_history"
        prices=[child_prices.get((date,s)) for s in group.child]
        if status=="applied" and not all(p is not None and p>0 for p in prices):
            status="missing_child_close"
        cash = a = existing = None
        if status=="applied":
            prior=quotes.loc[before].sort_values("date").iloc[-1]
            now=quotes.loc[event].iloc[0]
            cash=sum(float(p)*float(count) for p,count in zip(prices,group.shares))
            a=now.raw_close/(now.raw_close+cash)
            # Undo any pseudo-split already present for this same distribution.
            # Motorola and DuPont reverse-split the retained parent shares.
            existing=(prior.close/prior.raw_close)/(now.close/now.raw_close)
            same_day_reverse={("MSI","2011-01-04"):7.,("DD","2019-06-03"):3.}
            reverse_split=same_day_reverse.get((parent,str(date.date())),1.)
            if not (0<a<1 and .001<existing<1000):
                raise ValueError(f"Invalid spin-off adjustment {parent} {date}: {a}, {existing}")
            cols=["high","low","close"]
            if "open" in quotes:
                cols.append("open")
            quotes.loc[before,cols]=quotes.loc[before,cols]*(reverse_split*a/existing)
            quotes.loc[event,"spin_cash"]=cash
            quotes.loc[event,"spin_ratio"]=a
        if parent=="EBAY" and mask.any() and status!="applied":
            raise ValueError(f"Cannot value PayPal separation: {status}, {prices}")
        for row,px in zip(group.itertuples(index=False),prices):
            ledger.append(dict(date=str(date.date()),parent=parent,child=row.child,shares=row.shares,
                               child_raw_close=px,cash_per_parent=cash,spin_ratio=a,
                               upstream_factor_ratio=existing,status=status,source=row.source))
    audit=pd.DataFrame(ledger)
    if audit_path is not None:
        audit.to_csv(audit_path,index=False)
    print("CORPORATE_ACTIONS",audit.to_json(orient="records"),flush=True)
    return quotes
