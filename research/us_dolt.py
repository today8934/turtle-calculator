"""Prepare the publicly maintained DoltHub US equity tape for a closing-price backtest."""
import argparse
import csv
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import duckdb


TABLES = ("symbol", "ohlcv", "split", "dividend")
DOLT_VERSION = "2.7.7"
REPOSITORY = "post-no-preference/stocks"


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def get_binary(cache):
    installed = shutil.which("dolt")
    if installed:
        return installed
    binary = cache / "bin" / "dolt"
    if binary.exists():
        return str(binary)
    if platform.system() != "Linux" or platform.machine() not in ("amd64", "x86_64"):
        raise RuntimeError("Install the Dolt CLI on this platform")
    binary.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/dolthub/dolt/releases/download/v{DOLT_VERSION}/dolt-linux-amd64.tar.gz"
    archive = cache / "dolt-cli.tar.gz"
    print(f"Downloading Dolt {DOLT_VERSION}", flush=True)
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive) as package:
        member = next(m for m in package.getmembers() if m.isfile() and Path(m.name).name == "dolt")
        with package.extractfile(member) as src, binary.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    binary.chmod(0o755)
    return str(binary)


def export_tables(cache):
    repo = cache / "stocks"
    dolt = get_binary(cache)
    if not (repo / ".dolt").exists():
        print("Cloning the latest stock database snapshot", flush=True)
        subprocess.run([dolt, "clone", "--depth", "1", REPOSITORY, str(repo)], check=True)
    revision = subprocess.check_output([dolt, "log", "-n", "1"], cwd=repo, text=True).splitlines()[0]
    for table in TABLES:
        path = cache / f"{table}.csv"
        if path.is_file() and path.stat().st_size > 100:
            continue
        print("Exporting", table, flush=True)
        tmp = cache / f"{table}.partial.csv"
        subprocess.run([dolt, "table", "export", "-f", table, str(tmp)], cwd=repo, check=True)
        tmp.replace(path)
    return revision


def prepare_dolt(cache, output, end):
    cache = Path(cache).resolve()
    cache.mkdir(exist_ok=True, parents=True)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    end = dt.date.fromisoformat(str(end)).isoformat()
    revision = export_tables(cache)
    db = duckdb.connect()
    db.execute("SET memory_limit='3GB'")
    db.execute("SET threads=3")
    db.execute("SET preserve_insertion_order=false")
    db.execute("SET temp_directory=" + sql_literal(cache / "duck-spill"))
    for table in TABLES:
        csv_path = sql_literal(cache / (table + ".csv"))
        # String schema avoids false numeric inference on tickers/names and the older
        # decimal precision in this tape. Explicit casts follow after classification.
        db.execute(f'CREATE VIEW "{table}_src" AS SELECT * FROM read_csv({csv_path}, all_varchar=true, header=true)')
    name_exclusions = (
        r"\b(warrants?|rights?|preferred|preference|debentures?|notes|units?|"
        r"etfs?|etns?|etvs?|funds?|closed.end|acquisition rights|exchange.traded)\b"
        r"|\b(proshares|direxion|ishares|spdr|graniteshares|yieldmax|tradr|"
        r"tuttle capital|rex shares|vaneck vectors|wisdomtree)\b"
    )
    db.execute(f"""
        CREATE TEMP TABLE master AS
        SELECT act_symbol ticker, security_name, listing_exchange,
          (NOT regexp_matches(lower(coalesce(security_name, '')), {sql_literal(name_exclusions)})
           AND upper(coalesce(is_etf, '')) IN ('0','N','NO','FALSE')
           AND upper(coalesce(is_test_issue, '')) NOT IN ('1','Y','YES','TRUE')
           AND regexp_matches(upper(coalesce(listing_exchange, '')), 'NASDAQ|NYSE|AMEX|NYSEMKT|BATS|CBOE|IEX|^([QNAPZV])$')
           AND security_name IS NOT NULL
          ) eligible
        FROM symbol_src
    """)
    print("Building classified, verified daily bars", flush=True)
    # FB was the Meta stock before the later ETF reused the ticker. META was
    # itself a former ETF ticker. Preserve this corporate rename across dates.
    db.execute(f"""
        CREATE TEMP TABLE bars AS
        WITH base AS (
          SELECT o.act_symbol ticker, try_cast(o.date AS DATE) date,
            try_cast(o.open AS DOUBLE) open, try_cast(o.high AS DOUBLE) high,
            try_cast(o.low AS DOUBLE) low, try_cast(o.close AS DOUBLE) close,
            try_cast(o.volume AS DOUBLE) volume, coalesce(m.eligible, false) eligible
          FROM ohlcv_src o LEFT JOIN master m ON o.act_symbol=m.ticker
        )
        SELECT *, CASE WHEN ticker='FB' AND date BETWEEN DATE '2012-05-18' AND DATE '2022-06-08'
                       THEN 'META'
                       WHEN ticker='DELL' AND date < DATE '2014-01-01' THEN 'DELL~2013'
                       ELSE replace(ticker, '.', '-') END AS symbol
        FROM base
        WHERE date BETWEEN DATE '2011-01-03' AND DATE '{end}'
          AND open>0 AND low>0 AND close>0 AND high>=low
          AND high>=close AND low<=close AND high>=open AND low<=open AND volume>=0
          AND (CASE WHEN ticker='FB' THEN date BETWEEN DATE '2012-05-18' AND DATE '2022-06-08'
                    WHEN ticker='META' THEN date>=DATE '2022-06-09'
                    ELSE eligible END)
    """)
    # A listed split may be stale or a vendor may have pre-adjusted an interval.
    # Check the opening discontinuity over the actual effective trading day.
    # Also allow a 3-session reporting offset. The event must explain the move
    # better than leaving prices alone. No unreported split is invented.
    db.execute(f"""
      CREATE TEMP TABLE actions AS
      WITH known AS (
        SELECT act_symbol ticker, CAST(ex_date AS DATE) ex_date,
          try_cast(to_factor AS DOUBLE)/nullif(try_cast(for_factor AS DOUBLE),0) ratio
        FROM split_src WHERE CAST(ex_date AS DATE)<=DATE '{end}'
      ), candidate AS (
        SELECT k.ticker, k.ex_date, k.ratio, b.date day, b.symbol,
               b.open, prev.close prior, abs(ln(prev.close/b.open/k.ratio)) error,
               abs(ln(prev.close/b.open)) unchanged_error
        FROM known k JOIN bars b
          ON b.ticker=k.ticker AND b.date BETWEEN k.ex_date-4 AND k.ex_date+4
        ASOF LEFT JOIN bars prev ON prev.symbol=b.symbol AND prev.date<b.date
        WHERE ratio>0 AND ratio<>1 AND prev.close>0 AND b.open>0
      ), selected AS (
        SELECT * FROM candidate
        WHERE error<unchanged_error AND error<least(0.40,abs(ln(ratio))*0.6)
        QUALIFY row_number() OVER (PARTITION BY ticker,ex_date ORDER BY
          CASE WHEN day=ex_date AND error<least(0.16,abs(ln(ratio))*0.25) THEN 0 ELSE 1 END,
          error, abs(day-ex_date), day)=1
      )
      SELECT symbol,day, exp(sum(ln(ratio))) ratio FROM selected GROUP BY symbol,day
    """)
    db.execute("""
      CREATE TEMP TABLE factors AS
      SELECT symbol,day,exp(-sum(ln(ratio)) OVER
         (PARTITION BY symbol ORDER BY day DESC ROWS UNBOUNDED PRECEDING)) factor
      FROM actions
    """)
    db.execute("""
      CREATE TEMP TABLE distributions AS
      SELECT act_symbol ticker, CAST(ex_date AS DATE) ex_date, sum(try_cast(amount AS DOUBLE)) amount
      FROM dividend_src GROUP BY ticker,ex_date
    """)
    partial = output.with_suffix(".writing.parquet")
    print("Writing split-consistent daily prices", flush=True)
    db.execute(f"""
      COPY (
        SELECT b.date, b.symbol, b.high*coalesce(f.factor,1) high,
          b.low*coalesce(f.factor,1) low,b.close*coalesce(f.factor,1) close,
          b.close raw_close,b.close*b.volume turnover,coalesce(d.amount,0) dividend,true eligible
        FROM bars b ASOF LEFT JOIN factors f ON b.symbol=f.symbol AND b.date<f.day
        LEFT JOIN distributions d ON b.ticker=d.ticker AND b.date=d.ex_date
      ) TO {sql_literal(partial)} (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    view = "read_parquet(" + sql_literal(partial) + ")"
    stats = db.execute(f"SELECT count(*),count(DISTINCT symbol),min(date),max(date),count(DISTINCT date) FROM {view}").fetchone()
    duplicates = db.execute(f"SELECT date,symbol,count(*) FROM {view} GROUP BY date,symbol HAVING count(*)>1 LIMIT 3").fetchall()
    if duplicates:
        raise RuntimeError(f"Duplicate securities after ticker normalization: {duplicates}")
    if stats[1]<2000 or stats[4]<100:
        raise RuntimeError(f"Implausible Dolt coverage {stats}")
    checks = db.execute("SELECT symbol,day,ratio FROM actions WHERE (symbol='AAPL' AND year(day) IN (2014,2020)) OR (symbol='NVDA' AND year(day)=2024) ORDER BY day").fetchall()
    if dt.date.fromisoformat(end) >= dt.date(2024,6,10):
        for stock, expected_date, ratio in (("AAPL","2014-06-09",7),("AAPL","2020-08-31",4),("NVDA","2024-06-10",10)):
            if not any(s==stock and str(d)==expected_date and abs(r-ratio)<0.001 for s,d,r in checks):
                raise RuntimeError(f"Split validation failed for {stock} {expected_date}. Observed {checks}")
    partial.replace(output)
    metrics = dict(revision=revision, rows=stats[0],symbols=stats[1],start=str(stats[2]),end=str(stats[3]),
                   sessions=stats[4], splits=db.execute("SELECT count(*) FROM actions").fetchone()[0],
                   example_splits=checks)
    output.with_suffix(".metadata.json").write_text(json.dumps(metrics,indent=2,default=str))
    print(json.dumps(metrics,default=str),flush=True)
    db.close()
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--cache", default="us-cache/dolt")
    p.add_argument("--end", default="2026-11-27")
    args = p.parse_args()
    prepare_dolt(args.cache, args.output, args.end)
