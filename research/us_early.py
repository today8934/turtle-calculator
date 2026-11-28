"""Load the frozen 3,199-name WIKI US-equity history as an early-market supplement."""
import argparse
from pathlib import Path
import re
from shutil import copyfileobj
import tempfile
from zipfile import ZipFile

import numpy as np
import pandas as pd


SLUG = "marketneutral/quandl-wiki-prices-us-equites"
DOWNLOAD_URLS = (
    f"https://api.kaggle.com/v1/datasets/download/{SLUG}",
    f"https://www.kaggle.com/api/v1/datasets/download/{SLUG}",
)
FIELDS = {"symbol", "date", "open", "high", "low", "close", "volume", "dividend", "split_ratio"}
PRICES = ["open", "high", "low", "close"]


def _canonical(name):
    key = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    return {"ticker": "symbol", "timestamp": "date", "ex_dividend": "dividend",
            "stock_splits": "split_ratio", "splitratio": "split_ratio"}.get(key, key)


def _download(cache):
    import requests
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / "quandl-wiki-prices-us-equites.zip"
    if target.is_file():
        return target
    failures = []
    for url in DOWNLOAD_URLS:
        tmp = None
        try:
            with requests.get(url, stream=True, timeout=(12, 100)) as resp:
                resp.raise_for_status()
                chunks = resp.iter_content(1024 * 1024)
                first = next(chunks, b"")
                if not first.startswith(b"PK\x03\x04"):
                    raise ValueError("Response is not the public dataset ZIP")
                with tempfile.NamedTemporaryFile(dir=cache, delete=False) as out:
                    tmp = Path(out.name)
                    out.write(first)
                    for chunk in chunks:
                        out.write(chunk)
            with ZipFile(tmp) as archive:
                if not archive.namelist():
                    raise ValueError("Empty dataset ZIP")
            tmp.replace(target)
            return target
        except (OSError, ValueError, requests.RequestException) as exc:
            failures.append(f"{url}: {exc}")
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
    raise RuntimeError("WIKI download failed: " + " | ".join(failures))


def _prices_file(cache, source):
    if source.suffix.lower() != ".zip":
        return source
    with ZipFile(source) as zipped:
        members = [item for item in zipped.infolist() if
                   Path(item.filename).suffix.lower() in {".csv", ".parquet", ".pq"}
                   and not item.filename.startswith("__MACOSX/")]
        if not members:
            raise ValueError(f"No price table found in {zipped.namelist()}")
        member = max(members, key=lambda item: item.file_size)
        target = cache / ("wiki-price-table" + Path(member.filename).suffix.lower())
        if target.exists() and target.stat().st_size == member.file_size:
            return target
        with zipped.open(member) as inp, tempfile.NamedTemporaryFile(dir=cache, delete=False) as out:
            tmp = Path(out.name)
            try:
                copyfileobj(inp, out, 4 * 1024 * 1024)
            except BaseException:
                out.close()
                tmp.unlink(missing_ok=True)
                raise
        tmp.replace(target)
        return target


def _batches(source):
    if source.suffix.lower() in {".pq", ".parquet"}:
        import pyarrow.parquet as pq
        pqf = pq.ParquetFile(source)
        columns = [col for col in pqf.schema_arrow.names if _canonical(col) in FIELDS]
        for chunk in pqf.iter_batches(batch_size=250_000, columns=columns):
            yield chunk.to_pandas(ignore_metadata=True)
    else:
        yield from pd.read_csv(source, chunksize=250_000, usecols=lambda col: _canonical(col) in FIELDS)


def _read_window(source, start, end):
    parts = []
    for rows in _batches(source):
        rows = rows.rename(columns=_canonical)
        missing = FIELDS.difference(rows.columns)
        if missing:
            raise ValueError(f"Missing raw fields in {source.name}: {sorted(missing)}")
        rows["date"] = pd.to_datetime(rows["date"]).dt.normalize()
        rows = rows[rows.date.between(start, end)]
        if not rows.empty:
            parts.append(rows.copy())
    if not parts:
        raise ValueError(f"No WIKI price records from {start} through {end}")
    return pd.concat(parts, ignore_index=True)


def _split_prices(data):
    data = data.copy()
    data["symbol"] = (data.symbol.astype(str).str.upper().str.strip()
                      .str.replace(r"^WIKI/", "", regex=True)
                      .str.replace(r"[._/]", "-", regex=True))
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)
    if data.duplicated(["symbol", "date"]).any():
        raise ValueError("Duplicate (symbol, date) in WIKI")
    for col in PRICES + ["volume", "dividend", "split_ratio"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
    data["split_ratio"] = data.split_ratio.fillna(1.)
    data["dividend"] = data.dividend.fillna(0.)
    if not (np.isfinite(data.split_ratio) & data.split_ratio.gt(0)).all():
        raise ValueError("Invalid WIKI split ratio")
    # The split-date bar is already in post-split shares. Adjust earlier rows.
    growth = data.groupby("symbol", sort=False).split_ratio.cumprod()
    basis = growth / growth.groupby(data.symbol, sort=False).transform("last")
    data["raw_close"] = data.close
    data["raw_volume"] = data.volume
    data["turnover"] = data.raw_close * data.raw_volume
    data[PRICES] = data[PRICES].multiply(basis, axis=0)
    data["volume"] = data.raw_volume / basis
    # WIKI/PRICES is a US stock dataset, not an ETF universe. A separate
    # point-in-time security master, where available, can tighten eligibility.
    data["eligible"] = True
    mask = (np.isfinite(data[PRICES]).all(axis=1) & data[PRICES].min(axis=1).gt(0)
            & data.volume.ge(0) & np.isfinite(data.volume))
    return data[mask].reset_index(drop=True)


def load_wiki(cache, start="2008-01-01", end="2011-06-30", source=None):
    """Read date-inclusive raw trades and split-only OHLCV in end-date share units.

    Returns date,symbol,open,high,low,close,volume,raw_close,raw_volume,turnover,
    dividend,split_ratio,eligible. turnover is estimated by raw close times raw
    share volume. Dividend is dollars per historical share on its ex-date.
    Pass an existing WIKI CSV, Parquet or Kaggle ZIP as source to work offline.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not start <= end <= pd.Timestamp("2018-03-27"):
        raise ValueError("WIKI requires start <= end <= 2018-03-27")
    out = cache / f"wiki-split-only-v1-{start.date()}-{end.date()}.parquet"
    if source is None and out.exists():
        return pd.read_parquet(out)
    original = Path(source) if source else _download(cache)
    table = _prices_file(cache, original)
    frame = _split_prices(_read_window(table, start, end))
    if source is None:
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as handle:
            tmp = Path(handle.name)
        try:
            frame.to_parquet(tmp, index=False)
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--start", default="2008-01-01")
    parser.add_argument("--end", default="2011-06-30")
    parser.add_argument("--out", type=Path, required=True)
    opts = parser.parse_args()
    prices = load_wiki(opts.cache, opts.start, opts.end, opts.source)
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(opts.out, index=False)
    print(f"WIKI: {len(prices):,} valid bars, {prices.symbol.nunique():,} names, "
          f"{prices.date.min().date()} to {prices.date.max().date()}", flush=True)
