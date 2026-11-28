"""HTTP download and SQL literal helpers for daily price verification."""
from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import time


HEX_BASE = ("https://huggingface.co/datasets/HexQuant/Stocks-Daily-Price/resolve/"
            "cdc5ccc0a7b34bd0a5969a538ed49939051969e9/data")


def fetch(url, path):
    path = Path(path)
    if path.is_file() and path.stat().st_size:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    for attempt in range(4):
        try:
            request = Request(url, headers={"User-Agent": "US-equity-history-research/1.0"})
            with urlopen(request, timeout=150) as response, part.open("wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
                length = response.headers.get("Content-Length")
            if length and part.stat().st_size != int(length):
                raise IOError(f"Partial download of {path.name}")
            part.replace(path)
            return path
        except (OSError, TimeoutError):
            part.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def sqlval(value):
    return "'" + str(value).replace("'", "''") + "'"


def sqlpaths(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return "[" + ",".join(sqlval(Path(p).as_posix()) for p in paths) + "]"


