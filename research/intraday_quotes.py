"""Inspect public time-stamped KRX chart data before using it to simulate domestic equities."""
import json
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
from zoneinfo import ZoneInfo


def fetch_json(url):
    with urlopen(Request(url, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"}),timeout=20) as res:
        return json.load(res)


def inspect_sources():
    start = datetime(2025,11,28,tzinfo=ZoneInfo("Asia/Seoul"))
    end = datetime(2026,11,28,tzinfo=ZoneInfo("Asia/Seoul"))
    urls = {
        "toss": "https://wts-info-api.tossinvest.com/api/v1/c-chart/kr-s/A005930/min:30?"+urlencode(
             {"count":200,"session":"all","investMode":"krx","useAdjustedRate":"false","to":"2025-12-19T15:30:00+09:00"}),
        "naver": "https://api.stock.naver.com/chart/domestic/item/005930/minute10?startDateTime=202512190900&endDateTime=202512191530",
        "yahoo":"https://query1.finance.yahoo.com/v8/finance/chart/005930.KS?"+urlencode(
             {"interval":"60m","period1":int(start.timestamp()),"period2":int(end.timestamp())}),
    }
    for name,url in urls.items():
        try:
            obj = fetch_json(url)
            if name=="yahoo":
                rows = obj["chart"]["result"][0]["timestamp"]
                print("SOURCE",name,len(rows),[datetime.fromtimestamp(t,ZoneInfo("Asia/Seoul")).isoformat() for t in (rows[0],rows[-1])])
            elif name=="toss":
                data = obj.get("result",obj)
                print("SOURCE",name,str(data)[:950],str(data)[-550:])
            else:
                print("SOURCE",name,str(obj)[:800],str(obj)[-400:])
        except HTTPError as err:
            print("SOURCE",name,err.code,err.read(350))
        except (OSError,ValueError,KeyError,TypeError) as err:
            print("SOURCE",name,str(err)[:300])

