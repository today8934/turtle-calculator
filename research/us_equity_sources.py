import json,requests,io,zipfile,concurrent.futures
import pandas as pd
def get(item):
    name,url=item
    try:
        r=requests.get(url,timeout=35,stream=True,headers={'User-Agent':'US academic portfolio research/1.0'}); r.raise_for_status()
        if 'parquet' in url and 'datasets-server' not in url:
            df=pd.read_parquet(io.BytesIO(r.content));return (name,r.status_code,df.shape,list(df.columns),str(df.head(2).to_dict('records'))[:500])
        if 'tiingo' in url:
            z=zipfile.ZipFile(io.BytesIO(r.content)); df=pd.read_csv(z.open(z.namelist()[0]));return(name,len(df),list(df.columns),str(df[df.iloc[:,0].isin(['LEH','TWTR','AAPL','SPY','BSC','GM','BAC'])].to_dict('records'))[:1500])
        if 'dolthub' in url:
            return (name,r.status_code,r.text[:2500])
        if 'federalreserve' in url or 'fredgraph' in url:return(name,r.status_code,r.text[:150],r.text[-150:])
        if 'kibot' in url:return(name,r.status_code,r.text[:240],r.text[-100:])
        j=r.json()
        if 'datasets-server' in url:
            vals=j.get('parquet_files',[]);return name,len(vals),vals[:1],vals[-1:]
        if isinstance(j,list):return (name,len(j),[(a.get('path'),a.get('size')) for a in j[:35]])
        return (name,j.get('sha'),j.get('lastModified'),[(x.get('rfilename')) for x in j.get('siblings',[])[:26]])
    except Exception as e: return name,str(e)[:500]
urls={}
for dataset in ('HexQuant/Stocks-Daily-Price','AmirTrader/YahooFinance'):
    urls['hf_'+dataset]='https://huggingface.co/api/datasets/'+dataset
    urls['hfroot_'+dataset]='https://huggingface.co/api/datasets/'+dataset+'/tree/main?limit=50'
    urls['hfparq_'+dataset]='https://datasets-server.huggingface.co/parquet?dataset='+dataset
base='https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master?q='
from urllib.parse import quote
for key,sql in [('min','select * from ohlcv order by date,act_symbol limit 2'),('max','select * from ohlcv order by date desc,act_symbol desc limit 2'),('type','show tables'),('meta','select * from symbol limit 5'),('types','describe symbol')]:
    urls['dolt_'+key]=base+quote(sql)
urls.update(tiingo='https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip',
    fx='https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS&cosd=2008-01-01',
    last='https://marketparquet.com/api/data/download/stock_daily/2026-11-27.parquet',
    kibot='https://api.kibot.com/?action=history&symbol=AAPL&interval=daily&period=5&user=guest&password=guest')
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    for z in ex.map(get,urls.items()):print('SOURCE',json.dumps(z,default=str)[:4100],flush=True)
