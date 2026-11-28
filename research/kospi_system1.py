import pandas as pd
from pathlib import Path
for p in sorted(Path("market").glob("*.parquet")):
    d = pd.read_parquet(p)
    print(p.name, d.shape, d.index.names, d.columns.tolist(), d.tail(2).to_string(index=True), flush=True)
    print(d.Market.value_counts().to_dict())
