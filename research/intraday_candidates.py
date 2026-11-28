"""Identify historic candidates by pre-trade turnover and provide contemporaneous issuer names."""
import numpy as np
import pandas as pd
from kospi_system1 import market_data

def main():
    index, stocks, ranks, names, _ = market_data(range(2008,2015), pd.Timestamp("2009-12-30"), pd.Timestamp("2013-11-13"))
    selected = ranks.loc["2010":].le(30).any()
    ids = selected[selected].index
    inside = stocks.loc[(stocks.Date >= "2009-12-30") & stocks.Code.isin(ids)]
    for subset in np.array_split(ids,5):
        print("CANDIDATES",inside.loc[inside.Code.isin(subset)].drop_duplicates(["Code","Name"])[["Code","Name"]].to_json(orient="values",force_ascii=False))
if __name__ == "__main__":
    main()
