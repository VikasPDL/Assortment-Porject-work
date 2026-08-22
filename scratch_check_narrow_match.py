import pandas as pd
from core import load_store_lookup, process_stock_file

store_codes, store_lookup = load_store_lookup()
stock_grouped = process_stock_file("stock - 20-07-2026.xlsx", store_codes, store_lookup)
srp = pd.read_csv("reference_srp.csv")
bs = pd.read_csv("reference_base_stock.csv")

narrow_key = ["Store Code", "Style No", "Category"]
m_srp = stock_grouped.merge(srp[narrow_key].drop_duplicates(), on=narrow_key, how="left", indicator=True)
print("SRP match rate, narrow key:", (m_srp["_merge"] == "both").mean())

m_bs = stock_grouped.merge(bs[narrow_key].drop_duplicates(), on=narrow_key, how="left", indicator=True)
print("BaseStock match rate, narrow key:", (m_bs["_merge"] == "both").mean())
