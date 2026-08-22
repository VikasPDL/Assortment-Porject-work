import pandas as pd
from core import MATCH_KEYS, load_store_lookup, process_stock_file

store_codes, store_lookup = load_store_lookup("store detail.xlsx")
stock_grouped = process_stock_file("stock - 20-07-2026.xlsx", store_codes, store_lookup)
print("stock_grouped shape:", stock_grouped.shape)

base_stock_ref = pd.read_csv("reference_base_stock.csv")[MATCH_KEYS + ["BaseStock"]]
srp_ref = pd.read_csv("reference_srp.csv")[MATCH_KEYS + ["SRP"]]

position = stock_grouped.merge(base_stock_ref, on=MATCH_KEYS, how="left")
print("after base_stock merge, shape:", position.shape, "BaseStock notna:", position["BaseStock"].notna().sum())

position2 = position.merge(srp_ref, on=MATCH_KEYS, how="left")
print("after srp merge, shape:", position2.shape, "SRP notna:", position2["SRP"].notna().sum())
