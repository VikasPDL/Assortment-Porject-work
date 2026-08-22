import pandas as pd
from core import MATCH_KEYS, load_store_lookup, process_stock_file

print("MATCH_KEYS:", MATCH_KEYS)

store_codes, store_lookup = load_store_lookup()
stock_grouped = process_stock_file("stock - 20-07-2026.xlsx", store_codes, store_lookup)
print("stock_grouped columns:", list(stock_grouped.columns))
print("stock_grouped shape:", stock_grouped.shape)

srp_ref = pd.read_csv("reference_srp.csv")[MATCH_KEYS + ["SRP"]]
print("srp_ref shape:", srp_ref.shape)
print("srp_ref dup keys:", srp_ref.duplicated(subset=MATCH_KEYS).sum())

merged = stock_grouped.merge(srp_ref, on=MATCH_KEYS, how="left", indicator=True)
print("merge indicator counts:")
print(merged["_merge"].value_counts())

# check dtypes
for k in MATCH_KEYS:
    print(k, "stock dtype:", stock_grouped[k].dtype, "ref dtype:", srp_ref[k].dtype)
