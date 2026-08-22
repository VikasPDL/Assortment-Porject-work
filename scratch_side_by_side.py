import pandas as pd
from core import MATCH_KEYS, load_store_lookup, process_stock_file, stringify_keys
from allocation import build_stock_position

# Path A: call the real function
position_a = build_stock_position("stock - 20-07-2026.xlsx")
print("A) via build_stock_position - SRP notna:", position_a["SRP"].notna().sum())

# Path B: inline the same steps manually, right here
store_codes, store_lookup = load_store_lookup("store detail.xlsx")
stock_grouped = process_stock_file("stock - 20-07-2026.xlsx", store_codes, store_lookup)
stock_grouped = stringify_keys(stock_grouped, MATCH_KEYS)

base_stock_ref = pd.read_csv("reference_base_stock.csv")[MATCH_KEYS + ["BaseStock"]]
base_stock_ref = stringify_keys(base_stock_ref, MATCH_KEYS)
srp_ref = pd.read_csv("reference_srp.csv")[MATCH_KEYS + ["SRP"]]
srp_ref = stringify_keys(srp_ref, MATCH_KEYS)

position_b = stock_grouped.merge(base_stock_ref, on=MATCH_KEYS, how="left")
position_b = stringify_keys(position_b, MATCH_KEYS)
position_b = position_b.merge(srp_ref, on=MATCH_KEYS, how="left")
print("B) inline copy - SRP notna:", position_b["SRP"].notna().sum())

print("Are position_a and position_b equal?", position_a.equals(position_b))
print("position_a shape:", position_a.shape, "position_b shape:", position_b.shape)
