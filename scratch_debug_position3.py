import pandas as pd
from core import MATCH_KEYS, load_store_lookup, process_stock_file

store_codes, store_lookup = load_store_lookup("store detail.xlsx")
stock_grouped = process_stock_file("stock - 20-07-2026.xlsx", store_codes, store_lookup)

base_stock_ref = pd.read_csv("reference_base_stock.csv")[MATCH_KEYS + ["BaseStock"]]
srp_ref = pd.read_csv("reference_srp.csv")[MATCH_KEYS + ["SRP"]]

position = stock_grouped.merge(base_stock_ref, on=MATCH_KEYS, how="left")

print("dtypes after first merge:")
for k in MATCH_KEYS:
    print(" ", k, position[k].dtype)

print("dtypes of srp_ref:")
for k in MATCH_KEYS:
    print(" ", k, srp_ref[k].dtype)

# try merging again explicitly with indicator
m2 = position.merge(srp_ref, on=MATCH_KEYS, how="left", indicator=True)
print(m2["_merge"].value_counts())

# check for exact-duplicate MATCH_KEYS rows within position (post first merge) that could cause issues
print("dup rows in position (post base merge) on MATCH_KEYS:", position.duplicated(subset=MATCH_KEYS).sum())
print("total rows in position:", len(position))

# check whether "Category" column values match set-wise
print("Category dtype position:", position["Category"].dtype, "unique sample:", position["Category"].unique()[:5])
print("Category dtype srp_ref:", srp_ref["Category"].dtype, "unique sample:", srp_ref["Category"].unique()[:5])

# check whitespace equality directly for one known matching row from earlier debug (ANU2, AFDT1400 EARRING matched with SRP=500)
row = position[(position["Store Code"]=="ANU2") & (position["Style No"]=="AFDT1400") & (position["Category"]=="EARRING")]
print(row[MATCH_KEYS])
ref_row = srp_ref[(srp_ref["Store Code"]=="ANU2") & (srp_ref["Style No"]=="AFDT1400") & (srp_ref["Category"]=="EARRING")]
print(ref_row)
for k in MATCH_KEYS:
    a = row[k].iloc[0]
    b = ref_row[k].iloc[0] if len(ref_row) else None
    print(k, repr(a), repr(b), a==b if b is not None else "n/a")
