import pandas as pd

pd.set_option("display.max_rows", 100)

# Full price point / brand table
pp = pd.read_excel("Price Point,brand.xlsx")
print("=== Price Point, brand (full) ===")
print(pp.to_string())

# Store detail - check grade / potential distinct values, row count
sd = pd.read_excel("store detail.xlsx")
print("\n=== store detail: shape, distinct grade/potential ===")
print(sd.shape)
print(sd["grade"].value_counts())
print(sd["store_potential"].value_counts())
print(sd["store_code"].head(20).tolist())

# Stock: distinct base metal vs style-no prefix pattern
stock = pd.read_excel("stock - 20-07-2026.xlsx")
print("\n=== stock shape ===", stock.shape)
print(stock["Base Metal Raw Material"].value_counts())
stock["prefix"] = stock["Style No"].astype(str).str.extract(r"^([A-Za-z]+)")
print(stock.groupby("prefix")["Base Metal Raw Material"].value_counts().head(40))
print("\nCategory values:", stock["Category"].value_counts().head(20))
print("\nClient Code / Client Name sample (is this store?):")
print(stock[["Client Code", "Client Name", "Location Name"]].drop_duplicates().head(20))

# Sales: distinct HSNCode, PartyCode sample vs store_code set
sales = pd.read_csv("Sales_Merged.csv", usecols=["PartyCode", "StyleCode", "HSNCode", "LocationCode", "LocationName", "JewelTransDate", "MRP"], nrows=200000)
print("\n=== Sales sample ===")
print("Distinct HSNCode:", sales["HSNCode"].unique())
store_codes = set(sd["store_code"].astype(str))
party_codes = set(sales["PartyCode"].astype(str))
print("PartyCode values overlapping store_code:", len(party_codes & store_codes), "of", len(party_codes), "distinct party codes in sample")
print("Sample PartyCode not in store list:", list(party_codes - store_codes)[:20])
print("\nLocationCode distinct:", sales["LocationCode"].unique()[:20])
print("LocationName distinct:", sales["LocationName"].unique()[:20])

sales["prefix"] = sales["StyleCode"].astype(str).str.extract(r"^([A-Za-z]+)")
print("\nStyleCode prefix vs HSNCode cross tab (sample):")
print(pd.crosstab(sales["prefix"], sales["HSNCode"]).head(30))
