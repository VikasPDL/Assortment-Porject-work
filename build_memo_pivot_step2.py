import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

SPARQ_BINS = [-np.inf, 2000, 4000, 7000, 10000, 15000, 25000, np.inf]
SPARQ_LABELS = ["Less Then 2k", "2k-4k", "4k-7k", "7k-10k", "10k-15k", "15k-25k", "Above 25k"]
SPARKLES_BINS = [-np.inf, 30000, 50000, 75000, 100000, 150000, 200000, np.inf]
SPARKLES_LABELS = ["Less Then 30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "Above 200k"]


def price_point(value, brand):
    result = np.empty(len(value), dtype=object)
    sparq_mask = brand == "Sparq"
    sparkles_mask = brand == "Sparkles"
    result[sparq_mask] = pd.cut(value[sparq_mask], bins=SPARQ_BINS, labels=SPARQ_LABELS, right=True).astype(object)
    result[sparkles_mask] = pd.cut(value[sparkles_mask], bins=SPARKLES_BINS, labels=SPARKLES_LABELS, right=True).astype(object)
    return result


def brand_from_hsn(hsn_series):
    hsn_str = hsn_series.astype("int64").astype(str)
    prefix6 = hsn_str.str[:6]
    return np.where(prefix6 == "711311", "Sparq", np.where(prefix6 == "711319", "Sparkles", None))


# ---------- Store detail ----------
store = pd.read_excel("store detail.xlsx")
store["store_code"] = store["store_code"].astype(str).str.strip()
store_codes = set(store["store_code"])
store_lookup = store.set_index("store_code")[["store_name", "grade", "store_potential"]]

# ---------- Style No -> item Category, from stock master ----------
stock = pd.read_excel("stock - 20-07-2026.xlsx", usecols=["Style No", "Category"])
stock["Style No"] = stock["Style No"].astype(str).str.strip()
style_category = stock.groupby("Style No")["Category"].agg(lambda s: s.value_counts().idxmax())

# ---------- Memo Issue ----------
print("Loading Gati_Memo_Issue_Merged.csv ...")
cols = ["PartyCode", "StyleCode", "HSNCode", "MRP", "JewelTransDate"]
df = pd.read_csv("Gati_Memo_Issue_Merged.csv", usecols=cols, dtype={"PartyCode": str, "StyleCode": str})
df["PartyCode"] = df["PartyCode"].str.strip()
df["StyleCode"] = df["StyleCode"].str.strip()

before = len(df)
df = df[df["PartyCode"].isin(store_codes)]
print(f"Store filter: kept {len(df)} of {before} rows")

brand = brand_from_hsn(df["HSNCode"])
df["brand"] = brand
before = len(df)
df = df[df["brand"].isin(["Sparq", "Sparkles"])]
print(f"Brand (Gold/Silver) filter: kept {len(df)} of {before} rows")

df["Price Point"] = price_point(df["MRP"].to_numpy(dtype=float), df["brand"].to_numpy())
df["Brand"] = df["brand"]

dt = pd.to_datetime(df["JewelTransDate"], errors="coerce")
before = len(df)
df = df[dt.notna()]
dt = dt[dt.notna()]
print(f"Date parse filter: kept {len(df)} of {before} rows")
df["Month"] = dt.dt.month.astype(int)
df["Year"] = dt.dt.year.astype(int)

df["Category"] = df["StyleCode"].map(style_category)
before = len(df)
matched = df["Category"].notna().sum()
print(f"Style No -> Category match: {matched} of {before} rows matched stock master; {before - matched} unmatched (dropped)")
df = df[df["Category"].notna()]

df = df.join(store_lookup, on="PartyCode")
df = df.rename(columns={"PartyCode": "Store Code", "store_name": "Store Name", "StyleCode": "Style No",
                         "grade": "Grade", "store_potential": "Store Potential"})

grouped = df.groupby(
    ["Store Name", "Store Code", "Style No", "Grade", "Store Potential", "Brand", "Price Point", "Month", "Year", "Category"],
    dropna=False, observed=True
).size().reset_index(name="Qty")

pivot = grouped.pivot_table(
    index=["Store Name", "Store Code", "Style No", "Grade", "Store Potential", "Brand", "Price Point", "Month", "Year"],
    columns="Category",
    values="Qty",
    aggfunc="sum",
    fill_value=0,
).reset_index()
pivot.columns.name = None

pivot.to_csv("memo_pivot_step2.csv", index=False)

print("\n=== PIVOT SHAPE ===", pivot.shape)
print("\n=== SAMPLE ROWS ===")
print(pivot.head(25).to_string(index=False))
print("\nTotal Qty across all category columns:", grouped["Qty"].sum())
print("Saved to memo_pivot_step2.csv")
