import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------- Price band definitions (upper-inclusive: (lower, upper]) ----------
SPARQ_BINS = [-np.inf, 2000, 4000, 7000, 10000, 15000, 25000, np.inf]
SPARQ_LABELS = ["Less Then 2k", "2k-4k", "4k-7k", "7k-10k", "10k-15k", "15k-25k", "Above 25k"]

SPARKLES_BINS = [-np.inf, 30000, 50000, 75000, 100000, 150000, 200000, np.inf]
SPARKLES_LABELS = ["Less Then 30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "Above 200k"]


def price_point(value, brand):
    bins = np.where(brand == "Sparq", 0, 1)
    result = np.empty(len(value), dtype=object)
    sparq_mask = brand == "Sparq"
    sparkles_mask = brand == "Sparkles"
    result[sparq_mask] = pd.cut(value[sparq_mask], bins=SPARQ_BINS, labels=SPARQ_LABELS, right=True).astype(object)
    result[sparkles_mask] = pd.cut(value[sparkles_mask], bins=SPARKLES_BINS, labels=SPARKLES_LABELS, right=True).astype(object)
    return result


# ---------- Store detail ----------
store = pd.read_excel("store detail.xlsx")
store["store_code"] = store["store_code"].astype(str).str.strip()
store_codes = set(store["store_code"])
store_lookup = store.set_index("store_code")[["store_name", "grade", "store_potential"]]

print(f"Stores loaded: {len(store)}")

# ---------- Helper: brand from HSN code ----------
def brand_from_hsn(hsn_series):
    hsn_str = hsn_series.astype("int64").astype(str)
    prefix6 = hsn_str.str[:6]
    brand = np.where(prefix6 == "711311", "Sparq", np.where(prefix6 == "711319", "Sparkles", None))
    return brand


def build_txn_frame(path, category_label):
    cols = ["PartyCode", "StyleCode", "HSNCode", "MRP", "JewelTransDate"]
    df = pd.read_csv(path, usecols=cols, dtype={"PartyCode": str, "StyleCode": str})
    df["PartyCode"] = df["PartyCode"].str.strip()
    df = df[df["PartyCode"].isin(store_codes)]

    brand = brand_from_hsn(df["HSNCode"])
    df["brand"] = brand
    df = df[df["brand"].isin(["Sparq", "Sparkles"])]

    df["Price Point"] = price_point(df["MRP"].to_numpy(dtype=float), df["brand"].to_numpy())

    dt = pd.to_datetime(df["JewelTransDate"], errors="coerce")
    n_bad_dates = dt.isna().sum()
    if n_bad_dates:
        print(f"  WARNING: dropping {n_bad_dates} rows with unparseable JewelTransDate")
    df = df[dt.notna()]
    dt = dt[dt.notna()]
    df["Month"] = dt.dt.month.astype(int)
    df["Year"] = dt.dt.year.astype(int)

    df = df.join(store_lookup, on="PartyCode")
    df["Category"] = category_label

    out = df.groupby(
        ["PartyCode", "store_name", "StyleCode", "Price Point", "grade", "store_potential", "Month", "Year", "Category"],
        dropna=False, observed=True
    ).size().reset_index(name="Qty")
    out = out.rename(columns={"PartyCode": "Store Code", "store_name": "Store Name", "StyleCode": "Style No",
                               "grade": "Grade", "store_potential": "Store Potential"})
    return out


print("Processing Sales_Merged.csv ...")
sales_out = build_txn_frame("Sales_Merged.csv", "Sales")
print(f"  -> {len(sales_out)} grouped rows, {sales_out['Qty'].sum()} total qty")

print("Processing Gati_Memo_Issue_Merged.csv ...")
memo_out = build_txn_frame("Gati_Memo_Issue_Merged.csv", "Memo Issue")
print(f"  -> {len(memo_out)} grouped rows, {memo_out['Qty'].sum()} total qty")

# ---------- Stock ----------
print("Processing stock - 20-07-2026.xlsx ...")
stock = pd.read_excel("stock - 20-07-2026.xlsx", usecols=["Client Code", "Style No", "Base Metal Raw Material", "Sale Price"])
stock["Client Code"] = stock["Client Code"].astype(str).str.strip()
stock = stock[stock["Client Code"].isin(store_codes)]
stock = stock[stock["Base Metal Raw Material"].isin(["Gold", "Silver"])]
stock["brand"] = np.where(stock["Base Metal Raw Material"] == "Gold", "Sparkles", "Sparq")
stock["Price Point"] = price_point(stock["Sale Price"].to_numpy(dtype=float), stock["brand"].to_numpy())
stock = stock.join(store_lookup, on="Client Code")
stock["Category"] = "Stock"
stock["Month"] = "N/A"
stock["Year"] = "N/A"

stock_out = stock.groupby(
    ["Client Code", "store_name", "Style No", "Price Point", "grade", "store_potential", "Month", "Year", "Category"],
    dropna=False, observed=True
).size().reset_index(name="Qty")
stock_out = stock_out.rename(columns={"Client Code": "Store Code", "store_name": "Store Name",
                                       "grade": "Grade", "store_potential": "Store Potential"})
print(f"  -> {len(stock_out)} grouped rows, {stock_out['Qty'].sum()} total qty")

# ---------- Combine + pivot ----------
combined = pd.concat([stock_out, sales_out, memo_out], ignore_index=True)

pivot = combined.pivot_table(
    index=["Store Name", "Store Code", "Style No", "Price Point", "Grade", "Store Potential", "Month", "Year"],
    columns="Category",
    values="Qty",
    aggfunc="sum",
    fill_value=0,
).reset_index()

for cat in ["Stock", "Sales", "Memo Issue"]:
    if cat not in pivot.columns:
        pivot[cat] = 0

pivot = pivot[["Store Name", "Store Code", "Style No", "Price Point", "Grade", "Store Potential", "Month", "Year", "Stock", "Sales", "Memo Issue"]]

pivot.to_csv("assortment_pivot_v2.csv", index=False)

combined_out = combined[["Store Name", "Store Code", "Style No", "Price Point", "Grade", "Store Potential", "Month", "Year", "Category", "Qty"]]
combined_out.to_csv("assortment_long_format.csv", index=False)

print("\n=== PIVOT SHAPE ===", pivot.shape)
print("\n=== TOTALS ===")
print(pivot[["Stock", "Sales", "Memo Issue"]].sum())
print("\n=== SAMPLE ROWS ===")
print(pivot.head(40).to_string(index=False))
print("\nSaved full result to assortment_pivot_preview.csv")
