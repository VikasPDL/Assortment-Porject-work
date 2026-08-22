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


# ---------- Store detail ----------
store = pd.read_excel("store detail.xlsx")
store["store_code"] = store["store_code"].astype(str).str.strip()
store_codes = set(store["store_code"])
store_lookup = store.set_index("store_code")[["store_name", "grade", "store_potential"]]

# ---------- Stock ----------
print("Loading stock - 20-07-2026.xlsx ...")
stock = pd.read_excel("stock - 20-07-2026.xlsx", usecols=["Client Code", "Style No", "Base Metal Raw Material", "Sale Price", "Category"])
stock["Client Code"] = stock["Client Code"].astype(str).str.strip()
stock["Style No"] = stock["Style No"].astype(str).str.strip()

before = len(stock)
stock = stock[stock["Client Code"].isin(store_codes)]
print(f"Store filter: kept {len(stock)} of {before} rows")

before = len(stock)
stock = stock[stock["Base Metal Raw Material"].isin(["Gold", "Silver"])]
print(f"Brand (Gold/Silver, Platinum excluded) filter: kept {len(stock)} of {before} rows")

stock["brand"] = np.where(stock["Base Metal Raw Material"] == "Gold", "Sparkles", "Sparq")
stock["Price Point"] = price_point(stock["Sale Price"].to_numpy(dtype=float), stock["brand"].to_numpy())
stock["Brand"] = stock["brand"]

stock = stock.join(store_lookup, on="Client Code")
stock["Month"] = "N/A"
stock["Year"] = "N/A"
stock = stock.rename(columns={"Client Code": "Store Code", "store_name": "Store Name",
                               "grade": "Grade", "store_potential": "Store Potential"})

grouped = stock.groupby(
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

pivot.to_csv("stock_pivot_step2.csv", index=False)

print("\n=== PIVOT SHAPE ===", pivot.shape)
print("\n=== SAMPLE ROWS ===")
print(pivot.head(25).to_string(index=False))
print("\nTotal Qty across all category columns:", grouped["Qty"].sum())
print("Saved to stock_pivot_step2.csv")
