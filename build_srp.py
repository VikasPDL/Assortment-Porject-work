import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

ID_COLS = ["Store Name", "Store Code", "Style No", "Grade", "Store Potential", "Brand", "Price Point"]


def to_long(csv_path, value_name):
    df = pd.read_csv(csv_path)
    cat_cols = [c for c in df.columns if c not in ID_COLS + ["Month", "Year"]]
    long_df = df.melt(id_vars=ID_COLS, value_vars=cat_cols, var_name="Category", value_name=value_name)
    totals = long_df.groupby(ID_COLS + ["Category"], as_index=False)[value_name].sum()
    return totals


sales_totals = to_long("sales_pivot_step1.csv", "SalesQty")
memo_totals = to_long("memo_pivot_step2.csv", "MemoQty")

merged = pd.merge(sales_totals, memo_totals, on=ID_COLS + ["Category"], how="outer")
merged["SalesQty"] = merged["SalesQty"].fillna(0)
merged["MemoQty"] = merged["MemoQty"].fillna(0)

# drop rows with no activity at all
merged = merged[(merged["SalesQty"] > 0) | (merged["MemoQty"] > 0)]

merged["SRP"] = np.where(merged["SalesQty"] > 0, (merged["MemoQty"] / merged["SalesQty"] * 100).round(1), np.nan)

pivot = merged.pivot_table(
    index=ID_COLS,
    columns="Category",
    values="SRP",
    aggfunc="first",
).reset_index()
pivot.columns.name = None

pivot.to_csv("srp_pivot.csv", index=False)

print("=== SRP PIVOT SHAPE ===", pivot.shape)
print(pivot.head(25).to_string(index=False))
print("\nSaved to srp_pivot.csv")
