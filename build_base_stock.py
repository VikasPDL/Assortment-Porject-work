from datetime import date

import pandas as pd

today = date.today()
TODAY_YEAR = today.year
TODAY_MONTH = today.month

df = pd.read_csv("sales_pivot_step1.csv")

id_cols = ["Store Name", "Store Code", "Style No", "Grade", "Store Potential", "Brand", "Price Point", "Month", "Year"]
category_cols = [c for c in df.columns if c not in id_cols]

months_elapsed = (TODAY_YEAR - df["Year"]) * 12 + (TODAY_MONTH - df["Month"])
months_elapsed = months_elapsed.clip(lower=1)

base_stock = df.copy()
base_stock.insert(len(id_cols), "Months Elapsed", months_elapsed)

for c in category_cols:
    base_stock[c] = (df[c] / months_elapsed).round(2)

base_stock.to_csv("base_stock.csv", index=False)
print("Base Stock sheet shape:", base_stock.shape)
print(base_stock.head(15).to_string(index=False))
