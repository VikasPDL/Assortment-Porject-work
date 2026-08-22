import pandas as pd

stock = pd.read_excel("stock - 20-07-2026.xlsx", usecols=["Style No", "Category"])
g = stock.groupby("Style No")["Category"].nunique()
print("Total distinct Style No in stock:", stock["Style No"].nunique())
print("Style No with >1 distinct Category:", (g > 1).sum())
print(g[g > 1].head(10))
multi = stock[stock["Style No"].isin(g[g > 1].index)].groupby("Style No")["Category"].value_counts()
print(multi.head(20))
