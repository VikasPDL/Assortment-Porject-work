from allocation import build_stock_position

position = build_stock_position("stock - 20-07-2026.xlsx")
print("position shape:", position.shape)
print("columns:", list(position.columns))
print("SRP notna rate:", position["SRP"].notna().mean())
print("SRP notna count:", position["SRP"].notna().sum(), "of", len(position))
print(position[["Store Code", "Style No", "Category", "BaseStock", "SRP"]].head(20).to_string())
