from allocation import allocate_surplus_to_deficit, build_stock_position

position = build_stock_position("stock - 20-07-2026.xlsx")
print("Position shape:", position.shape)
print(position.head(10).to_string(index=False))

rec = allocate_surplus_to_deficit(position)
print("\nRecommendations shape:", rec.shape)
print(rec.head(15).to_string(index=False))
