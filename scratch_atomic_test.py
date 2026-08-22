from reference_pipeline import build_reference
from allocation import build_stock_position

meta = build_reference(progress_cb=print)
print("meta:", meta)

position = build_stock_position("stock - 20-07-2026.xlsx")
print("SRP notna rate:", position["SRP"].notna().mean())
print("SRP notna count:", position["SRP"].notna().sum(), "of", len(position))
print("BaseStock > 0 rate:", (position["BaseStock"] > 0).mean())
