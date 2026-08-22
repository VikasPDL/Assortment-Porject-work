import pandas as pd

df = pd.read_csv("srp_pivot.csv")

with pd.ExcelWriter("SRP_Pivot.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="SRP", index=False)
    ws = writer.sheets["SRP"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)

print("done", df.shape)
