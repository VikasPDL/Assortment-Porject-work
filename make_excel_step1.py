import pandas as pd

df = pd.read_csv("sales_pivot_step1.csv")

with pd.ExcelWriter("Sales_Pivot_Step1.xlsx", engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Sales Pivot", index=False)
    ws = writer.sheets["Sales Pivot"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)

print("done", df.shape)
