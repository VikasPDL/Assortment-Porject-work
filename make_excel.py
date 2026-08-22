import pandas as pd

pivot_df = pd.read_csv("assortment_pivot_v2.csv")
long_df = pd.read_csv("assortment_long_format.csv")


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)


with pd.ExcelWriter("Assortment_Pivot_v2.xlsx", engine="openpyxl") as writer:
    pivot_df.to_excel(writer, sheet_name="Pivot (Stock-Sales-Memo cols)", index=False)
    format_sheet(writer.sheets["Pivot (Stock-Sales-Memo cols)"])

    long_df.to_excel(writer, sheet_name="Long Format (Category-Qty)", index=False)
    format_sheet(writer.sheets["Long Format (Category-Qty)"])

print("done", pivot_df.shape, long_df.shape)
