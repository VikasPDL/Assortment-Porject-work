import pandas as pd

sales = pd.read_csv("sales_pivot_step1.csv")
memo = pd.read_csv("memo_pivot_step2.csv")
stock = pd.read_csv("stock_pivot_step2.csv")


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)


with pd.ExcelWriter("Sales_Memo_Stock_Pivots.xlsx", engine="openpyxl") as writer:
    sales.to_excel(writer, sheet_name="Sales", index=False)
    format_sheet(writer.sheets["Sales"])

    memo.to_excel(writer, sheet_name="Memo Issue", index=False)
    format_sheet(writer.sheets["Memo Issue"])

    stock.to_excel(writer, sheet_name="Stock", index=False)
    format_sheet(writer.sheets["Stock"])

print("done", sales.shape, memo.shape, stock.shape)
