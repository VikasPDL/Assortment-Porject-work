import pandas as pd


def format_sheet(ws):
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_cells in ws.columns:
        length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)


def save(csv_path, sheet_name, xlsx_path):
    df = pd.read_csv(csv_path)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        format_sheet(writer.sheets[sheet_name])
    print(xlsx_path, df.shape)


save("sales_pivot_step1.csv", "Sales", "Sales_Pivot.xlsx")
save("memo_pivot_step2.csv", "Memo Issue", "Memo_Issue_Pivot.xlsx")
save("stock_pivot_step2.csv", "Stock", "Stock_Pivot.xlsx")
