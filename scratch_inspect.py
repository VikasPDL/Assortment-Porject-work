import pandas as pd

files_csv = [
    "Sales_Merged.csv",
    "Gati_Memo_Issue_Merged.csv",
    "Sales_Return_Merged.csv",
]
files_xlsx = [
    "Price Point,brand.xlsx",
    "stock - 20-07-2026.xlsx",
    "store detail.xlsx",
]

for f in files_csv:
    print("=" * 80)
    print(f)
    df = pd.read_csv(f, nrows=5)
    print("Columns:", list(df.columns))
    print(df.head(5).to_string())

for f in files_xlsx:
    print("=" * 80)
    print(f)
    xls = pd.ExcelFile(f)
    print("Sheets:", xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(f, sheet_name=sheet, nrows=5)
        print(f"--- sheet: {sheet} ---")
        print("Columns:", list(df.columns))
        print(df.head(5).to_string())
