import numpy as np
import pandas as pd

SPARQ_BINS = [-np.inf, 2000, 4000, 7000, 10000, 15000, 25000, np.inf]
SPARQ_LABELS = ["Less Then 2k", "2k-4k", "4k-7k", "7k-10k", "10k-15k", "15k-25k", "Above 25k"]
SPARKLES_BINS = [-np.inf, 30000, 50000, 75000, 100000, 150000, 200000, np.inf]
SPARKLES_LABELS = ["Less Then 30k", "30k-50k", "50k-75k", "75k-100k", "100k-150k", "150k-200k", "Above 200k"]

# Row keys shared by every pivot table in the app. Category is deliberately
# not in here — it becomes the pivoted column set instead.
# Zone is a 1:1 attribute of Store Code (every row for a given store always
# has the same Zone), so adding it here doesn't fragment any existing
# grouping — it just rides along as an extra column on the same rows.
ROW_KEYS = ["Store Name", "Store Code", "Style No", "Store Grade", "Brand", "Price Point", "Zone"]
MATCH_KEYS = ROW_KEYS + ["Category"]

# Raw BaseMetal code -> Brand. Sourced from the business's own code list;
# Alloy/Platinum codes map to None and are dropped (not tracked as brands).
BASE_METAL_TO_BRAND = {
    # ---- Sparkles (Gold family) ----
    "G9KW": "Sparkles", "G9K3T": "Sparkles", "G9KY": "Sparkles", "G9K2T": "Sparkles",
    "G9KR": "Sparkles", "G9KP": "Sparkles", "G22KY": "Sparkles", "G22KW": "Sparkles",
    "G18K": "Sparkles", "G18KY": "Sparkles", "G18KYW": "Sparkles", "G18KR": "Sparkles",
    "G18KW": "Sparkles", "G18KG": "Sparkles", "G18K2T": "Sparkles", "G18KRW": "Sparkles",
    "G18KP": "Sparkles", "G18KPW": "Sparkles", "G18K3T": "Sparkles",
    "G14K3T": "Sparkles", "G14KY": "Sparkles", "G14KW": "Sparkles", "G14K2T": "Sparkles",
    "G14KR": "Sparkles", "G14K": "Sparkles", "G14KP": "Sparkles",
    "G10KY": "Sparkles", "G10KW": "Sparkles", "G10KR": "Sparkles",
    "G4KY": "Sparkles", "G1KY": "Sparkles", "G995KW": "Sparkles", "G995K": "Sparkles",
    "GPBC9KW": "Sparkles", "GPBC14KW": "Sparkles", "GPBC14KR": "Sparkles", "GPBC14KY": "Sparkles",
    "GPBC10KY": "Sparkles", "GPBC10KW": "Sparkles",
    "GPBC18KR": "Sparkles", "GPBC18KY": "Sparkles", "GPBC18KW": "Sparkles",
    "GAC2518KW": "Sparkles", "GAC2518KY": "Sparkles", "GAC2518KR": "Sparkles",
    "GAC2514KW": "Sparkles", "GAC2514KY": "Sparkles", "GAC2514KR": "Sparkles",
    "GAC2510KW": "Sparkles", "GAC2510KP": "Sparkles",
    "GAC3010KY": "Sparkles", "GAC3010KW": "Sparkles",
    "GAC3014KW": "Sparkles", "GAC3014KY": "Sparkles", "GAC3014KP": "Sparkles", "GAC3014K": "Sparkles",
    "GAC3018KY": "Sparkles", "GAC3018KW": "Sparkles", "GAC3018KP": "Sparkles",
    "GAC259KW": "Sparkles", "GAC259KY": "Sparkles",
    "GAC309KY": "Sparkles", "GAC309KW": "Sparkles",
    "GMCH18KY": "Sparkles", "GMCH14K": "Sparkles",
    "GNPRD14KY": "Sparkles", "GNPRD14KR": "Sparkles", "GNPRD14KW": "Sparkles",
    "GBBR14KR": "Sparkles", "GBBR14KW": "Sparkles", "GBBR14KY": "Sparkles",
    "GLC18KW": "Sparkles", "GLC14KY": "Sparkles", "GLC14KW": "Sparkles",
    "GSOC14KY": "Sparkles", "GSOC18KY": "Sparkles",
    "GIPRD6.5MM9KW": "Sparkles",

    # ---- Sparq (Silver family) ----
    "S925KW": "Sparq", "S925KY": "Sparq", "S925KR": "Sparq", "S925KYW": "Sparq",
    "S925K2T": "Sparq", "S925K3T": "Sparq", "S925K": "Sparq",
    "S92KW": "Sparq", "S92KY": "Sparq", "S92KR": "Sparq", "S92KP": "Sparq", "S92KPW": "Sparq",
    "S92KPY": "Sparq", "S92KYW": "Sparq", "S92K2T": "Sparq", "S92K3T": "Sparq",
    "S92KRW": "Sparq", "S92KSG": "Sparq",
    "S99KW": "Sparq", "S995K": "Sparq",
    "SAC25925KY": "Sparq", "SAC30925K": "Sparq",
    "SRCH925KW": "Sparq", "SBRCH925KW": "Sparq", "SCU925K": "Sparq",
    "SNPRD925KY": "Sparq", "SNPFL925KW": "Sparq",
    "P995K": "Sparq", "P950K": "Sparq", "PAC25P950K": "Sparq",
    "G92KY": "Sparq",  # mislabeled with a G prefix in source data; follows the Silver-92 pattern

    # ---- Dropped (not tracked as a brand) ----
    "ALLOY": None, "AALLOYW-PD11": None, "ALLOY RESTRICT": None, "GPLATINUM": None,
}


def months_between(month, year, today):
    """Total whole months between a (month, year) and today, floored at 1."""
    return ((today.year - year) * 12 + (today.month - month)).clip(lower=1)


def price_point(value, brand):
    result = np.empty(len(value), dtype=object)
    sparq_mask = brand == "Sparq"
    sparkles_mask = brand == "Sparkles"
    result[sparq_mask] = pd.cut(value[sparq_mask], bins=SPARQ_BINS, labels=SPARQ_LABELS, right=True).astype(object)
    result[sparkles_mask] = pd.cut(value[sparkles_mask], bins=SPARKLES_BINS, labels=SPARKLES_LABELS, right=True).astype(object)
    return result


def brand_from_base_metal(base_metal_series):
    return base_metal_series.astype(str).str.strip().map(BASE_METAL_TO_BRAND)


def load_store_lookup(store_detail_path="store detail.xlsx"):
    store = pd.read_excel(store_detail_path)
    store["store_code"] = store["store_code"].astype(str).str.strip()
    store_codes = set(store["store_code"])
    store_lookup = store.set_index("store_code")[["store_name", "grade", "Zone"]]
    return store_codes, store_lookup


def load_valid_styles(stock_master_path="stock - 20-07-2026.xlsx"):
    """Style Nos present in the current stock master, used to keep Sales
    history limited to styles that are still part of the active assortment."""
    stock = pd.read_excel(stock_master_path, usecols=["Style No"])
    return set(stock["Style No"].dropna().astype(str).str.strip())


TRANSACTION_COLS = ["PartyCode", "StyleCode", "BaseMetal", "Category", "MRP", "JewelTransDate", "JewelCode"]


def _load_transaction_source(source):
    """source is a CSV path, a DataFrame, or a list mixing either — e.g. the
    old Sales_Merged.csv plus a newer API-fetched DataFrame/CSV, combined
    into one history."""
    if not isinstance(source, list):
        source = [source]
    parts = []
    for s in source:
        if isinstance(s, pd.DataFrame):
            parts.append(s[TRANSACTION_COLS].copy())
        else:
            parts.append(pd.read_csv(s, usecols=TRANSACTION_COLS, dtype={"PartyCode": str, "StyleCode": str}))
    return pd.concat(parts, ignore_index=True)


def prepare_transactions(source, store_codes, store_lookup, valid_styles):
    """Cleans and enriches raw transaction rows (Sales_Merged.csv-shaped):
    valid store/style filter, Brand from Base Metal, Price Point bucket,
    parsed Date/Month/Year, Store Name/Grade joined in. Shared by the
    Month+Year grouping (group_monthly, used for Base Stock/Sales
    Calendar/Memo Issue) and the exact-Date grouping (group_daily, used for
    the SRP Period window) so a source is only read and cleaned once when a
    caller needs both views. source may be a CSV path, a DataFrame, or a
    list mixing either — e.g. the old Sales_Merged.csv plus a newer
    API-fetched DataFrame/CSV, combined into one history."""
    df = _load_transaction_source(source)
    df["PartyCode"] = df["PartyCode"].astype(str).str.strip()
    df["StyleCode"] = df["StyleCode"].astype(str).str.strip()

    df = df[df["PartyCode"].isin(store_codes)]
    df = df[df["StyleCode"].isin(valid_styles)]

    df["Brand"] = brand_from_base_metal(df["BaseMetal"])
    df = df[df["Brand"].notna()]

    df["Price Point"] = price_point(df["MRP"].to_numpy(dtype=float), df["Brand"].to_numpy())

    dt = pd.to_datetime(df["JewelTransDate"], errors="coerce")
    df = df[dt.notna()].copy()
    dt = dt[dt.notna()]
    df["Date"] = dt.dt.normalize()
    df["Month"] = dt.dt.month.astype(int)
    df["Year"] = dt.dt.year.astype(int)

    df = df.join(store_lookup, on="PartyCode")
    df = df.rename(columns={"PartyCode": "Store Code", "store_name": "Store Name", "StyleCode": "Style No",
                             "grade": "Store Grade"})
    return df


def group_monthly(df, agg="count", out_col="Qty"):
    """df is the output of prepare_transactions. Used to build the Base
    Stock rate, the Sales Calendar reference, and the Memo Issue reference
    (agg="nunique" for the latter two)."""
    return df.groupby(
        ROW_KEYS + ["Month", "Year", "Category"],
        dropna=False, observed=True
    )["JewelCode"].agg(agg).reset_index(name=out_col)


def group_daily(df, agg="nunique", out_col="JewelCodeCount"):
    """Same idea as group_monthly, but keyed on exact calendar Date instead
    of Month+Year. Needed for the SRP Period window (Day/Week/Month/Year/
    Past 3 Years) — those rolling ranges can't be recovered once a source
    has already been collapsed to a month bucket."""
    return df.groupby(
        ROW_KEYS + ["Date", "Category"],
        dropna=False, observed=True
    )["JewelCode"].agg(agg).reset_index(name=out_col)


def process_transactions(csv_path, store_codes, store_lookup, valid_styles, agg="count", out_col="Qty"):
    """Used to build the Base Stock reference from Sales_Merged.csv, and the
    Memo Issue reference from Gati_Memo_Issue_Merged.csv (agg="nunique").
    csv_path may also be a DataFrame or a list of paths/DataFrames — used to
    fold in API-fetched data alongside the historical file."""
    df = prepare_transactions(csv_path, store_codes, store_lookup, valid_styles)
    return group_monthly(df, agg=agg, out_col=out_col)


STOCK_COLS = ["Client Code", "Jewel Code", "Style No", "Base Metal", "Sale Price", "Category", "ItemPcs"]


def _load_stock_source(stock_source, cols):
    """stock_source is an uploaded Excel path, or a DataFrame (e.g. fetched
    from the Stock API — no date field, always a full current snapshot)."""
    if isinstance(stock_source, pd.DataFrame):
        stock = stock_source[cols].copy()
    else:
        stock = pd.read_excel(stock_source, usecols=cols)

    # Excel exports of this report carry a trailing "Total" row with every
    # identifying field blank — drop it (and any other incomplete row)
    # before summing, or it silently inflates ItemPcs by thousands of units.
    stock = stock[stock["Style No"].notna() & stock["Category"].notna()].copy()

    # A JSON API is more likely to send "" than a true null for a blank
    # Client Code — normalize so "fresh stock" detection (isna()) still works.
    stock["Client Code"] = stock["Client Code"].replace("", np.nan)
    return stock


def process_stock_file(stock_path, store_codes, store_lookup):
    """Stock currently sitting at a store, pivoted from the uploaded stock
    file or an API-fetched DataFrame. Value is the real ItemPcs column (not
    a row count) — a jewel row can represent more than one physical piece."""
    stock = _load_stock_source(stock_path, STOCK_COLS)

    stock["Client Code"] = stock["Client Code"].astype(str).str.strip()
    stock["Style No"] = stock["Style No"].astype(str).str.strip()

    stock = stock[stock["Client Code"].isin(store_codes)]

    stock["Brand"] = brand_from_base_metal(stock["Base Metal"])
    stock = stock[stock["Brand"].notna()]

    stock["Price Point"] = price_point(stock["Sale Price"].to_numpy(dtype=float), stock["Brand"].to_numpy())

    stock = stock.join(store_lookup, on="Client Code")
    stock = stock.rename(columns={"Client Code": "Store Code", "store_name": "Store Name", "grade": "Store Grade"})

    grouped = stock.groupby(
        ROW_KEYS + ["Category"],
        dropna=False, observed=True
    )["ItemPcs"].sum().reset_index(name="Stock")
    return grouped


def process_fresh_stock(stock_path):
    """Rows from the uploaded stock file (or API-fetched DataFrame) not yet
    allocated to any store — Client Code is blank (fresh stock still sitting
    at HO), but Jewel Code and Style No are present. Returned at Jewel Code
    (piece) level, not grouped, since each piece needs its own store
    recommendation."""
    stock = _load_stock_source(stock_path, STOCK_COLS)

    fresh = stock[stock["Client Code"].isna()].copy()
    fresh["Style No"] = fresh["Style No"].astype(str).str.strip()

    fresh["Brand"] = brand_from_base_metal(fresh["Base Metal"])
    fresh = fresh[fresh["Brand"].notna()]

    fresh["Price Point"] = price_point(fresh["Sale Price"].to_numpy(dtype=float), fresh["Brand"].to_numpy())

    return fresh[["Jewel Code", "Style No", "Brand", "Category", "Price Point", "ItemPcs"]].reset_index(drop=True)
