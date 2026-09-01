import json
import os
from datetime import date, datetime

from core import (
    MATCH_KEYS, group_daily, group_monthly, load_store_lookup, load_valid_styles, months_between,
    prepare_memo_return_transactions, prepare_transactions,
)

SALES_CSV = "Sales_Merged.csv"
MEMO_CSV = "Gati_Memo_Issue_Merged.csv"
MEMO_RETURN_CSV = "Gati_Stock_Return_Merged.csv"
STORE_DETAIL = "store detail.xlsx"
STOCK_MASTER = "stock - 20-07-2026.xlsx"

# Written by the Sales/Memo Issue/Memo Return API fetch in the sidebar
# (Window Settings). Old CSV = history before the API existed; this file =
# everything the API has returned since. Both get combined automatically
# whenever present.
SALES_API_OUT = "Sales_API_Fetched.csv"
MEMO_API_OUT = "Memo_Issue_API_Fetched.csv"
MEMO_RETURN_API_OUT = "Memo_Return_API_Fetched.csv"

# Sales Return (customer returns of already-sold items — different from
# Memo Return, which is a store handing back unsold consignment stock).
# Saved like the other 3, but not yet merged into any Base Stock/SRP %
# calculation — nothing in build_reference() reads this file yet.
SALES_RETURN_API_OUT = "Sales_Return_API_Fetched.csv"

# Stock has no date field — it's always a full current snapshot, not
# history to merge. The last fetch is kept here and reloaded automatically
# on every app start, so it's only ever stale until the next "Fetch Stock
# from API" click, never missing (no file upload fallback needed).
STOCK_API_OUT = "Stock_API_Fetched.csv"

BASE_STOCK_OUT = "reference_base_stock.csv"
SALES_CALENDAR_OUT = "reference_sales_calendar.csv"
MEMO_ISSUE_OUT = "reference_memo_issue.csv"
MEMO_RETURN_OUT = "reference_memo_return.csv"
SALES_DAILY_OUT = "reference_sales_daily.csv"
MEMO_DAILY_OUT = "reference_memo_daily.csv"
MEMO_RETURN_DAILY_OUT = "reference_memo_return_daily.csv"
META_OUT = "reference_meta.json"

# Per-piece (Jewel Code) event logs — one row per real transaction, not
# grouped/counted like the references above. Needed for Days-to-Sell /
# Cycle Count (Window 15): those require tracing one specific physical
# piece's own Issue -> [Return -> Issue]* -> Sale timeline, which a
# nunique(JewelCode)-per-bucket count can't reconstruct.
EVENTS_COLS = [
    "JewelCode", "Store Name", "Store Code", "Style No", "Store Grade",
    "Brand", "Price Point", "Zone", "Category", "Date",
]
SALES_EVENTS_OUT = "reference_sales_events.csv"
MEMO_EVENTS_OUT = "reference_memo_events.csv"
MEMO_RETURN_EVENTS_OUT = "reference_memo_return_events.csv"


def _sales_sources():
    sources = [SALES_CSV]
    if os.path.exists(SALES_API_OUT):
        sources.append(SALES_API_OUT)
    return sources


def _memo_sources():
    sources = [MEMO_CSV]
    if os.path.exists(MEMO_API_OUT):
        sources.append(MEMO_API_OUT)
    return sources


def _memo_return_sources():
    sources = [MEMO_RETURN_CSV]
    if os.path.exists(MEMO_RETURN_API_OUT):
        sources.append(MEMO_RETURN_API_OUT)
    return sources


def build_reference(progress_cb=None):
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    today = date.today()

    log("Loading store detail and valid style list...")
    store_codes, store_lookup = load_store_lookup(STORE_DETAIL)
    valid_styles = load_valid_styles(STOCK_MASTER)

    sales_sources = _sales_sources()
    log(f"Processing Sales data ({', '.join(sales_sources)}, this can take a minute)...")
    prepared_sales = prepare_transactions(sales_sources, store_codes, store_lookup, valid_styles)
    sales_grouped = group_monthly(prepared_sales, agg="count", out_col="Qty")
    prepared_sales[EVENTS_COLS].to_csv(SALES_EVENTS_OUT, index=False)

    # Base Stock = Total Sales Qty since the style's first sale, divided by
    # the number of months since then (till today). Grouped on MATCH_KEYS,
    # which includes Brand, so Sparq and Sparkles sales of the same Style No
    # are never summed together.
    log("Computing Base Stock reference...")
    group_keys = MATCH_KEYS
    first_seen = sales_grouped.assign(_ym=sales_grouped["Year"] * 100 + sales_grouped["Month"])
    first_seen = first_seen.groupby(group_keys)["_ym"].min().reset_index()
    first_seen["first_year"] = first_seen["_ym"] // 100
    first_seen["first_month"] = first_seen["_ym"] % 100

    total_sales = sales_grouped.groupby(group_keys)["Qty"].sum().reset_index(name="TotalSalesQty")
    base_stock = total_sales.merge(first_seen[group_keys + ["first_year", "first_month"]], on=group_keys)
    base_stock["MonthsSpan"] = months_between(base_stock["first_month"], base_stock["first_year"], today)
    base_stock["BaseStock"] = (base_stock["TotalSalesQty"] / base_stock["MonthsSpan"]).round(2)
    base_stock = base_stock[group_keys + ["TotalSalesQty", "MonthsSpan", "BaseStock"]]
    base_stock.to_csv(BASE_STOCK_OUT, index=False)

    # Sales Calendar = distinct count of Jewel Code per Store/Style/Category/Month/Year,
    # same method as Memo Issue below, just from Sales_Merged.csv.
    log("Computing Sales Calendar reference...")
    sales_calendar = group_monthly(prepared_sales, agg="nunique", out_col="JewelCodeCount")
    sales_calendar.to_csv(SALES_CALENDAR_OUT, index=False)

    # Sales by exact Date (not collapsed to Month+Year) — backs the SRP
    # Period window (Day/Week/Month/Year/Past 3 Years), which needs to slice
    # by a rolling date range that a month bucket can't reproduce.
    log("Computing daily Sales reference (for the SRP Period window)...")
    sales_daily = group_daily(prepared_sales, agg="nunique", out_col="JewelCodeCount")
    sales_daily.to_csv(SALES_DAILY_OUT, index=False)

    # Memo Issue = distinct count of Jewel Code per Store/Style/Category/Month/Year,
    # from the memo issue file (same shape as Sales_Merged.csv), plus anything
    # the Memo Issue API has fetched since.
    memo_sources = _memo_sources()
    log(f"Processing Memo Issue data ({', '.join(memo_sources)}, this can take a minute)...")
    prepared_memo = prepare_transactions(memo_sources, store_codes, store_lookup, valid_styles)
    prepared_memo[EVENTS_COLS].to_csv(MEMO_EVENTS_OUT, index=False)
    memo_issue = group_monthly(prepared_memo, agg="nunique", out_col="JewelCodeCount")
    memo_issue.to_csv(MEMO_ISSUE_OUT, index=False)

    log("Computing daily Memo Issue reference (for the SRP Period window)...")
    memo_daily = group_daily(prepared_memo, agg="nunique", out_col="JewelCodeCount")
    memo_daily.to_csv(MEMO_DAILY_OUT, index=False)

    # Memo Return, from Gati_Stock_Return_Merged.csv (plus anything the Memo
    # Return API has fetched since). Netted against Memo Issue in the SRP %
    # calculation: SRP % = Sales / (Memo - Return). Every row in that export
    # is recorded as inward to Poddar's own HO/branch, never the retail
    # store, so prepare_memo_return_transactions traces each return back to
    # whichever store most recently received that exact Jewel Code on Memo
    # Issue (see its docstring) — it needs prepared_memo (already built
    # above) to do that matching.
    memo_return_sources = _memo_return_sources()
    log(f"Processing Memo Return data ({', '.join(memo_return_sources)}, this can take a minute)...")
    prepared_memo_return = prepare_memo_return_transactions(memo_return_sources, prepared_memo, store_lookup, valid_styles)
    prepared_memo_return[EVENTS_COLS].to_csv(MEMO_RETURN_EVENTS_OUT, index=False)
    memo_return = group_monthly(prepared_memo_return, agg="nunique", out_col="JewelCodeCount")
    memo_return.to_csv(MEMO_RETURN_OUT, index=False)

    log("Computing daily Memo Return reference (for the SRP Period window)...")
    memo_return_daily = group_daily(prepared_memo_return, agg="nunique", out_col="JewelCodeCount")
    memo_return_daily.to_csv(MEMO_RETURN_DAILY_OUT, index=False)

    meta = {
        "last_refreshed": datetime.now().isoformat(timespec="seconds"),
        "base_stock_rows": len(base_stock),
        "sales_calendar_rows": len(sales_calendar),
        "memo_issue_rows": len(memo_issue),
        "memo_return_rows": len(memo_return),
        "sales_daily_rows": len(sales_daily),
        "memo_daily_rows": len(memo_daily),
        "memo_return_daily_rows": len(memo_return_daily),
    }
    with open(META_OUT, "w") as f:
        json.dump(meta, f, indent=2)

    log("Done.")
    return meta


if __name__ == "__main__":
    build_reference(progress_cb=print)
