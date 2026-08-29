import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Some pivots (e.g. SRP-with-row-stock) exceed pandas Styler's default
# render cap (262,144 cells) once styled — raise it so those still render.
pd.set_option("styler.render.max_elements", 2_000_000)

from api_client import apply_fixed_mapping, fetch_api_dataframe
from core import STOCK_COLS, TRANSACTION_COLS, load_store_lookup, process_fresh_stock, process_stock_file
from reference_pipeline import MEMO_API_OUT, META_OUT, SALES_API_OUT, build_reference

# Bump this whenever core.py/reference_pipeline.py processing logic changes,
# so a stale cached result (same file, old columns) never lingers in a
# browser session across a code update — the cache key below depends on it.
PIPELINE_VERSION = "6-fresh-stock"

st.set_page_config(page_title="Assortment Stock & Base Stock", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class^="css"], [class*=" css"] { font-family: 'Inter', sans-serif; }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    h1 { font-size: 2.5rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.25rem !important; }

    .stApp {
        background:
            radial-gradient(circle at 12% -10%, rgba(139, 92, 246, 0.20) 0%, rgba(139, 92, 246, 0) 40%),
            radial-gradient(circle at 90% 0%, rgba(236, 72, 153, 0.14) 0%, rgba(236, 72, 153, 0) 35%),
            linear-gradient(165deg, #0F1222 0%, #131735 60%, #0F1222 100%);
    }

    /* Home tile cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: linear-gradient(160deg, rgba(255,255,255,0.06), rgba(255,255,255,0.015)) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        padding: 4px 2px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 36px rgba(0,0,0,0.40);
        border-color: rgba(139,92,246,0.55) !important;
    }

    /* Buttons -> gradient pill */
    div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #8B5CF6, #EC4899) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: filter 0.15s ease, transform 0.15s ease;
    }
    div[data-testid="stButton"] button:hover {
        filter: brightness(1.15);
        transform: translateY(-1px);
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button p { font-weight: 600 !important; }

    .home-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #C4B5FD;
        margin-bottom: 2px;
    }
    .tile-icon {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.6rem; margin-bottom: 10px;
    }
    .tile-title { font-size: 1.08rem; font-weight: 700; margin-bottom: 4px; color: #F3F0FF; }
    .tile-desc { font-size: 0.85rem; color: #A7ADC6; min-height: 44px; line-height: 1.35; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="home-eyebrow">Poddar Diamonds · Assortment Intelligence</div>', unsafe_allow_html=True)
st.title("Assortment Stock & Base Stock Dashboard")


def build_api_headers(auth_name, auth_value, from_date=None, to_date=None):
    """auth_name/value become one header (e.g. "AuthorizationToken": "..."),
    since some APIs don't use the standard "Authorization" header name.
    from_date/to_date (date objects), if given, are sent as "FromDate"/
    "ToDate" headers in DD-MM-YYYY — the format this vendor's API expects."""
    headers = {}
    if auth_name and auth_value:
        headers[auth_name] = auth_value
    if from_date is not None:
        headers["FromDate"] = from_date.strftime("%d-%m-%Y")
    if to_date is not None:
        headers["ToDate"] = to_date.strftime("%d-%m-%Y")
    return headers


# ---------------- Sidebar: reference data + upload ----------------
with st.sidebar:
    st.header("Reference Data (Base Stock)")
    if os.path.exists(META_OUT):
        with open(META_OUT) as f:
            meta = json.load(f)
        st.caption(f"Last refreshed: **{meta['last_refreshed']}**  \n{meta['base_stock_rows']} base-stock rows")
    else:
        st.warning("Reference data not built yet.")

    if st.button("Recompute Base Stock", width="stretch"):
        status_box = st.empty()

        def report(msg):
            status_box.write(msg)

        with st.spinner("Processing Sales_Merged.csv..."):
            meta = build_reference(progress_cb=report)
        st.success(f"Done. Refreshed at {meta['last_refreshed']}.")
        st.rerun()

    st.divider()
    st.header("Upload Stock File")
    uploaded = st.file_uploader("Stock file (.xlsx)", type=["xlsx"])

    st.divider()
    with st.expander("⚙️ API Settings"):
        st.caption(
            "Enter each API's URL and auth header name + value, sent as-is. Nothing here is saved to disk — "
            "re-enter each session. All 3 APIs share the same response shape, so field mapping is automatic."
        )

        # ---- Sales API ----
        st.markdown("**Sales API** — date-range based (defaults to today only; widen the range for a backfill)")
        sales_api_url = st.text_input("Sales API URL", key="sales_api_url")
        sales_auth_name = st.text_input("Auth Header Name", value="AuthorizationToken", key="sales_auth_name")
        sales_auth_value = st.text_input("Auth Header Value", key="sales_auth_value", type="password")
        sales_date_col1, sales_date_col2 = st.columns(2)
        with sales_date_col1:
            sales_from_date = st.date_input("From Date", value=datetime.now().date(), key="sales_from_date")
        with sales_date_col2:
            sales_to_date = st.date_input("To Date", value=datetime.now().date(), key="sales_to_date")
        if st.button("Fetch Sales from API", width="stretch"):
            try:
                with st.spinner("Fetching Sales API..."):
                    headers = build_api_headers(sales_auth_name, sales_auth_value, sales_from_date, sales_to_date)
                    raw_df = fetch_api_dataframe(sales_api_url, headers=headers)
                    mapped = apply_fixed_mapping(raw_df, TRANSACTION_COLS)
                    mapped.to_csv(SALES_API_OUT, index=False)
                st.success(f"Fetched and saved {len(mapped)} rows. Click 'Recompute Base Stock' above to merge them in.")
            except Exception as e:
                st.error(f"Sales API fetch failed: {e}")

        st.divider()

        # ---- Memo Issue API ----
        st.markdown("**Memo Issue API** — date-range based (defaults to today only; widen the range for a backfill)")
        memo_api_url = st.text_input("Memo Issue API URL", key="memo_api_url")
        memo_auth_name = st.text_input("Auth Header Name", value="AuthorizationToken", key="memo_auth_name")
        memo_auth_value = st.text_input("Auth Header Value", key="memo_auth_value", type="password")
        memo_date_col1, memo_date_col2 = st.columns(2)
        with memo_date_col1:
            memo_from_date = st.date_input("From Date", value=datetime.now().date(), key="memo_from_date")
        with memo_date_col2:
            memo_to_date = st.date_input("To Date", value=datetime.now().date(), key="memo_to_date")
        if st.button("Fetch Memo Issue from API", width="stretch"):
            try:
                with st.spinner("Fetching Memo Issue API..."):
                    headers = build_api_headers(memo_auth_name, memo_auth_value, memo_from_date, memo_to_date)
                    raw_df = fetch_api_dataframe(memo_api_url, headers=headers)
                    mapped = apply_fixed_mapping(raw_df, TRANSACTION_COLS)
                    mapped.to_csv(MEMO_API_OUT, index=False)
                st.success(f"Fetched and saved {len(mapped)} rows. Click 'Recompute Base Stock' above to merge them in.")
            except Exception as e:
                st.error(f"Memo Issue API fetch failed: {e}")

        st.divider()

        # ---- Stock API ----
        st.markdown("**Stock API** — no date field, always a full current snapshot, replaces the file upload")
        stock_api_url = st.text_input("Stock API URL", key="stock_api_url")
        stock_auth_name = st.text_input("Auth Header Name", value="AuthorizationToken", key="stock_auth_name")
        stock_auth_value = st.text_input("Auth Header Value", key="stock_auth_value", type="password")
        if st.button("Fetch Stock from API", width="stretch"):
            try:
                with st.spinner("Fetching Stock API..."):
                    headers = build_api_headers(stock_auth_name, stock_auth_value)
                    raw_df = fetch_api_dataframe(stock_api_url, headers=headers)
                    mapped = apply_fixed_mapping(raw_df, STOCK_COLS)
                    api_store_codes, api_store_lookup = load_store_lookup()
                    st.session_state["stock_grouped"] = process_stock_file(mapped, api_store_codes, api_store_lookup)
                    st.session_state["fresh_stock"] = process_fresh_stock(mapped)
                    st.session_state["file_key"] = f"api-fetch-{datetime.now().isoformat()}"
                st.success(f"Loaded {len(mapped)} rows from Stock API.")
                st.rerun()
            except Exception as e:
                st.error(f"Stock API fetch failed: {e}")

    st.divider()
    if st.button("🔄 Clear cached results", width="stretch",
                  help="Forces a full reprocess of the currently uploaded file."):
        for key in ["file_key", "stock_grouped", "fresh_stock"]:
            st.session_state.pop(key, None)
        st.rerun()


reference_ready = os.path.exists("reference_base_stock.csv")
if not reference_ready:
    st.error("Base Stock reference not found. Use 'Recompute Base Stock' in the sidebar first.")
    st.stop()

if uploaded is None and "stock_grouped" not in st.session_state:
    st.info("Upload today's stock file (.xlsx), or fetch it from the Stock API in the sidebar, to see the dashboard.")
    st.stop()

# ---------------- Process the uploaded file once, cache in session state ----------------
# (Skipped entirely if stock data instead came from the Stock API fetch above,
# which already wrote stock_grouped/fresh_stock straight into session state.)
if uploaded is not None:
    file_key = f"{uploaded.name}-{uploaded.size}-{PIPELINE_VERSION}"
    if st.session_state.get("file_key") != file_key:
        tmp_path = f"_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        with open(tmp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        try:
            with st.spinner("Processing stock file..."):
                store_codes, store_lookup = load_store_lookup()
                stock_grouped = process_stock_file(tmp_path, store_codes, store_lookup)
                fresh_stock = process_fresh_stock(tmp_path)
        finally:
            os.remove(tmp_path)

        st.session_state["file_key"] = file_key
        st.session_state["stock_grouped"] = stock_grouped
        st.session_state["fresh_stock"] = fresh_stock

stock_grouped = st.session_state["stock_grouped"]
fresh_stock = st.session_state["fresh_stock"]
base_stock = pd.read_csv("reference_base_stock.csv")
sales_calendar = pd.read_csv("reference_sales_calendar.csv") if os.path.exists("reference_sales_calendar.csv") else None
memo_issue = pd.read_csv("reference_memo_issue.csv") if os.path.exists("reference_memo_issue.csv") else None
sales_daily = pd.read_csv("reference_sales_daily.csv", parse_dates=["Date"]) if os.path.exists("reference_sales_daily.csv") else None
memo_daily = pd.read_csv("reference_memo_daily.csv", parse_dates=["Date"]) if os.path.exists("reference_memo_daily.csv") else None

ROW_KEYS = ["Store Name", "Store Code", "Style No", "Price Point", "Store Grade"]
ROW_KEYS_NO_STYLE = [k for k in ROW_KEYS if k != "Style No"]

MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
MONTH_ORDER = list(MONTH_NAMES.values())
SEASON_ORDER = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]

# Window 9's seasonal (current-month-only) matching needs at least this many
# qualifying years of data before it'll name a "Best Store" — a single
# one-off month (e.g. 8 sold vs 1 memoed that month = 800% SRP) shouldn't be
# enough on its own to drive a recommendation.
MIN_SEASONAL_SAMPLES = 2

# India_Culture_Events_2019-2026.csv: one row per festival/holiday date, 2019-2026,
# with a Season label. Used only to give Window 10 real-world context for the
# calendar months — mapping each Month (1-12) to its typical Season (the most
# common Season across all years for that month) and to the festivals that
# have fallen in it.
CULTURE_CSV = "India_Culture_Events_2019-2026.csv"
culture_month_lookup = None
month_to_season = {}
if os.path.exists(CULTURE_CSV):
    culture_events = pd.read_csv(CULTURE_CSV)
    culture_events["Month"] = pd.to_datetime(culture_events["Date"], format="%d-%b-%Y").dt.month

    month_to_season = culture_events.groupby("Month")["Season"].agg(lambda s: s.mode().iloc[0]).to_dict()
    festivals_by_month = culture_events.groupby("Month")["Event Name"].agg(lambda names: ", ".join(sorted(set(names))))

    culture_month_lookup = pd.DataFrame({"Month": list(MONTH_NAMES.keys())})
    culture_month_lookup["MonthName"] = culture_month_lookup["Month"].map(MONTH_NAMES)
    culture_month_lookup["Season"] = culture_month_lookup["Month"].map(month_to_season)
    culture_month_lookup["Festivals"] = culture_month_lookup["Month"].map(festivals_by_month).fillna("—")

# ---------------- Filters (sidebar) ----------------
with st.sidebar:
    st.divider()
    st.header("Filters")
    store_options = sorted(stock_grouped["Store Name"].dropna().unique().tolist())
    category_options = set(stock_grouped["Category"].dropna().unique()) | set(base_stock["Category"].dropna().unique())
    if sales_calendar is not None:
        category_options |= set(sales_calendar["Category"].dropna().unique())
    if memo_issue is not None:
        category_options |= set(memo_issue["Category"].dropna().unique())
    category_options = sorted(category_options)

    selected_stores = st.multiselect("Store Name", store_options, default=store_options)
    selected_categories = st.multiselect("Category", category_options, default=category_options)

if not selected_stores:
    selected_stores = store_options
if not selected_categories:
    selected_categories = category_options


def brand_pivot(df, value_col, brand, fill_value=0, row_keys=None):
    row_keys = row_keys or ROW_KEYS
    filtered = df[
        (df["Brand"] == brand)
        & df["Store Name"].isin(selected_stores)
        & df["Category"].isin(selected_categories)
    ]
    pivot = filtered.pivot_table(
        index=row_keys, columns="Category", values=value_col, aggfunc="sum", fill_value=fill_value
    ).reset_index()
    return pivot


def binarize(pivot, threshold=0.5, row_keys=None):
    """Blank stays blank (no sales history at all); otherwise 1 if the value
    is above the threshold, 0 if at or below it."""
    row_keys = row_keys or ROW_KEYS
    result = pivot.copy()
    cat_cols = [c for c in result.columns if c not in row_keys]
    for c in cat_cols:
        col = result[c]
        result[c] = np.where(col.isna(), np.nan, np.where(col > threshold, 1, 0))
    return result


def blank_zeros(pivot, row_keys=None):
    """0 (no stock in that Category) shows as blank instead of a literal 0."""
    row_keys = row_keys or ROW_KEYS
    result = pivot.copy()
    cat_cols = [c for c in result.columns if c not in row_keys]
    for c in cat_cols:
        result[c] = result[c].replace(0, np.nan)
    return result


def calendar_pivot(df, brand, row_keys=None):
    """Row = row_keys + Category, one column per calendar Month-Year present
    in df. Value = distinct count of Jewel Code. Used for both the Sales
    Calendar (Window 4) and Memo Issue (Window 5) references."""
    row_keys = (row_keys or ROW_KEYS) + ["Category"]
    filtered = df[
        (df["Brand"] == brand)
        & df["Store Name"].isin(selected_stores)
        & df["Category"].isin(selected_categories)
    ].copy()
    filtered["MonthYear"] = pd.to_datetime(
        filtered["Year"].astype(str) + "-" + filtered["Month"].astype(str) + "-01"
    ).dt.strftime("%b-%Y")

    pivot = filtered.pivot_table(
        index=row_keys, columns="MonthYear", values="JewelCodeCount", aggfunc="sum", fill_value=0
    ).reset_index()
    pivot.columns.name = None

    month_cols = [c for c in pivot.columns if c not in row_keys]
    month_cols.sort(key=lambda c: pd.to_datetime(c, format="%b-%Y"))
    return pivot[row_keys + month_cols]


def srp_pivot(sales_df, memo_df, row_keys, pivot_col, brand=None, min_months=1):
    """SRP % = total Sales ÷ total Memo Issue, one column per pivot_col value.
    Only months where that row+pivot_col had BOTH a sale and a memo issue
    count toward the totals (an inner join on Month+Year) — e.g. Memo 10 and
    Sales 5 in the same month contributes 10 to Memo and 5 to Sales; a
    memo-only or sale-only month is dropped entirely before summing.
    min_months: a row needs at least this many matched Month+Year pairs
    backing its total, or its SRP is left blank — guards against a single
    unusual month (e.g. 8 sold vs 1 memoed that same month = 800%) driving
    the number when there's barely any data behind it. Default 1 = no
    guard, matches every existing caller's behavior unless raised."""
    join_keys = list(dict.fromkeys(row_keys + [pivot_col, "Month", "Year"]))

    sales_f = sales_df[sales_df["Store Name"].isin(selected_stores) & sales_df["Category"].isin(selected_categories)]
    memo_f = memo_df[memo_df["Store Name"].isin(selected_stores) & memo_df["Category"].isin(selected_categories)]
    if brand is not None:
        sales_f = sales_f[sales_f["Brand"] == brand]
        memo_f = memo_f[memo_f["Brand"] == brand]

    matched = pd.merge(
        sales_f[join_keys + ["JewelCodeCount"]].rename(columns={"JewelCodeCount": "Sales"}),
        memo_f[join_keys + ["JewelCodeCount"]].rename(columns={"JewelCodeCount": "Memo"}),
        on=join_keys, how="inner",
    )

    totals = matched.groupby(row_keys + [pivot_col]).agg(
        Sales=("Sales", "sum"), Memo=("Memo", "sum"), MonthsMatched=("Sales", "size")
    ).reset_index()
    totals["SRP"] = (totals["Sales"] / totals["Memo"] * 100).round(1)
    totals.loc[totals["MonthsMatched"] < min_months, "SRP"] = np.nan

    pivot = totals.pivot_table(index=row_keys, columns=pivot_col, values="SRP", aggfunc="first").reset_index()
    pivot.columns.name = None
    return pivot


PERIOD_DAYS = {
    "Day": 1,
    "Week": 7,
    "Month": 30,
    "Year": 365,
    "Past 3 Years": 365 * 3,
}


def srp_period_anchor(sales_df, memo_df):
    """Latest transaction date across both references — the window anchors
    here instead of wall-clock today, so Day/Week/Month reflect 'the most
    recent data you have' rather than going blank whenever the source export
    lags behind the real calendar (e.g. data ends in June but it's August)."""
    dates = [d for d in (sales_df["Date"].max(), memo_df["Date"].max()) if pd.notna(d)]
    return max(dates) if dates else pd.Timestamp.now().normalize()


def srp_period_pivot(sales_df, memo_df, row_keys, pivot_col, period, anchor, brand=None):
    """SRP % = total Sales ÷ total Memo Issue within a trailing window ending
    at `anchor` (Day = last 24h before anchor, Week = last 7 days, ...
    Past 3 Years = last 3 years — see srp_period_anchor for why anchor isn't
    just today). Unlike Window 7's SRP % (which only counts a month where
    Sales and Memo both occurred), this sums directly over the whole window
    with no same-month requirement — a short window like Day or Week would
    otherwise come back blank almost every time.
    A row with Memo = 0 in the window is left blank (nothing issued to
    divide by); Sales = 0 with Memo > 0 correctly shows 0%, not blank."""
    cutoff = anchor - pd.Timedelta(days=PERIOD_DAYS[period] - 1)

    sales_f = sales_df[
        (sales_df["Date"] >= cutoff)
        & sales_df["Store Name"].isin(selected_stores)
        & sales_df["Category"].isin(selected_categories)
    ]
    memo_f = memo_df[
        (memo_df["Date"] >= cutoff)
        & memo_df["Store Name"].isin(selected_stores)
        & memo_df["Category"].isin(selected_categories)
    ]
    if brand is not None:
        sales_f = sales_f[sales_f["Brand"] == brand]
        memo_f = memo_f[memo_f["Brand"] == brand]

    sales_totals = sales_f.groupby(row_keys + [pivot_col])["JewelCodeCount"].sum().reset_index(name="Sales")
    memo_totals = memo_f.groupby(row_keys + [pivot_col])["JewelCodeCount"].sum().reset_index(name="Memo")

    totals = pd.merge(sales_totals, memo_totals, on=row_keys + [pivot_col], how="outer")
    totals["Sales"] = totals["Sales"].fillna(0)
    totals["Memo"] = totals["Memo"].fillna(0)
    totals["SRP"] = np.where(totals["Memo"] > 0, (totals["Sales"] / totals["Memo"] * 100).round(1), np.nan)

    pivot = totals.pivot_table(index=row_keys, columns=pivot_col, values="SRP", aggfunc="first").reset_index()
    pivot.columns.name = None
    return pivot


def value_pivot(df, value_col, row_keys, pivot_col, brand=None):
    """Generic pivot: one column per pivot_col value, aggregated by sum.
    Filtered by the current store/category selection, and by Brand if given."""
    filtered = df[df["Store Name"].isin(selected_stores) & df["Category"].isin(selected_categories)]
    if brand is not None:
        filtered = filtered[filtered["Brand"] == brand]
    pivot = filtered.pivot_table(index=row_keys, columns=pivot_col, values=value_col, aggfunc="sum").reset_index()
    pivot.columns.name = None
    return pivot


# SRP % + Stock traffic-light thresholds (Windows 7 & 8). Colors are set as
# explicit inline CSS (not theme tokens) so they read the same, with solid
# contrast, whether Streamlit is in light or dark mode.
SRP_HIGH = 60
SRP_LOW = 30
_SRP_STOCK_STYLES = {
    "green": "background-color: #B7E4C7; color: #14532D; font-weight: 600;",
    "yellow": "background-color: #FDE68A; color: #7A5B00; font-weight: 600;",
    "red": "background-color: #FCA5A5; color: #7A1212; font-weight: 600;",
}


def srp_stock_signal(srp_val, stock_val):
    """green = strong sell-through (SRP % >= SRP_HIGH) AND stock in hand.
    red = weak sell-through (SRP % < SRP_LOW) — stock sitting unsold, or
    nothing moving either way. yellow = everything in between, including
    a store that's selling well but is currently stocked out (SRP % high,
    stock 0) — worth a look, but not the same problem as dead stock."""
    if pd.isna(srp_val):
        return None
    stock_val = 0 if pd.isna(stock_val) else stock_val
    if srp_val >= SRP_HIGH and stock_val > 0:
        return "green"
    if srp_val < SRP_LOW:
        return "red"
    return "yellow"


def style_srp_with_row_stock(pivot, row_keys, stock_col="Current Stock (Pcs)"):
    """Window 7 shape: stock is already a column on the same pivot (from
    with_total_stock). Colors every Category SRP % cell using that row's
    Current Stock."""
    value_cols = [c for c in pivot.columns if c not in row_keys + [stock_col]]

    def row_styles(row):
        stock_val = row[stock_col]
        styles = {}
        for col in value_cols:
            signal = srp_stock_signal(row[col], stock_val)
            styles[col] = _SRP_STOCK_STYLES.get(signal, "")
        return pd.Series(styles).reindex(pivot.columns, fill_value="")

    styler = pivot.style.apply(row_styles, axis=1)
    styler = styler.format({c: "{:.1f}%".format for c in value_cols}, na_rep="")
    return styler


def style_srp_with_store_stock(srp_pivot, stock_pivot, row_keys, store_cols, extra_cols=()):
    """Window 8 shape: SRP % and Stock are two separate pivots, both with
    Store Name as columns. Colors each store's SRP % cell using that same
    row + store's value from stock_pivot (aligned on row_keys; a row/store
    missing from stock_pivot is treated as 0 stock)."""
    stock_lookup = stock_pivot.set_index(row_keys) if len(row_keys) > 1 else stock_pivot.set_index(row_keys[0])

    def row_styles(row):
        key = tuple(row[k] for k in row_keys) if len(row_keys) > 1 else row[row_keys[0]]
        try:
            stock_row = stock_lookup.loc[key]
        except KeyError:
            stock_row = None
        styles = {}
        for col in store_cols:
            stock_val = stock_row[col] if stock_row is not None and col in stock_row.index else 0
            signal = srp_stock_signal(row[col], stock_val)
            styles[col] = _SRP_STOCK_STYLES.get(signal, "")
        return pd.Series(styles).reindex(srp_pivot.columns, fill_value="")

    styler = srp_pivot.style.apply(row_styles, axis=1)
    fmt_cols = [c for c in store_cols if c in srp_pivot.columns]
    if "Best Store Value" in srp_pivot.columns:
        fmt_cols = fmt_cols + ["Best Store Value"]
    styler = styler.format({c: "{:.1f}%".format for c in fmt_cols}, na_rep="")
    return styler


def with_total_stock(pivot, row_keys, brand, stock_col="Current Stock (Pcs)"):
    """Adds a `stock_col` column = current Stock (Window 1's ItemPcs sum,
    from the uploaded stock file) summed across every Category for each
    row_keys combination — i.e. how many pieces are physically sitting in
    that store right now for that Style, regardless of which Category
    column the SRP % happens to fall under. Missing rows (no current
    stock at all) show 0, not blank. Placed right after row_keys, before
    the pivoted value columns."""
    filtered = stock_grouped[
        (stock_grouped["Brand"] == brand)
        & stock_grouped["Store Name"].isin(selected_stores)
        & stock_grouped["Category"].isin(selected_categories)
    ]
    totals = filtered.groupby(row_keys)["Stock"].sum().reset_index(name=stock_col)
    merged = pivot.merge(totals, on=row_keys, how="left")
    merged[stock_col] = merged[stock_col].fillna(0).astype(int)
    value_cols = [c for c in merged.columns if c not in row_keys + [stock_col]]
    return merged[row_keys + [stock_col] + value_cols]


def seasonal_base_stock_pivot(cal_df, row_keys, pivot_col, brand, month, min_years=1):
    """Base Stock (Window 2) is a lifetime average that blends every month
    together, which dilutes seasonal categories with their off-season
    months. This is the same idea (average pieces sold per month) but
    scoped to just calendar `month` (e.g. every August on file): total
    distinct Jewel Codes sold in that month, divided by how many different
    years that month has data for — so a style that only ever sells in
    August isn't penalized for the other 11 months. Source is sales_calendar
    (Window 4), same as Base Stock is Sales-based, not Memo-based.
    min_years: a row needs data from at least this many different years for
    that month, or its value is left blank — a single year isn't a real
    seasonal pattern yet, just one data point. Default 1 = no guard."""
    filtered = cal_df[
        (cal_df["Brand"] == brand)
        & (cal_df["Month"] == month)
        & cal_df["Store Name"].isin(selected_stores)
        & cal_df["Category"].isin(selected_categories)
    ]
    totals = filtered.groupby(row_keys + [pivot_col]).agg(
        Total=("JewelCodeCount", "sum"), Years=("Year", "nunique")
    ).reset_index()
    totals["SeasonalBaseStock"] = (totals["Total"] / totals["Years"]).round(2)
    totals.loc[totals["Years"] < min_years, "SeasonalBaseStock"] = np.nan
    pivot = totals.pivot_table(
        index=row_keys, columns=pivot_col, values="SeasonalBaseStock", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    return pivot


def month_category_pivot(df, brand, pivot_col="Category"):
    """Row = calendar Month, Jan through Dec in that fixed order (not sorted
    by value), one column per pivot_col value (Category by default; pass
    pivot_col="Zone" for an area breakdown instead). Value = total distinct
    Jewel Code sold in that calendar month, summed across every year on
    file. Same source and Brand/Store/Category filtering as Window 4 (Sales
    Calendar)."""
    filtered = df[
        (df["Brand"] == brand)
        & df["Store Name"].isin(selected_stores)
        & df["Category"].isin(selected_categories)
    ]
    pivot = filtered.pivot_table(
        index="Month", columns=pivot_col, values="JewelCodeCount", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(range(1, 13), fill_value=0)
    pivot.index = pivot.index.map(MONTH_NAMES)
    pivot.index.name = "Month"
    pivot = pivot.reset_index()
    pivot.columns.name = None
    return pivot


def season_category_pivot(df, brand):
    """Row = India season (Winter/Spring/Summer/Monsoon/Autumn), one column
    per Category. Every calendar Month is mapped to a Season using
    India_Culture_Events_2019-2026.csv (the most common Season for that
    month, e.g. Jan/Feb/Dec -> Winter), then all months sharing a season are
    summed together. Value = total distinct Jewel Code sold."""
    filtered = df[
        (df["Brand"] == brand)
        & df["Store Name"].isin(selected_stores)
        & df["Category"].isin(selected_categories)
    ].copy()
    filtered["Season"] = filtered["Month"].map(month_to_season)
    pivot = filtered.pivot_table(
        index="Season", columns="Category", values="JewelCodeCount", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(SEASON_ORDER, fill_value=0)
    pivot.index.name = "Season"
    pivot = pivot.reset_index()
    pivot.columns.name = None
    return pivot


def add_best_col_no_sort(pivot, row_keys, label="Best Category"):
    """Like add_best_column, but keeps the existing row order (Jan->Dec, or
    Winter->Autumn) instead of sorting by value — chronological order matters
    more than rank for a month/season trend view."""
    value_cols = [c for c in pivot.columns if c not in row_keys]
    result = pivot.copy()
    sub = result[value_cols]
    result[label] = sub.idxmax(axis=1, skipna=True)
    result[f"{label} Value"] = sub.max(axis=1, skipna=True)
    no_sales = result[f"{label} Value"] == 0
    result.loc[no_sales, label] = None
    result.loc[no_sales, f"{label} Value"] = np.nan
    return result


def progress_tracker(total_steps):
    """Returns a `tick(label)` function that advances a visible progress bar
    by one step each call, showing a light-lavender '<label>… NN%' line
    above the bar. Used on windows with multiple pivot/style computations
    (Windows 7/8/9/12/13) so the app doesn't sit silently for a few seconds
    while Sales/Memo/Stock get pivoted per brand. Both the bar and the
    label are cleared automatically on the step that reaches 100%."""
    text_slot = st.empty()
    bar_slot = st.empty()
    state = {"step": 0}

    def tick(label=""):
        state["step"] += 1
        pct = min(100, round(state["step"] / total_steps * 100))
        text_slot.markdown(
            f"<span style='color:#C4B5FD; font-size:0.82rem; font-weight:600;'>{label or 'Working'}… {pct}%</span>",
            unsafe_allow_html=True,
        )
        bar_slot.progress(pct / 100)
        if pct >= 100:
            text_slot.empty()
            bar_slot.empty()

    return tick


def add_best_column(pivot, row_keys, label="Best Store"):
    """Adds a <label> column (the pivot-column name with the highest value in
    that row) and a <label> Value column, then sorts rows by that value
    descending so the strongest performers surface first."""
    value_cols = [c for c in pivot.columns if c not in row_keys]
    result = pivot.copy()
    sub = result[value_cols]
    result[label] = sub.idxmax(axis=1, skipna=True)
    result[f"{label} Value"] = sub.max(axis=1, skipna=True)
    result = result.sort_values(f"{label} Value", ascending=False, na_position="last").reset_index(drop=True)
    return result


def show_pivot(pivot, key_prefix, fmt="%.2f", row_keys=None, styler=None):
    row_keys = row_keys or ROW_KEYS
    cat_cols = [c for c in pivot.columns if c not in row_keys]
    if styler is not None:
        st.dataframe(styler, width="stretch")
    else:
        st.dataframe(
            pivot,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format=fmt) for c in cat_cols},
        )
    csv = pivot.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{key_prefix}.csv", "text/csv", key=f"dl_{key_prefix}")


def show_store_pivot(pivot, row_keys, key_prefix, fmt="%.1f%%", best_label="Best Store", styler=None):
    """Like show_pivot, but skips number-formatting the text 'Best Store'
    column while still formatting its paired 'Best Store Value' column."""
    text_col = best_label
    value_col = f"{best_label} Value"
    numeric_cols = [c for c in pivot.columns if c not in row_keys + [text_col]]
    if styler is not None:
        st.dataframe(styler, width="stretch")
    else:
        st.dataframe(
            pivot,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format=fmt) for c in numeric_cols},
        )
    csv = pivot.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{key_prefix}.csv", "text/csv", key=f"dl_{key_prefix}")


MERGE_KEYS = ROW_KEYS + ["Brand", "Category"]
diff_long = pd.merge(
    stock_grouped[MERGE_KEYS + ["Stock"]],
    base_stock[MERGE_KEYS + ["BaseStock"]],
    on=MERGE_KEYS, how="outer",
)
diff_long["Stock"] = diff_long["Stock"].fillna(0)
diff_long["BaseStock"] = diff_long["BaseStock"].fillna(0)
diff_long["Difference"] = diff_long["Stock"] - diff_long["BaseStock"]


st.caption(
    f"Filters: {len(selected_stores)}/{len(store_options)} stores · "
    f"{len(selected_categories)}/{len(category_options)} categories"
)

# ---------------- Home tile grid + window navigation ----------------
# Windows used to be st.tabs() panes, which Streamlit always renders in full
# regardless of which tab is visible (every window's pivots recomputed on
# every rerun). Session-state navigation instead renders only the active
# window, driven by clicking a tile here or the sidebar "Jump to" dropdown —
# both just set st.session_state["nav"] and rerun.
WINDOW_TILES = [
    ("stock", "📦", "Window 1", "Stock", "Current stock pieces by store, style, and category."),
    ("basestock", "📊", "Window 2", "Base Stock", "Average monthly sell-through rate per style — what 'normal' stocking looks like."),
    ("diff", "➕➖", "Window 3", "Difference", "Stock minus Base Stock — spot surplus and shortfall at a glance."),
    ("sales_cal", "📅", "Window 4", "Sales Calendar", "Month-by-month sales count, per store and style."),
    ("flag", "🧾", "Window 5", "Memo Issue", "Month-by-month pieces issued to each store on memo."),
    ("no_style", "🏬", "Window 6", "Store Summary (No Style)", "Base Stock and Difference rolled up to store level, styles combined."),
    ("srp", "🎯", "Window 7", "SRP %", "Sell-through % over a period you pick, colored red / yellow / green."),
    ("store_compare", "🏆", "Window 8", "Store Comparison", "SRP %, Base Stock, and Current Stock side-by-side across every store."),
    ("fresh", "🚚", "Window 9", "Fresh Stock Suggestion", "Which store each unallocated piece should go to, and why."),
    ("seasonal", "🎉", "Window 10", "Seasonal Trends", "Month and season best-sellers, mapped to India's festival calendar."),
    ("zone", "🗺️", "Window 11", "Area (Zone) Sales", "Sales performance rolled up by Zone instead of individual store."),
    ("srp_period", "🕒", "Window 12", "SRP % by Period", "Day / Week / Month / Year / 3-Year sell-through, anchored to your latest data."),
    ("style_lookup", "🔎", "Window 13", "Style Lookup", "Paste one Style No, see every window's numbers for it in one place."),
]
WINDOW_LOOKUP = {key: (icon, num, title) for key, icon, num, title, _ in WINDOW_TILES}
TILE_ACCENTS = ["#8B5CF6", "#EC4899", "#F59E0B", "#22C55E", "#3B82F6", "#F43F5E", "#14B8A6"]

if "nav" not in st.session_state:
    st.session_state["nav"] = "home"


def _go_to(key):
    st.session_state["nav"] = key


with st.sidebar:
    st.divider()
    jump_options = ["🏠 Home"] + [f"{icon} {title}" for _, icon, num, title, _ in WINDOW_TILES]
    jump_keys = ["home"] + [key for key, *_ in WINDOW_TILES]
    current_idx = jump_keys.index(st.session_state["nav"])
    # Keying the widget by the current nav (instead of a fixed key) forces
    # Streamlit to re-init it with `index=current_idx` whenever nav changes
    # via a tile click — otherwise Streamlit keeps the dropdown's own
    # remembered value and a stale selection here would overwrite nav right
    # back on the next line, undoing the tile click.
    jump_choice = st.selectbox("Jump to", jump_options, index=current_idx, key=f"nav_jump_{st.session_state['nav']}")
    st.session_state["nav"] = jump_keys[jump_options.index(jump_choice)]

nav = st.session_state["nav"]

if nav == "home":
    st.markdown('<div class="home-eyebrow">13 windows · one dashboard</div>', unsafe_allow_html=True)
    st.header("Choose a window")
    st.write("Click a tile to open that window's data. You can jump between windows any time with the sidebar dropdown.")
    st.write("")
    cols_per_row = 3
    for row_start in range(0, len(WINDOW_TILES), cols_per_row):
        row_tiles = WINDOW_TILES[row_start:row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (col, (key, icon, num, title, desc)) in enumerate(zip(cols, row_tiles)):
            accent = TILE_ACCENTS[(row_start + i) % len(TILE_ACCENTS)]
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="tile-icon" style="background:{accent}22; border:1px solid {accent}66;">{icon}</div>
                        <div class="tile-title">{title}</div>
                        <div class="tile-desc">{desc}</div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button("Open →", key=f"tile_{key}", on_click=_go_to, args=(key,), width="stretch")
else:
    icon, num, title = WINDOW_LOOKUP[nav]
    st.button("🏠 Home", key="nav_home_btn", on_click=_go_to, args=("home",))

# ---------------- Window 1: Stock pivot ----------------
if nav == "stock":
    st.subheader("Stock — Item Pcs by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = total ItemPcs from the uploaded stock file."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "1. Start from every row in the uploaded stock file (or the Stock API fetch).\n"
            "2. Drop rows with a blank Style No or Category — this removes the report's trailing "
            "'Total' row and any incomplete row, so it can't silently inflate the piece count.\n"
            "3. Keep only rows whose **Client Code** matches a real store in `store detail.xlsx` "
            "— rows with a blank Client Code are unallocated 'fresh stock' and are excluded here "
            "(they show up instead in Window 9).\n"
            "4. **Brand** is looked up from the **Base Metal** code (e.g. `G18KY` → Sparkles, "
            "`S925KW` → Sparq). Alloy/Platinum codes aren't tracked as a brand, so those rows are dropped.\n"
            "5. **Price Point** is a bucket of the **Sale Price**, using a different set of bands for "
            "Sparq (Silver) vs Sparkles (Gold) — e.g. Sparkles '50k-75k', Sparq '4k-7k'.\n"
            "6. Store Name and Store Grade are joined in from `store detail.xlsx` using Client Code.\n"
            "7. Rows are grouped by Store Name + Store Code + Style No + Store Grade + Brand + Price "
            "Point + Category, and **ItemPcs is summed** (the real physical piece count column — "
            "not just a row count, since one row can represent several pieces).\n"
            "8. The result is pivoted with Category as columns. A 0 (no stock in that category) "
            "displays as blank."
        )

    st.markdown("### Sparkles (Gold)")
    show_pivot(blank_zeros(brand_pivot(stock_grouped, "Stock", "Sparkles")), "stock_sparkles", fmt="%.0f")

    st.markdown("### SparQ (Silver)")
    show_pivot(blank_zeros(brand_pivot(stock_grouped, "Stock", "Sparq")), "stock_sparq", fmt="%.0f")

# ---------------- Window 2: Base Stock pivot ----------------
if nav == "basestock":
    st.subheader("Base Stock — by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value: **blank** = no sales history for that combination, **1** = Base Stock above 0.5, "
        "**0** = Base Stock at or below 0.5."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Base Stock is a **rate of monthly sell-through** for each Store + Style + Category "
            "combination, built from `Sales_Merged.csv` (plus anything fetched from the Sales API) "
            "by 'Recompute Base Stock' in the sidebar:\n\n"
            "1. Keep only sales rows for a real store (matches `store detail.xlsx`) and a Style No "
            "that's still in the current stock master.\n"
            "2. Brand comes from Base Metal, Price Point from a bucket of MRP — same rules as "
            "Window 1. Rows with an unparseable transaction date are dropped.\n"
            "3. Rows are grouped by Store + Style + Grade + Brand + Price Point + **Month + Year** + "
            "Category, counting the number of sale transactions (**Qty**) in each month.\n"
            "4. For each Store + Style + Grade + Brand + Price Point + Category combination: find the "
            "**first month it ever had a sale**, and sum all Qty from then until now → **Total Sales Qty**.\n"
            "5. **Months Elapsed** = whole months from that first-sale month to today (minimum 1, so a "
            "brand-new style isn't divided by 0).\n"
            "6. **Base Stock = Total Sales Qty ÷ Months Elapsed**, rounded to 2 decimals — i.e. the "
            "average number of pieces per month this combination has sold since it started selling.\n"
            "7. For display, that number is simplified to blank/0/1 using a 0.5 cutoff: **blank** = "
            "never sold at all, **0** = selling at 0.5/month or slower, **1** = selling faster than "
            "0.5/month (worth keeping stocked)."
        )

    st.markdown("### Sparkles (Gold)")
    show_pivot(binarize(brand_pivot(base_stock, "BaseStock", "Sparkles", fill_value=None)), "basestock_sparkles", fmt="%.0f")

    st.markdown("### SparQ (Silver)")
    show_pivot(binarize(brand_pivot(base_stock, "BaseStock", "Sparq", fill_value=None)), "basestock_sparq", fmt="%.0f")

# ---------------- Window 3: Difference (Stock - Base Stock) ----------------
if nav == "diff":
    st.subheader("Difference — Stock minus Base Stock, by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = Stock − Base Stock: **positive = surplus, negative = short of Base Stock**. "
        "0 shows as blank."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "1. Take Window 1's Stock pivot and Window 2's Base Stock pivot, **before** the "
            "blank/0/1 display simplification — i.e. the actual `Stock` piece count and the actual "
            "`BaseStock` monthly rate.\n"
            "2. **Outer-join** them on Store Name + Store Code + Style No + Store Grade + Brand + "
            "Price Point + Category, so a combination that only has Stock (no sales history) or "
            "only has Base Stock (sold before, none in stock now) still shows up.\n"
            "3. Any side with no match is treated as **0** (0 stock, or 0 base stock).\n"
            "4. **Difference = Stock − Base Stock.** A positive number means more is sitting at the "
            "store than its typical monthly sell-through (surplus, candidate to move elsewhere); "
            "a negative number means it's stocked below its usual sell-through rate (short, "
            "candidate to replenish). Exactly 0 displays as blank."
        )

    st.markdown("### Sparkles (Gold)")
    show_pivot(blank_zeros(brand_pivot(diff_long, "Difference", "Sparkles")), "difference_sparkles", fmt="%.2f")

    st.markdown("### SparQ (Silver)")
    show_pivot(blank_zeros(brand_pivot(diff_long, "Difference", "Sparq")), "difference_sparq", fmt="%.2f")

# ---------------- Window 4: Sales Calendar ----------------
if nav == "sales_cal":
    st.subheader("Sales — distinct Jewel Code count by Month")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade + Category, "
        "one column per calendar Month-Year. Value = distinct count of Jewel Code from Sales."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Built from `Sales_Merged.csv` (plus the Sales API fetch) during 'Recompute Base Stock', "
            "using the same cleanup as Window 2 (valid store, valid style, Brand from Base Metal, "
            "Price Point bucketed from MRP, invalid dates dropped) — but kept broken out **by "
            "calendar month** instead of collapsed into one rate:\n\n"
            "1. Group by Store + Style + Grade + Brand + Price Point + Category + Month + Year.\n"
            "2. Count the **distinct Jewel Codes** sold in that group that month (not a row count — "
            "each physical piece's Jewel Code is only counted once).\n"
            "3. Pivot so each calendar Month-Year becomes its own column, oldest to newest. 0 shows "
            "as blank.\n\n"
            "This is the month-by-month sales detail that Window 7 (SRP %) sums up against Window 5."
        )

    if sales_calendar is None:
        st.error("reference_sales_calendar.csv not found. Use 'Recompute Base Stock' in the sidebar to build it.")
    else:
        cal_row_keys = ROW_KEYS + ["Category"]

        st.markdown("### Sparkles (Gold)")
        show_pivot(
            blank_zeros(calendar_pivot(sales_calendar, "Sparkles"), row_keys=cal_row_keys),
            "sales_cal_sparkles", fmt="%.0f", row_keys=cal_row_keys,
        )

        st.markdown("### SparQ (Silver)")
        show_pivot(
            blank_zeros(calendar_pivot(sales_calendar, "Sparq"), row_keys=cal_row_keys),
            "sales_cal_sparq", fmt="%.0f", row_keys=cal_row_keys,
        )

# ---------------- Window 5: Memo Issue pivot ----------------
if nav == "flag":
    st.subheader("Memo Issue — distinct Jewel Code count by Month")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade + Category, "
        "one column per calendar Month-Year. Value = distinct count of Jewel Code from Memo Issue."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Identical process to Window 4, but starting from `Gati_Memo_Issue_Merged.csv` (plus the "
            "Memo Issue API fetch) instead of Sales — same store/style validation, same Brand and "
            "Price Point rules, grouped the same way by Store + Style + Grade + Brand + Price Point + "
            "Category + Month + Year, counting **distinct Jewel Codes issued on memo** that month "
            "(0 shown as blank).\n\n"
            "Memo Issue represents pieces sent out to a store on approval/consignment — not "
            "necessarily sold. Compare against Window 4 (actual Sales) to see the sell-through, which "
            "is exactly what Window 7 (SRP %) does."
        )

    if memo_issue is None:
        st.error("reference_memo_issue.csv not found. Use 'Recompute Base Stock' in the sidebar to build it.")
    else:
        memo_row_keys = ROW_KEYS + ["Category"]

        st.markdown("### Sparkles (Gold)")
        show_pivot(
            blank_zeros(calendar_pivot(memo_issue, "Sparkles"), row_keys=memo_row_keys),
            "memo_sparkles", fmt="%.0f", row_keys=memo_row_keys,
        )

        st.markdown("### SparQ (Silver)")
        show_pivot(
            blank_zeros(calendar_pivot(memo_issue, "Sparq"), row_keys=memo_row_keys),
            "memo_sparq", fmt="%.0f", row_keys=memo_row_keys,
        )

# ---------------- Window 6: Base Stock + Difference, no Style No ----------------
if nav == "no_style":
    st.subheader("Store Summary — Base Stock and Difference, Style No removed")
    st.write(
        "Same Base Stock and Difference logic as Windows 2 and 3, but rolled up to "
        "Store Name + Store Code + Price Point + Store Grade (Style No dropped, categories summed across all styles)."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Exactly the Window 2 (Base Stock) and Window 3 (Difference) calculations described "
            "there — same source data, same Brand/Price Point rules, same Stock−BaseStock formula — "
            "just with **Style No removed from the row key** before pivoting. Because Style No isn't "
            "part of the grouping anymore, every style's numbers for a Store + Price Point + Store "
            "Grade + Category get summed together into one row. Use this view for a store-level read "
            "rather than a style-by-style one."
        )

    st.markdown("## Base Stock")
    st.write(
        "**blank** = no sales history for that combination, **1** = Base Stock above 0.5, "
        "**0** = Base Stock at or below 0.5."
    )
    st.markdown("### Sparkles (Gold)")
    show_pivot(
        binarize(brand_pivot(base_stock, "BaseStock", "Sparkles", fill_value=None, row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "basestock_nostyle_sparkles", fmt="%.0f", row_keys=ROW_KEYS_NO_STYLE,
    )
    st.markdown("### SparQ (Silver)")
    show_pivot(
        binarize(brand_pivot(base_stock, "BaseStock", "Sparq", fill_value=None, row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "basestock_nostyle_sparq", fmt="%.0f", row_keys=ROW_KEYS_NO_STYLE,
    )

    st.divider()
    st.markdown("## Difference")
    st.write("Value = Stock − Base Stock: **positive = surplus, negative = short of Base Stock**. 0 shows as blank.")
    st.markdown("### Sparkles (Gold)")
    show_pivot(
        blank_zeros(brand_pivot(diff_long, "Difference", "Sparkles", row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "difference_nostyle_sparkles", fmt="%.2f", row_keys=ROW_KEYS_NO_STYLE,
    )
    st.markdown("### SparQ (Silver)")
    show_pivot(
        blank_zeros(brand_pivot(diff_long, "Difference", "Sparq", row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "difference_nostyle_sparq", fmt="%.2f", row_keys=ROW_KEYS_NO_STYLE,
    )

# ---------------- Window 7: SRP % (Sales ÷ Memo Issue, Period logic) ----------------
if nav == "srp":
    st.subheader("SRP % — Sales ÷ Memo Issue, by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = SRP % = Sales ÷ Memo Issue, summed over the trailing period selected below. "
        "**Current Stock (Pcs)** = pieces of that Style physically sitting in that store right now."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "SRP % measures **sell-through**: of what was issued to a store on memo, how much "
            "actually sold — scoped to a **trailing date window** (same engine as Window 12) instead "
            "of Window 4/5's Month+Year buckets:\n\n"
            "1. Start from `reference_sales_daily.csv` and `reference_memo_daily.csv` — exact "
            "transaction dates, not collapsed to a calendar month.\n"
            "2. The window ends at the **latest transaction date in the data**, not necessarily "
            "today — so a lagging export doesn't make short periods come back empty for no real "
            "reason. **Day** = last 24 hours before that date, **Week** = last 7 days, **Month** = "
            "last 30 days, **Year** = last 365 days, **Past 3 Years** = last 3 years.\n"
            "3. Filter both Sales and Memo Issue rows to that window, then sum **Sales** and **Memo** "
            "separately for each Store + Style + Grade + Price Point + Category — no requirement that "
            "a Sale and a Memo Issue land in the same month; a short window would otherwise come back "
            "blank almost every time.\n"
            "4. **SRP % = (Sales ÷ Memo) × 100**, rounded to 1 decimal. A row with **0 Memo Issue** "
            "in the window is left blank (nothing to divide by); **0 Sales** with Memo > 0 correctly "
            "shows **0%**, not blank.\n"
            "5. **Current Stock (Pcs)** is Window 1's Stock (the uploaded stock file's ItemPcs), "
            "summed across every Category for that Store + Style + Grade + Price Point — not scoped "
            "to the period, since it's a right-now snapshot, not a historical total.\n"
            f"6. **Color** — 🟩 green: SRP % ≥ {SRP_HIGH}% *and* stock in hand (selling well, still "
            f"stocked). 🟥 red: SRP % < {SRP_LOW}% (weak sell-through — includes dead stock sitting "
            "unsold, and styles with nothing moving either way). 🟨 yellow: everything in between, "
            f"including a store selling well (SRP % ≥ {SRP_HIGH}%) but currently out of that stock — "
            "a different problem (missed sales) than dead stock, so it isn't colored red."
        )

    if sales_daily is None or memo_daily is None:
        st.error("reference_sales_daily.csv or reference_memo_daily.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        srp7_anchor = srp_period_anchor(sales_daily, memo_daily)
        st.caption(f"Data as of **{srp7_anchor.strftime('%d %b %Y')}** — every period below counts back from that date.")
        srp7_period = st.radio(
            "Period", list(PERIOD_DAYS.keys()), index=2, horizontal=True, key="srp_period_choice_w7",
        )
        st.caption(
            f"🟩 SRP % ≥ {SRP_HIGH}% with stock in hand &nbsp;·&nbsp; "
            f"🟨 mixed signal (moderate SRP %, or strong SRP % but stocked out) &nbsp;·&nbsp; "
            f"🟥 SRP % < {SRP_LOW}% — weak sell-through"
        )

        srp7_tick = progress_tracker(4)

        st.markdown("### Sparkles (Gold)")
        srp7_tick("Computing Sparkles SRP %")
        srp7_sparkles = srp_period_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", srp7_period, srp7_anchor, brand="Sparkles")
        srp7_sparkles = with_total_stock(srp7_sparkles, ROW_KEYS, "Sparkles")
        srp7_tick("Rendering Sparkles table")
        show_pivot(
            srp7_sparkles, "srp_sparkles", fmt="%.1f%%", row_keys=ROW_KEYS + ["Current Stock (Pcs)"],
            styler=style_srp_with_row_stock(srp7_sparkles, ROW_KEYS),
        )

        st.markdown("### SparQ (Silver)")
        srp7_tick("Computing SparQ SRP %")
        srp7_sparq = srp_period_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", srp7_period, srp7_anchor, brand="Sparq")
        srp7_sparq = with_total_stock(srp7_sparq, ROW_KEYS, "Sparq")
        srp7_tick("Rendering SparQ table")
        show_pivot(
            srp7_sparq, "srp_sparq", fmt="%.1f%%", row_keys=ROW_KEYS + ["Current Stock (Pcs)"],
            styler=style_srp_with_row_stock(srp7_sparq, ROW_KEYS),
        )

# ---------------- Window 8: Store Comparison (SRP % and Base Stock, Store Name as columns) ----------------
if nav == "store_compare":
    st.subheader("Store Comparison — SRP %, Base Stock, and Current Stock, side by side across stores")
    st.write(
        "Row = Style No + Price Point + Category, one column per Store Name, split into Sparkles and SparQ. "
        "'Best Store' / 'Most Stocked Store' = the store with the highest value in that row, and rows are "
        "sorted by that value, highest first, so the top performers surface immediately."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Same underlying numbers as Windows 7, 2, and 1, just re-pivoted with **Store Name as the "
            "columns** instead of Category, so stores can be compared side by side for one "
            "Style No + Price Point + Category:\n\n"
            "- **SRP % table** — Window 7's period-based SRP % (Sales ÷ Memo Issue over the trailing "
            "window picked below, anchored to the latest data date, no same-month requirement), "
            "pivoted with Store Name as columns instead of Category.\n"
            "- **Base Stock table** — Window 2's raw Base Stock rate (Total Sales Qty ÷ Months "
            "Elapsed, *before* the blank/0/1 simplification), pivoted the same way.\n"
            "- **Current Stock table** — Window 1's raw Stock (ItemPcs from the uploaded stock file, "
            "a right-now snapshot, not period-scoped), pivoted the same way.\n\n"
            "For each row, **'Best Store'** (or **'Most Stocked Store'** for the Current Stock table) "
            "is the column name (store) holding the highest value, and its paired 'Value' column is "
            "that value. Rows are then sorted by that value, highest first, so the strongest-performing "
            "combinations surface at the top.\n\n"
            f"**Color on the SRP % table** — same rule as Window 7: 🟩 SRP % ≥ {SRP_HIGH}% *and* stock "
            f"in hand at that store. 🟥 SRP % < {SRP_LOW}% (weak sell-through, dead stock risk if "
            f"stock's still sitting there). 🟨 everything in between, including a store selling well "
            "but currently out of that stock."
        )
    STORE_COMPARE_ROW_KEYS = ["Style No", "Price Point", "Category"]

    # Computed once here and reused below by both the SRP % coloring and the
    # standalone Current Stock table, instead of re-pivoting stock_grouped twice.
    stock_raw_sparkles = value_pivot(stock_grouped, "Stock", STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparkles")
    stock_raw_sparq = value_pivot(stock_grouped, "Stock", STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparq")

    srp8_ready = sales_daily is not None and memo_daily is not None
    st8_tick = progress_tracker(6 if srp8_ready else 4)

    st.markdown("## SRP % — best store by sell-through (Window 7 / period logic)")
    if not srp8_ready:
        st.error("reference_sales_daily.csv or reference_memo_daily.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        srp8_anchor = srp_period_anchor(sales_daily, memo_daily)
        st.caption(f"Data as of **{srp8_anchor.strftime('%d %b %Y')}** — every period below counts back from that date.")
        srp8_period = st.radio(
            "Period", list(PERIOD_DAYS.keys()), index=2, horizontal=True, key="srp_period_choice_w8",
        )
        st.caption(
            f"🟩 SRP % ≥ {SRP_HIGH}% with stock in hand &nbsp;·&nbsp; "
            f"🟨 mixed signal (moderate SRP %, or strong SRP % but stocked out) &nbsp;·&nbsp; "
            f"🟥 SRP % < {SRP_LOW}% — weak sell-through"
        )

        st.markdown("### Sparkles (Gold)")
        st8_tick("Computing Sparkles SRP %")
        srp_store_sparkles = srp_period_pivot(
            sales_daily, memo_daily, STORE_COMPARE_ROW_KEYS, "Store Name", srp8_period, srp8_anchor, brand="Sparkles"
        )
        srp_store_sparkles = add_best_column(srp_store_sparkles, STORE_COMPARE_ROW_KEYS)
        srp8_cols_sparkles = [c for c in srp_store_sparkles.columns if c not in STORE_COMPARE_ROW_KEYS + ["Best Store", "Best Store Value"]]
        show_store_pivot(
            srp_store_sparkles, STORE_COMPARE_ROW_KEYS, "store_compare_srp_sparkles", fmt="%.1f%%",
            styler=style_srp_with_store_stock(srp_store_sparkles, stock_raw_sparkles, STORE_COMPARE_ROW_KEYS, srp8_cols_sparkles),
        )

        st.markdown("### SparQ (Silver)")
        st8_tick("Computing SparQ SRP %")
        srp_store_sparq = srp_period_pivot(
            sales_daily, memo_daily, STORE_COMPARE_ROW_KEYS, "Store Name", srp8_period, srp8_anchor, brand="Sparq"
        )
        srp_store_sparq = add_best_column(srp_store_sparq, STORE_COMPARE_ROW_KEYS)
        srp8_cols_sparq = [c for c in srp_store_sparq.columns if c not in STORE_COMPARE_ROW_KEYS + ["Best Store", "Best Store Value"]]
        show_store_pivot(
            srp_store_sparq, STORE_COMPARE_ROW_KEYS, "store_compare_srp_sparq", fmt="%.1f%%",
            styler=style_srp_with_store_stock(srp_store_sparq, stock_raw_sparq, STORE_COMPARE_ROW_KEYS, srp8_cols_sparq),
        )

    st.divider()
    st.markdown("## Base Stock — best store by base stock (Window 2 logic)")
    st.markdown("### Sparkles (Gold)")
    st8_tick("Computing Sparkles Base Stock")
    basestock_store_sparkles = value_pivot(base_stock, "BaseStock", STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparkles")
    basestock_store_sparkles = add_best_column(basestock_store_sparkles, STORE_COMPARE_ROW_KEYS)
    show_store_pivot(basestock_store_sparkles, STORE_COMPARE_ROW_KEYS, "store_compare_basestock_sparkles", fmt="%.2f")

    st.markdown("### SparQ (Silver)")
    st8_tick("Computing SparQ Base Stock")
    basestock_store_sparq = value_pivot(base_stock, "BaseStock", STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparq")
    basestock_store_sparq = add_best_column(basestock_store_sparq, STORE_COMPARE_ROW_KEYS)
    show_store_pivot(basestock_store_sparq, STORE_COMPARE_ROW_KEYS, "store_compare_basestock_sparq", fmt="%.2f")

    st.divider()
    st.markdown("## Current Stock — pieces per store (Window 1 logic)")
    st.write("Value = current Stock (ItemPcs from the uploaded stock file). 'Most Stocked Store' = the store holding the most pieces right now.")
    st.markdown("### Sparkles (Gold)")
    st8_tick("Computing Sparkles Current Stock")
    stock_store_sparkles = add_best_column(stock_raw_sparkles.copy(), STORE_COMPARE_ROW_KEYS, label="Most Stocked Store")
    show_store_pivot(stock_store_sparkles, STORE_COMPARE_ROW_KEYS, "store_compare_stock_sparkles", fmt="%.0f", best_label="Most Stocked Store")

    st.markdown("### SparQ (Silver)")
    st8_tick("Computing SparQ Current Stock")
    stock_store_sparq = add_best_column(stock_raw_sparq.copy(), STORE_COMPARE_ROW_KEYS, label="Most Stocked Store")
    show_store_pivot(stock_store_sparq, STORE_COMPARE_ROW_KEYS, "store_compare_stock_sparq", fmt="%.0f", best_label="Most Stocked Store")

# ---------------- Window 9: Fresh Stock Suggestion ----------------
if nav == "fresh":
    st.subheader("Fresh Stock — which store should each piece be sent to")
    current_month = datetime.now().month
    current_month_name = datetime.now().strftime("%B")
    st.write(
        f"Fresh stock = rows in the uploaded stock file with a blank Client Code (not yet allocated "
        f"to any store), identified by Jewel Code + Style No. "
        f"For each piece, matched by Style No + Price Point + Brand + Category against **only "
        f"{current_month_name} history (every year on file)**: the store with the highest SRP %, and "
        f"the store with the highest seasonal Base Stock. Requires at least {MIN_SEASONAL_SAMPLES} "
        f"qualifying years of {current_month_name} data before naming a Best Store, so a single "
        f"one-off month can't drive the recommendation. Sorted by SRP Best Store Value, highest first."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "This window suggests where to send **new, unallocated pieces**, scoped to the "
            f"**current calendar month ({current_month_name}), across every year on file** — not "
            "lifetime averages — so a store that's strong in this specific month (festival/seasonal "
            "demand) outranks one that's only strong in other months:\n\n"
            "1. From the uploaded stock file, take every row with a **blank Client Code** — a piece "
            "still sitting at HO, not yet sent to any store. Unlike Window 1, these are kept "
            "**one row per piece** (by Jewel Code), not summed together, since each piece needs its "
            "own recommendation. Brand and Price Point are derived the same way as Window 1.\n"
            f"2. Sales (Window 4) and Memo Issue (Window 5) history is first filtered down to rows "
            f"dated in {current_month_name} — of any year. E.g. run in August, only every August on "
            "record counts; September/July etc. are excluded from this window's matching.\n"
            "3. **SRP Best Store** = Window 7's SRP % calculation (Sales ÷ Memo Issue, matched by "
            f"month+year), but computed only from that {current_month_name}-only data.\n"
            f"4. **Base Stock Best Store** = a seasonal version of Window 2's rate: total distinct "
            f"pieces sold in {current_month_name}, divided by how many different years {current_month_name} "
            "has data for, per Style No + Price Point + Category + Brand + Store — i.e. 'typical "
            f"{current_month_name} sales', not a 12-month blended average.\n"
            f"5. **Minimum sample guard**: a Store only counts toward SRP Best Store if it has at "
            f"least {MIN_SEASONAL_SAMPLES} different years where that Style/Price Point/Category had "
            f"*both* a sale and a memo issue in {current_month_name} (not just 1) — and toward Base "
            f"Stock Best Store only with at least {MIN_SEASONAL_SAMPLES} years of {current_month_name} "
            "sales data. This exists because SRP % is Sales ÷ Memo Issue *in the same month* — if a "
            "single one-off month had, say, 8 sold against only 1 memoed that month (leftover stock "
            "from an earlier memo selling through later), the ratio can spike to 800%+ off just that "
            "one data point. Requiring a second qualifying year filters that kind of fluke out.\n"
            "6. Match each fresh piece to those two lookups on Style No + Price Point + Category + "
            "Brand, attaching its 'SRP Best Store' / 'SRP Best Store Value' and 'Base Stock Best "
            "Store' / 'Base Stock Best Store Value'.\n"
            "7. Sort all pieces by SRP Best Store Value first, then Base Stock Best Store Value, both "
            "highest first. A piece whose Style No/Price Point/Category/Brand never sold in "
            f"{current_month_name} in any prior year — or only has one qualifying year — shows blank "
            "recommendations, even if it has sales history in other months."
        )

    if fresh_stock.empty:
        st.info("No fresh stock found in the uploaded file (no rows with a blank Client Code).")
    elif sales_calendar is None or memo_issue is None:
        st.error("reference_sales_calendar.csv or reference_memo_issue.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        match_keys = ["Style No", "Price Point", "Category"]
        sales_seasonal = sales_calendar[sales_calendar["Month"] == current_month]
        memo_seasonal = memo_issue[memo_issue["Month"] == current_month]

        fresh9_tick = progress_tracker(5)
        srp_lookup_parts = []
        bs_lookup_parts = []
        for b in ["Sparkles", "Sparq"]:
            fresh9_tick(f"Matching {b} seasonal SRP %")
            srp_b = srp_pivot(
                sales_seasonal, memo_seasonal, match_keys, "Store Name", brand=b, min_months=MIN_SEASONAL_SAMPLES,
            )
            srp_b = add_best_column(srp_b, match_keys, label="SRP Best Store")
            srp_b["Brand"] = b
            srp_lookup_parts.append(srp_b[match_keys + ["Brand", "SRP Best Store", "SRP Best Store Value"]])

            fresh9_tick(f"Matching {b} seasonal Base Stock")
            bs_b = seasonal_base_stock_pivot(
                sales_calendar, match_keys, "Store Name", b, current_month, min_years=MIN_SEASONAL_SAMPLES,
            )
            bs_b = add_best_column(bs_b, match_keys, label="Base Stock Best Store")
            bs_b["Brand"] = b
            bs_lookup_parts.append(bs_b[match_keys + ["Brand", "Base Stock Best Store", "Base Stock Best Store Value"]])

        fresh9_tick("Ranking fresh stock pieces")
        srp_lookup = pd.concat(srp_lookup_parts, ignore_index=True)
        bs_lookup = pd.concat(bs_lookup_parts, ignore_index=True)

        suggestions = fresh_stock.merge(srp_lookup, on=match_keys + ["Brand"], how="left")
        suggestions = suggestions.merge(bs_lookup, on=match_keys + ["Brand"], how="left")
        suggestions = suggestions.sort_values(
            ["SRP Best Store Value", "Base Stock Best Store Value"], ascending=False, na_position="last"
        ).reset_index(drop=True)

        display_cols = [
            "Jewel Code", "Style No", "Brand", "Category", "Price Point", "ItemPcs",
            "SRP Best Store", "SRP Best Store Value", "Base Stock Best Store", "Base Stock Best Store Value",
        ]
        st.caption(f"{len(suggestions)} fresh stock pieces found.")
        st.dataframe(
            suggestions[display_cols],
            width="stretch",
            column_config={
                "SRP Best Store Value": st.column_config.NumberColumn(format="%.1f%%"),
                "Base Stock Best Store Value": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        csv = suggestions[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "fresh_stock_suggestion.csv", "text/csv", key="dl_fresh_stock")

# ---------------- Window 10: Seasonal Trends (India Festival Calendar) ----------------
if nav == "seasonal":
    st.subheader("Seasonal Trends — Month-wise Best Sellers, mapped to India's Festival Calendar")
    st.write(
        "Month-wise: row = calendar Month (Jan → Dec, every year on file combined), one column per "
        "Category, plus that month's Season and Festivals from India_Culture_Events_2019-2026.csv. "
        "Season-wise: the same sales, rolled up to the 5 seasons. "
        "**Best Category** = the highest-selling category, in that Month or Season."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "1. Source is Window 4's Sales Calendar (distinct Jewel Code count per Store + Style + "
            "Category + **Month + Year**) — the same data, just summed a different way: across "
            "**every Store, Style and Year**, grouped only by Month and Category.\n"
            "2. Every calendar month (1-12) becomes one row, always shown in Jan → Dec order — "
            "**not** sorted by how much sold, so it reads as a year-round trend rather than a "
            "leaderboard. A month with zero matching sales still appears, with 0s.\n"
            "3. **Best Category** = the Category with the highest total that month, and **Best "
            "Category Value** its total. A month with no sales at all leaves both blank.\n"
            "4. **Season** and **Festivals** come from `India_Culture_Events_2019-2026.csv` (festival "
            "dates from 2019-2026): each Month is tagged with the Season it was most often marked as "
            "in that file (e.g. Jan/Feb/Dec → Winter), and with every festival name recorded in that "
            "month across all years — so you can see, e.g., whether a spike lines up with Diwali or "
            "Raksha Bandhan.\n"
            "5. **Season-wise rollup** = the same Month totals, regrouped by summing every month that "
            "shares a Season (e.g. Diwali/Dussehra's Autumn = Oct + Nov combined) — useful when a "
            "festival's date shifts month to month across years (lunar calendar) and a single-month "
            "view would split its effect.\n\n"
            "This uses every year of history on file — it is not scoped to the current month, unlike "
            "Window 9."
        )

    if sales_calendar is None:
        st.error("reference_sales_calendar.csv not found. Use 'Recompute Base Stock' in the sidebar to build it.")
    elif culture_month_lookup is None:
        st.error(f"{CULTURE_CSV} not found — Season/Festival columns can't be built without it.")
    else:
        for brand, label in [("Sparkles", "Sparkles (Gold)"), ("Sparq", "SparQ (Silver)")]:
            st.markdown(f"### {label}")

            st.markdown("**Month-wise**")
            month_pivot = month_category_pivot(sales_calendar, brand)
            month_pivot = add_best_col_no_sort(month_pivot, row_keys=["Month"], label="Best Category")
            month_pivot = month_pivot.merge(
                culture_month_lookup[["MonthName", "Season", "Festivals"]],
                left_on="Month", right_on="MonthName", how="left",
            ).drop(columns="MonthName")
            front_cols = ["Month", "Season", "Festivals"]
            other_cols = [c for c in month_pivot.columns if c not in front_cols]
            month_pivot = month_pivot[front_cols + other_cols]
            show_pivot(
                blank_zeros(month_pivot, row_keys=front_cols + ["Best Category"]),
                f"seasonal_month_{brand.lower()}", fmt="%.0f", row_keys=front_cols + ["Best Category"],
            )

            st.markdown("**Season-wise rollup**")
            season_pivot = season_category_pivot(sales_calendar, brand)
            season_pivot = add_best_col_no_sort(season_pivot, row_keys=["Season"], label="Best Category")
            show_pivot(
                blank_zeros(season_pivot, row_keys=["Season", "Best Category"]),
                f"seasonal_season_{brand.lower()}", fmt="%.0f", row_keys=["Season", "Best Category"],
            )

            st.divider()

# ---------------- Window 11: Area (Zone) Sales ----------------
if nav == "zone":
    st.subheader("Area (Zone) Sales — which Zone sells best, by Style and by Month")
    st.write(
        "Style-wise: row = Style No + Category, one column per Zone (North/South/East/West). "
        "Month-wise: row = calendar Month (Jan → Dec, every year on file combined), one column per "
        "Zone. **Best Zone** = the Zone with the highest sales in that row."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "**Zone** comes from `store detail.xlsx` (North/South/East/West Zone) — every store "
            "belongs to exactly one Zone, so this is just a different way of grouping the same "
            "stores you see everywhere else in the app, not new data.\n\n"
            "- **Style-wise** — Window 4's Sales Calendar data (distinct Jewel Code count), summed "
            "by Style No + Category, with Zone pivoted into columns instead of Store Name. **Best "
            "Zone** = the Zone with the highest total for that Style + Category, and rows are sorted "
            "by that value, highest first — same convention as Window 8's Store Comparison, just "
            "rolled up to Zone instead of individual store.\n"
            "- **Month-wise** — the same Sales Calendar data, but grouped only by Month (across every "
            "Store, Style and Year) with Zone as columns, kept in Jan → Dec order — same convention "
            "as Window 10, just Zone instead of Category. A row with no sales in any Zone that month "
            "leaves Best Zone blank.\n\n"
            "A Style/Category or Month that only ever sold in one Zone will naturally show blank in "
            "the other Zone columns — that's normal, not missing data."
        )

    if sales_calendar is None:
        st.error("reference_sales_calendar.csv not found. Use 'Recompute Base Stock' in the sidebar to build it.")
    elif "Zone" not in sales_calendar.columns:
        st.error(
            "This reference data was built before Zone tracking was added. "
            "Click 'Recompute Base Stock' in the sidebar to rebuild it with Zone included."
        )
    else:
        style_row_keys = ["Style No", "Category"]
        for brand, label in [("Sparkles", "Sparkles (Gold)"), ("Sparq", "SparQ (Silver)")]:
            st.markdown(f"### {label}")

            st.markdown("**Style-wise**")
            style_zone = value_pivot(sales_calendar, "JewelCodeCount", style_row_keys, "Zone", brand=brand)
            style_zone = add_best_column(style_zone, style_row_keys, label="Best Zone")
            show_store_pivot(
                style_zone, style_row_keys, f"zone_style_{brand.lower()}", fmt="%.0f", best_label="Best Zone",
            )

            st.markdown("**Month-wise**")
            month_zone = month_category_pivot(sales_calendar, brand, pivot_col="Zone")
            month_zone = add_best_col_no_sort(month_zone, row_keys=["Month"], label="Best Zone")
            month_zone = month_zone.merge(
                culture_month_lookup[["MonthName", "Season", "Festivals"]],
                left_on="Month", right_on="MonthName", how="left",
            ).drop(columns="MonthName")
            front_cols = ["Month", "Season", "Festivals"]
            other_cols = [c for c in month_zone.columns if c not in front_cols]
            month_zone = month_zone[front_cols + other_cols]
            show_pivot(
                blank_zeros(month_zone, row_keys=front_cols + ["Best Zone"]),
                f"zone_month_{brand.lower()}", fmt="%.0f", row_keys=front_cols + ["Best Zone"],
            )
            st.divider()

# ---------------- Window 12: SRP % by Period ----------------
if nav == "srp_period":
    st.subheader("SRP % by Period — Sales ÷ Memo Issue, over a trailing window you choose")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = SRP % = Sales ÷ Memo Issue, summed over the trailing period selected below."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Same idea as Window 7's SRP %, but scoped to a **trailing date window** instead of a "
            "lifetime total, and built from exact transaction dates instead of Month+Year buckets:\n\n"
            "1. Start from `reference_sales_daily.csv` and `reference_memo_daily.csv` — the same "
            "cleaned transaction rows as Windows 4 and 5 (valid store, valid style, Brand from Base "
            "Metal, Price Point from MRP), but kept at exact **Date** resolution instead of collapsed "
            "to a calendar month.\n"
            "2. The window ends at the **latest transaction date in the data**, not necessarily "
            "today — if your Sales/Memo export lags behind the real calendar, anchoring to actual "
            "today would make Day/Week/Month come back empty for no real reason. **Day** = last 24 "
            "hours before that date, **Week** = last 7 days, **Month** = last 30 days, **Year** = "
            "last 365 days, **Past 3 Years** = last 3 years.\n"
            "3. Filter both Sales and Memo Issue rows to that window, then sum **Sales** and **Memo** "
            "separately for each Store + Style + Grade + Price Point + Category — unlike Window 7, "
            "there's no requirement that a Sale and a Memo Issue land in the same month, since a "
            "short window like Day or Week would otherwise come back blank almost every time.\n"
            "4. **SRP % = (Sales ÷ Memo) × 100**, rounded to 1 decimal. A row with **0 Memo Issue** "
            "in the window is left blank (nothing to divide by); **0 Sales** with Memo > 0 correctly "
            "shows **0%**, not blank."
        )

    if sales_daily is None or memo_daily is None:
        st.error(
            "reference_sales_daily.csv or reference_memo_daily.csv not found. "
            "Click 'Recompute Base Stock' in the sidebar to build them."
        )
    else:
        anchor = srp_period_anchor(sales_daily, memo_daily)
        st.caption(f"Data as of **{anchor.strftime('%d %b %Y')}** — every period below counts back from that date.")

        period = st.radio(
            "Period", list(PERIOD_DAYS.keys()), index=2, horizontal=True, key="srp_period_choice",
        )

        srp12_tick = progress_tracker(2)

        st.markdown("### Sparkles (Gold)")
        srp12_tick("Computing Sparkles SRP %")
        show_pivot(
            srp_period_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", period, anchor, brand="Sparkles"),
            f"srp_period_{period.lower().replace(' ', '_')}_sparkles", fmt="%.1f%%",
        )

        st.markdown("### SparQ (Silver)")
        srp12_tick("Computing SparQ SRP %")
        show_pivot(
            srp_period_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", period, anchor, brand="Sparq"),
            f"srp_period_{period.lower().replace(' ', '_')}_sparq", fmt="%.1f%%",
        )

# ---------------- Window 13: Style Lookup — why this store? ----------------
if nav == "style_lookup":
    st.subheader("Style Lookup — why should this style go to this store?")
    lookup_month = datetime.now().month
    lookup_month_name = datetime.now().strftime("%B")
    st.write(
        "Paste a Style No to see **every window's numbers for that one style, in window order** — "
        "Stock, Base Stock, Difference, Sales/Memo history, SRP %, Fresh Stock, Seasonal, Month trend, "
        "Zone — all pre-filtered to just this style, instead of checking each tab separately."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "Every section below reuses its **source window's exact calculation** — same function, "
            "same filters, same math — just pre-filtered to the one Style No you enter. Nothing new "
            "is computed here; if a number looks different from the source window, the source window "
            "is the one that's scoped differently (e.g. Window 6 removes Style No entirely, so it has "
            "no equivalent here):\n\n"
            "1. **Current Stock, by store** — Window 1's Stock, filtered to this style.\n"
            "2. **Base Stock (lifetime rate), by store** — Window 2's raw rate, filtered to this style.\n"
            "3. **Difference, by store** — Window 3's Stock − Base Stock, filtered to this style.\n"
            "4. **Sales History by Month, by store** — Window 4's month-by-month distinct Jewel Code "
            "count, filtered to this style.\n"
            "5. **Memo Issue History by Month, by store** — Window 5's equivalent, from memo data.\n"
            "6. **SRP % by Period, by store** — Windows 7/8/12's trailing-window SRP % (Sales ÷ Memo "
            "Issue, anchored to the latest data date), filtered to this style.\n"
            "7. **Fresh Stock pieces** — Window 9's unallocated pieces of this style and their SRP / "
            f"Base Stock Best Store recommendations, scoped to {lookup_month_name}-only history.\n"
            f"8. **Seasonal performance, by store** — Window 9's full per-store ranking for "
            f"{lookup_month_name} across every year on file (same {MIN_SEASONAL_SAMPLES}-year "
            "minimum-sample guard), for *every* store, not just the winner — the detail behind "
            "Window 9's 'Best Store' pick, so you can see who came 2nd and 3rd, and by how much.\n"
            "9. **Month-wise Trend** — Window 10's Jan→Dec pattern, scoped to just this style, across "
            "every store and year, so you can see if it's a seasonal style.\n"
            "10. **Zone Breakdown** — Window 11's Style-wise Zone comparison, filtered to this style.\n\n"
            f"Color coding on SRP % tables matches Windows 7/8/12: 🟩 SRP % ≥ {SRP_HIGH}% with stock "
            f"in hand, 🟥 SRP % < {SRP_LOW}% (weak sell-through), 🟨 everything in between."
        )

    style_query = st.text_input("Style No", placeholder="e.g. T8565, AFDRE10002", key="style_lookup_query").strip().upper()

    if not style_query:
        st.info("Paste a Style No above to look it up.")
    else:
        def _match_style(df, col="Style No"):
            return df[df[col].astype(str).str.strip().str.upper() == style_query]

        style_stock_rows = _match_style(stock_grouped)
        style_stock_rows = style_stock_rows[
            style_stock_rows["Store Name"].isin(selected_stores) & style_stock_rows["Category"].isin(selected_categories)
        ]
        style_fresh_rows = _match_style(fresh_stock) if not fresh_stock.empty else fresh_stock

        found_brands = sorted(set(style_stock_rows["Brand"].dropna()) | set(style_fresh_rows["Brand"].dropna()))
        found_categories = sorted(set(style_stock_rows["Category"].dropna()) | set(style_fresh_rows["Category"].dropna()))
        found_price_points = sorted(set(style_stock_rows["Price Point"].dropna()) | set(style_fresh_rows["Price Point"].dropna()))

        if not found_brands and style_stock_rows.empty and style_fresh_rows.empty:
            st.warning(f"No stock, fresh-stock, or sales records found for Style No '{style_query}'.")
        else:
            st.caption(
                f"**Brand:** {', '.join(found_brands) or '—'} &nbsp;·&nbsp; "
                f"**Category:** {', '.join(found_categories) or '—'} &nbsp;·&nbsp; "
                f"**Price Point:** {', '.join(found_price_points) or '—'}"
            )
            match_keys = ["Style No", "Price Point", "Category"]
            style_row_keys = ROW_KEYS + ["Category"]
            style13_tick = progress_tracker(10)

            # ---- Window 1: Current Stock by store ----
            st.markdown("### 📦 1 — Current Stock, by store")
            style13_tick("Loading Current Stock")
            style_stock_display = (
                style_stock_rows.groupby(["Store Name", "Store Code", "Category"], dropna=False)["Stock"]
                .sum().reset_index().sort_values("Stock", ascending=False)
            )
            if style_stock_display.empty:
                st.info("No current stock of this style at any store.")
            else:
                st.dataframe(
                    style_stock_display, width="stretch",
                    column_config={"Stock": st.column_config.NumberColumn(format="%.0f")},
                )

            st.divider()
            # ---- Window 2: Base Stock (lifetime rate) by store ----
            st.markdown("### 📊 2 — Base Stock (lifetime rate), by store")
            style13_tick("Loading Base Stock")
            any_bs2 = False
            for b in ["Sparkles", "Sparq"]:
                bs2_b = _match_style(brand_pivot(base_stock, "BaseStock", b, fill_value=None))
                if bs2_b.empty:
                    continue
                any_bs2 = True
                st.markdown(f"**{b}**")
                show_pivot(bs2_b, f"lookup_basestock_{b.lower()}_{style_query}", fmt="%.2f")
            if not any_bs2:
                st.info("No Base Stock history for this style.")

            st.divider()
            # ---- Window 3: Difference (Stock - Base Stock) by store ----
            st.markdown("### ➕➖ 3 — Difference (Stock − Base Stock), by store")
            style13_tick("Loading Difference")
            any_diff = False
            for b in ["Sparkles", "Sparq"]:
                diff_b = _match_style(blank_zeros(brand_pivot(diff_long, "Difference", b)))
                if diff_b.empty:
                    continue
                any_diff = True
                st.markdown(f"**{b}**")
                show_pivot(diff_b, f"lookup_diff_{b.lower()}_{style_query}", fmt="%.2f")
            if not any_diff:
                st.info("No Stock/Base Stock difference data for this style.")

            st.divider()
            # ---- Window 4: Sales History by Month, by store ----
            st.markdown("### 📅 4 — Sales History by Month, by store")
            style13_tick("Loading Sales History")
            if sales_calendar is None:
                st.error("reference_sales_calendar.csv not found.")
            else:
                any_sales_cal = False
                for b in ["Sparkles", "Sparq"]:
                    sc_b = _match_style(blank_zeros(calendar_pivot(sales_calendar, b), row_keys=style_row_keys))
                    if sc_b.empty:
                        continue
                    any_sales_cal = True
                    st.markdown(f"**{b}**")
                    show_pivot(sc_b, f"lookup_salescal_{b.lower()}_{style_query}", fmt="%.0f", row_keys=style_row_keys)
                if not any_sales_cal:
                    st.info("No monthly sales history for this style.")

            st.divider()
            # ---- Window 5: Memo Issue History by Month, by store ----
            st.markdown("### 🧾 5 — Memo Issue History by Month, by store")
            style13_tick("Loading Memo Issue History")
            if memo_issue is None:
                st.error("reference_memo_issue.csv not found.")
            else:
                any_memo_cal = False
                for b in ["Sparkles", "Sparq"]:
                    mc_b = _match_style(blank_zeros(calendar_pivot(memo_issue, b), row_keys=style_row_keys))
                    if mc_b.empty:
                        continue
                    any_memo_cal = True
                    st.markdown(f"**{b}**")
                    show_pivot(mc_b, f"lookup_memocal_{b.lower()}_{style_query}", fmt="%.0f", row_keys=style_row_keys)
                if not any_memo_cal:
                    st.info("No monthly memo issue history for this style.")

            st.divider()
            # ---- SRP % by Period, by store (Windows 7/8/12 logic) ----
            st.markdown("### 📈 6 — SRP % by Period, by store")
            if sales_daily is None or memo_daily is None:
                st.error("reference_sales_daily.csv or reference_memo_daily.csv not found.")
            else:
                lookup_anchor = srp_period_anchor(sales_daily, memo_daily)
                st.caption(f"Data as of **{lookup_anchor.strftime('%d %b %Y')}**.")
                lookup_period = st.radio(
                    "Period", list(PERIOD_DAYS.keys()), index=3, horizontal=True, key="srp_period_choice_lookup",
                )
                any_period = False
                for b in ["Sparkles", "Sparq"]:
                    srp_p = srp_period_pivot(sales_daily, memo_daily, match_keys, "Store Name", lookup_period, lookup_anchor, brand=b)
                    srp_p = _match_style(srp_p)
                    if srp_p.empty:
                        continue
                    any_period = True
                    stock_p = value_pivot(stock_grouped, "Stock", match_keys, "Store Name", brand=b)
                    stock_p = _match_style(stock_p)
                    srp_p_ranked = add_best_column(srp_p.copy(), match_keys, label="Best Store")
                    store_cols = [c for c in srp_p_ranked.columns if c not in match_keys + ["Best Store", "Best Store Value"]]
                    st.markdown(f"**{b}**")
                    show_store_pivot(
                        srp_p_ranked, match_keys, f"lookup_period_srp_{b.lower()}_{style_query}", fmt="%.1f%%",
                        styler=style_srp_with_store_stock(srp_p_ranked, stock_p, match_keys, store_cols),
                    )
                if not any_period:
                    st.info("No SRP % data for this style in the selected period.")

            st.divider()
            # ---- Fresh Stock pieces + Window 9 recommendation ----
            st.markdown("### 📦 7 — Fresh Stock — unallocated pieces of this style")
            if fresh_stock.empty:
                st.info("No fresh stock file loaded, or nothing unallocated in it.")
            elif style_fresh_rows.empty:
                st.info("No unallocated pieces of this style right now.")
            elif sales_calendar is None or memo_issue is None:
                st.error("reference_sales_calendar.csv or reference_memo_issue.csv not found.")
            else:
                sales_seasonal = sales_calendar[sales_calendar["Month"] == lookup_month]
                memo_seasonal = memo_issue[memo_issue["Month"] == lookup_month]
                srp_lookup_parts, bs_lookup_parts = [], []
                for b in ["Sparkles", "Sparq"]:
                    srp_b = srp_pivot(
                        sales_seasonal, memo_seasonal, match_keys, "Store Name", brand=b, min_months=MIN_SEASONAL_SAMPLES,
                    )
                    srp_b = add_best_column(srp_b, match_keys, label="SRP Best Store")
                    srp_b["Brand"] = b
                    srp_lookup_parts.append(srp_b[match_keys + ["Brand", "SRP Best Store", "SRP Best Store Value"]])

                    bs_b = seasonal_base_stock_pivot(
                        sales_calendar, match_keys, "Store Name", b, lookup_month, min_years=MIN_SEASONAL_SAMPLES,
                    )
                    bs_b = add_best_column(bs_b, match_keys, label="Base Stock Best Store")
                    bs_b["Brand"] = b
                    bs_lookup_parts.append(bs_b[match_keys + ["Brand", "Base Stock Best Store", "Base Stock Best Store Value"]])

                srp_lookup = pd.concat(srp_lookup_parts, ignore_index=True)
                bs_lookup = pd.concat(bs_lookup_parts, ignore_index=True)
                style_suggestions = style_fresh_rows.merge(srp_lookup, on=match_keys + ["Brand"], how="left")
                style_suggestions = style_suggestions.merge(bs_lookup, on=match_keys + ["Brand"], how="left")
                style_suggestions = style_suggestions.sort_values(
                    ["SRP Best Store Value", "Base Stock Best Store Value"], ascending=False, na_position="last"
                ).reset_index(drop=True)
                display_cols = [
                    "Jewel Code", "Style No", "Brand", "Category", "Price Point", "ItemPcs",
                    "SRP Best Store", "SRP Best Store Value", "Base Stock Best Store", "Base Stock Best Store Value",
                ]
                st.dataframe(
                    style_suggestions[display_cols], width="stretch",
                    column_config={
                        "SRP Best Store Value": st.column_config.NumberColumn(format="%.1f%%"),
                        "Base Stock Best Store Value": st.column_config.NumberColumn(format="%.2f"),
                    },
                )

            st.divider()
            # ---- Seasonal SRP % and Base Stock by store, full breakdown ----
            st.markdown(f"### 🗓️ 8 — Seasonal performance — {lookup_month_name} history, every store")
            st.write("Full store-by-store ranking behind Window 9's recommendation — not just the winner.")
            if sales_calendar is None or memo_issue is None:
                st.error("reference_sales_calendar.csv or reference_memo_issue.csv not found.")
            else:
                sales_seasonal = sales_calendar[sales_calendar["Month"] == lookup_month]
                memo_seasonal = memo_issue[memo_issue["Month"] == lookup_month]
                any_seasonal = False
                for b in ["Sparkles", "Sparq"]:
                    srp_b = srp_pivot(sales_seasonal, memo_seasonal, match_keys, "Store Name", brand=b, min_months=MIN_SEASONAL_SAMPLES)
                    srp_b = _match_style(srp_b)
                    bs_b = seasonal_base_stock_pivot(sales_calendar, match_keys, "Store Name", b, lookup_month, min_years=MIN_SEASONAL_SAMPLES)
                    bs_b = _match_style(bs_b)
                    if srp_b.empty and bs_b.empty:
                        continue
                    any_seasonal = True
                    st.markdown(f"**{b}**")

                    if not srp_b.empty:
                        stock_b = value_pivot(stock_grouped, "Stock", match_keys, "Store Name", brand=b)
                        stock_b = _match_style(stock_b)
                        srp_b_ranked = add_best_column(srp_b.copy(), match_keys, label="Best Store")
                        store_cols = [c for c in srp_b_ranked.columns if c not in match_keys + ["Best Store", "Best Store Value"]]
                        st.caption("Seasonal SRP %")
                        show_store_pivot(
                            srp_b_ranked, match_keys, f"lookup_seasonal_srp_{b.lower()}_{style_query}", fmt="%.1f%%",
                            styler=style_srp_with_store_stock(srp_b_ranked, stock_b, match_keys, store_cols),
                        )
                    else:
                        st.caption(f"No qualifying seasonal SRP % data ({MIN_SEASONAL_SAMPLES}+ {lookup_month_name}s) for this style/brand.")

                    if not bs_b.empty:
                        bs_b_ranked = add_best_column(bs_b.copy(), match_keys, label="Best Store")
                        st.caption("Seasonal Base Stock")
                        show_store_pivot(bs_b_ranked, match_keys, f"lookup_seasonal_bs_{b.lower()}_{style_query}", fmt="%.2f")
                    else:
                        st.caption(f"No qualifying seasonal Base Stock data ({MIN_SEASONAL_SAMPLES}+ {lookup_month_name}s) for this style/brand.")
                if not any_seasonal:
                    st.info(f"No qualifying {lookup_month_name} history ({MIN_SEASONAL_SAMPLES}+ years) for this style at any store.")

            st.divider()
            # ---- Window 10: Month-wise Trend for this style ----
            st.markdown("### 🎉 9 — Month-wise Trend (Jan → Dec, every store and year)")
            if sales_calendar is None:
                st.error("reference_sales_calendar.csv not found.")
            else:
                sales_calendar_style = _match_style(sales_calendar)
                if sales_calendar_style.empty:
                    st.info("No sales history for this style to build a month-wise trend.")
                else:
                    available_years = sorted(sales_calendar_style["Year"].dropna().astype(int).unique().tolist())
                    year_choice = st.selectbox(
                        "Year", ["All years"] + [str(y) for y in available_years],
                        key=f"month_trend_year_{style_query}",
                    )
                    year_key = year_choice.replace(" ", "_")
                    sales_calendar_style_year = (
                        sales_calendar_style if year_choice == "All years"
                        else sales_calendar_style[sales_calendar_style["Year"] == int(year_choice)]
                    )
                    any_month_trend = False
                    for b in ["Sparkles", "Sparq"]:
                        mt_b = month_category_pivot(sales_calendar_style_year, b, pivot_col="Category")
                        cat_cols_mt = [c for c in mt_b.columns if c != "Month"]
                        if mt_b[cat_cols_mt].sum().sum() == 0:
                            continue
                        any_month_trend = True
                        st.markdown(f"**{b}**")
                        show_pivot(
                            blank_zeros(mt_b, row_keys=["Month"]),
                            f"lookup_month_trend_{b.lower()}_{style_query}_{year_key}",
                            fmt="%.0f", row_keys=["Month"],
                        )
                        st.bar_chart(mt_b.set_index("Month")[cat_cols_mt])
                    if not any_month_trend:
                        no_data_scope = "" if year_choice == "All years" else f" in {year_choice}"
                        st.info(f"No sales history for this style in either brand{no_data_scope}.")

            st.divider()
            # ---- Window 11: Zone Breakdown for this style ----
            st.markdown("### 🗺️ 10 — Zone Breakdown")
            if sales_calendar is None:
                st.error("reference_sales_calendar.csv not found.")
            elif "Zone" not in sales_calendar.columns:
                st.error(
                    "This reference data was built before Zone tracking was added. "
                    "Click 'Recompute Base Stock' in the sidebar to rebuild it with Zone included."
                )
            else:
                zone_row_keys = ["Style No", "Category"]
                any_zone = False
                for b in ["Sparkles", "Sparq"]:
                    zone_b = _match_style(value_pivot(sales_calendar, "JewelCodeCount", zone_row_keys, "Zone", brand=b))
                    if zone_b.empty:
                        continue
                    any_zone = True
                    zone_b = add_best_column(zone_b, zone_row_keys, label="Best Zone")
                    st.markdown(f"**{b}**")
                    show_store_pivot(zone_b, zone_row_keys, f"lookup_zone_{b.lower()}_{style_query}", fmt="%.0f", best_label="Best Zone")
                if not any_zone:
                    st.info("No zone-level sales data for this style.")
