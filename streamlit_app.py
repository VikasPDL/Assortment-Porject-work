import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Some pivots (e.g. SRP-with-row-stock) exceed pandas Styler's default
# render cap (262,144 cells) once styled — raise it so those still render.
pd.set_option("styler.render.max_elements", 2_000_000)

from api_client import apply_fixed_mapping, fetch_api_dataframe
from core import (
    STOCK_COLS, TRANSACTION_COLS, brand_from_base_metal, compute_piece_lifecycle, load_store_lookup, process_all_pieces,
    process_fresh_stock, process_stock_file,
)
from reference_pipeline import (
    MEMO_API_OUT, MEMO_RETURN_API_OUT, META_OUT, SALES_API_OUT, SALES_RETURN_API_OUT, STOCK_API_OUT, STORE_DETAIL,
    build_reference,
)

# Bump this whenever core.py/reference_pipeline.py processing logic changes,
# so a stale cached result (same file, old columns) never lingers in a
# browser session across a code update — the cache key below depends on it.
PIPELINE_VERSION = "7-memo-return"

# Saved API URL/token per API — see the sidebar's API sections. Not
# tracked in git (contains live credentials); listed in .gitignore.
STOCK_API_CONFIG = "stock_api_config.json"
SALES_API_CONFIG = "sales_api_config.json"
MEMO_API_CONFIG = "memo_issue_api_config.json"
MEMO_RETURN_API_CONFIG = "memo_return_api_config.json"
SALES_RETURN_API_CONFIG = "sales_return_api_config.json"

# Sales/Memo Issue/Memo Return are always fetched "till date" — from this
# fixed start (before the earliest data on file) through today — so there's
# no date-range picker to fill in every time.
API_FETCH_START_DATE = datetime(2019, 1, 1).date()

st.set_page_config(page_title="Assortment Stock & Base Stock", layout="wide", initial_sidebar_state="collapsed")

# `initial_sidebar_state="collapsed"` above only applies the first time a
# browser has ever opened this app — Streamlit's frontend then remembers
# the sidebar's last position in that browser's localStorage (key
# "stSidebarCollapsed-<hash>") and that stored value wins over this
# setting on every later visit, including a hard refresh. Clearing that
# key on every load forces every visit to behave like a first one, so the
# sidebar always starts collapsed regardless of what a previous session
# left behind — a user can still freely expand it during a session, this
# only resets the *next* page load's starting state. This relies on
# Streamlit's internal key naming (found by inspecting its JS bundle, not
# a documented API), so it could need updating if a future Streamlit
# version changes that internal naming.
components.html(
    """
    <script>
    try {
        const ls = window.parent.localStorage;
        Object.keys(ls).forEach((k) => {
            if (k.startsWith("stSidebarCollapsed-")) ls.removeItem(k);
        });
    } catch (e) {}
    </script>
    """,
    height=0,
)

st.markdown(
    """
    <style>
    /* ==========================================================================
       Design tokens — corporate/professional: light surfaces, single indigo
       accent, dark navy sidebar. Matches app-ui-reference_1.html.
       ========================================================================== */
    :root {
        --bg: #f4f6f9;
        --surface: #ffffff;
        --surface-alt: #f8fafc;
        --border: #e2e8f0;
        --border-strong: #cbd5e1;

        --text: #1e2a3b;
        --text-muted: #64748b;
        --text-faint: #94a3b8;

        --sidebar-bg: #101826;
        --sidebar-text: #cbd5e1;
        --sidebar-muted: #7c8aa0;

        --accent: #2f5dff;
        --accent-hover: #2549d1;
        --accent-soft: #eaefff;

        --success: #16a34a;
        --success-soft: #e8f8ee;
        --warning: #d97706;
        --warning-soft: #fef3e2;
        --danger: #dc2626;
        --danger-soft: #fdecec;

        --radius-sm: 6px;
        --radius-md: 10px;
        --radius-lg: 14px;
        --shadow-sm: 0 1px 2px rgba(16,24,38,.06);
        --shadow-md: 0 4px 12px rgba(16,24,38,.08);

        --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    html, body, [class^="css"], [class*=" css"] { font-family: var(--font); }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: var(--font) !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        color: var(--text) !important;
    }
    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.5rem !important; }
    h3 { font-size: 1.15rem !important; }

    /* ---- Font sizes, bumped up app-wide for readability ---- */
    html, body, .stApp, [data-testid="stAppViewContainer"] { font-size: 17px !important; }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span { font-size: 1rem !important; line-height: 1.55 !important; }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { font-size: 0.92rem !important; }
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p { font-size: 1rem !important; }
    div[data-testid="stMetricValue"] { font-size: 2.1rem !important; }
    div[data-testid="stMetricLabel"] { font-size: 1rem !important; }
    div[data-testid="stDataFrame"] * { font-size: 0.95rem !important; }
    section[data-testid="stSidebar"] * { font-size: 1rem !important; }
    div[data-testid="stExpander"] summary p { font-size: 1.02rem !important; }
    div[data-baseweb="select"] * , div[data-baseweb="input"] input { font-size: 1rem !important; }
    div[data-testid="stTabs"] button p { font-size: 1rem !important; }
    label[data-testid="stWidgetLabel"] p { font-size: 1rem !important; }

    .stApp { background: var(--bg) !important; }

    /* ---- Home tile cards ---- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: var(--radius-lg) !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow-sm);
        transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        padding: 4px 2px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-color: var(--accent) !important;
    }

    /* ---- Buttons: solid indigo, corporate flat ---- */
    div[data-testid="stButton"] button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: background 0.12s ease;
    }
    div[data-testid="stButton"] button:hover {
        background: var(--accent-hover) !important;
        color: #ffffff !important;
    }
    div[data-testid="stButton"] button p { font-weight: 600 !important; }
    /* Disabled buttons must look visibly different from enabled ones —
       without this override they inherit the same solid blue and look
       clickable even when they aren't (e.g. Fetch buttons before the
       "Allow live API calls" toggle is switched on). */
    div[data-testid="stButton"] button:disabled,
    div[data-testid="stButton"] button:disabled:hover {
        background: var(--surface-alt) !important;
        color: var(--text-faint) !important;
        border: 1px solid var(--border) !important;
        cursor: not-allowed !important;
    }

    .home-eyebrow {
        font-family: var(--font);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 2px;
    }
    .tile-icon {
        width: 48px; height: 48px; border-radius: var(--radius-md);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem; margin-bottom: 10px;
    }
    .tile-title { font-size: 1.02rem; font-weight: 700; margin-bottom: 4px; color: var(--text); }
    .tile-desc { font-size: 0.83rem; color: var(--text-muted); min-height: 44px; line-height: 1.4; }

    /* ---- Dividers ---- */
    hr {
        border: none !important;
        height: 1px !important;
        background: var(--border) !important;
        margin: 1.3rem 0 !important;
    }

    /* ---- Data tables ---- */
    div[data-testid="stDataFrame"], div[data-testid="stDataEditor"] {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    /* ---- HTML pivot tables (dark header, white header text) ----
       st.dataframe's header is drawn on an internal canvas grid, which
       can't be recolored with CSS — pivot tables are rendered as real
       HTML tables instead so the header can actually be styled. */
    .html-table-wrap {
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-sm);
        overflow: auto;
        max-height: 480px;
        margin-bottom: 0.5rem;
    }
    .html-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .html-table thead th {
        position: sticky; top: 0;
        background: var(--sidebar-bg) !important;
        color: #ffffff !important;
        text-align: left;
        font-weight: 700;
        padding: 0.55rem 0.9rem;
        white-space: nowrap;
        border-bottom: 1px solid var(--sidebar-bg);
    }
    .html-table tbody td, .html-table tbody th {
        padding: 0.5rem 0.9rem;
        border-bottom: 1px solid var(--border);
        color: var(--text);
        font-weight: 400;
        text-align: left;
        white-space: nowrap;
    }
    .html-table tbody tr:last-child td, .html-table tbody tr:last-child th { border-bottom: none; }
    .html-table tbody tr:hover td, .html-table tbody tr:hover th { background: var(--surface-alt); }

    /* ---- Expanders ---- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        background: var(--surface) !important;
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: var(--text) !important;
        padding: 0.5rem 0.9rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: var(--surface-alt) !important;
    }
    /* Home page section labels (Other API / Inventory / Sales & Performance /
       Reports & Lookup) — bold and smaller than the default expander label. */
    .st-key-home_group_expanders_other_api div[data-testid="stExpander"] summary p,
    .st-key-home_group_expanders_tiles div[data-testid="stExpander"] summary p {
        font-weight: 700 !important;
        font-size: 0.8rem !important;
    }

    /* ---- Info / success / warning / error boxes ---- */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border) !important;
        background: var(--surface) !important;
        padding: 0.85rem 1.05rem !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stAlert"] p { font-size: 0.92rem !important; color: var(--text) !important; }

    /* ---- Metric tiles ---- */
    div[data-testid="stMetric"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        padding: 0.9rem 1.1rem !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.74rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted) !important;
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--font) !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
    }

    /* ---- Sidebar: dark navy, matches the reference's console sidebar ---- */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: var(--font) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover {
        background: rgba(255,255,255,0.06) !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div,
    section[data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] p { color: var(--sidebar-text) !important; }

    /* ---- Text inputs, selects, textareas (main content) ---- */
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
    .stTextArea textarea {
        background: var(--surface) !important;
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text) !important;
    }
    div[data-baseweb="select"] > div:focus-within, .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-soft) !important;
    }

    /* Multiselect chips */
    span[data-baseweb="tag"] {
        background: var(--accent) !important;
        border-radius: var(--radius-sm) !important;
    }

    /* ---- Download button: outline style, matches .btn-secondary ---- */
    div[data-testid="stDownloadButton"] button {
        background: var(--surface) !important;
        border: 1px solid var(--border-strong) !important;
        color: var(--text) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: var(--surface-alt) !important;
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ---- Captions ---- */
    [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

    /* ---- Home tile grid section headers (Inventory / Sales & Performance / ...) ---- */
    .tile-group-label {
        color: var(--text-faint);
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 1.4rem 0 0.6rem 0.1rem;
        padding-top: 0.6rem;
        border-top: 1px solid var(--border);
    }
    .tile-group-label:first-of-type { border-top: none; padding-top: 0; margin-top: 0.4rem; }
    .choose-window-note {
        color: var(--text-muted);
        font-size: 0.78rem;
        margin: 0.9rem 0 0.3rem 0.1rem;
    }
    .choose-window-note b { color: var(--text); }

    /* ---- Stat cards (colored icon badge, matches app-ui-reference_1.html) ---- */
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 0.5rem 0.65rem;
        box-shadow: var(--shadow-sm);
        height: 100%;
    }
    /* min-height keeps the icon/label row the same height across every
       card in a row, so values line up even when one label wraps to 2
       lines and its neighbor doesn't. */
    .stat-card-top { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.3rem; margin-bottom: 0.3rem; min-height: 1.4rem; }
    .stat-card-label {
        color: var(--text-muted);
        font-size: 0.58rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        line-height: 1.25;
        padding-top: 0.1rem;
    }
    .stat-card-icon {
        width: 1.3rem; height: 1.3rem;
        border-radius: var(--radius-sm);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.7rem;
        flex-shrink: 0;
    }
    .stat-card-value { color: var(--text); font-size: 0.95rem; font-weight: 700; line-height: 1.2; word-break: break-word; }
    .stat-card-value.small { font-size: 0.72rem; }
    .stat-card-sub { color: var(--text-faint); font-size: 0.6rem; margin-top: 0.1rem; }

    /* ---- Page header (breadcrumb + title, matches app-ui-reference_1.html) ---- */
    .page-breadcrumb {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.1rem;
    }
    .page-title {
        color: var(--text);
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        margin-bottom: 0.4rem;
    }

    /* ---- Data freshness readout, next to the title (always visible, small text) ---- */
    .freshness-panel { padding-top: 0.2rem; }
    .freshness-row {
        font-size: 0.72rem;
        color: var(--text-faint);
        line-height: 1.55;
        white-space: nowrap;
    }
    .freshness-row b { color: var(--text-muted); font-weight: 600; }

    /* ---- Status badges (pill), matches app-ui-reference_1.html's Recent Files table ---- */
    .badge {
        display: inline-block;
        padding: 0.18rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .badge-green { background: var(--success-soft); color: var(--success); }
    .badge-amber { background: var(--warning-soft); color: var(--warning); }
    .badge-red { background: var(--danger-soft); color: var(--danger); }
    .badge-gray { background: var(--surface-alt); color: var(--text-muted); border: 1px solid var(--border); }

    /* ---- Topbar-style header row action buttons (Reference Data info, Refresh) ---- */
    .st-key-topbar_action_info div[data-testid="stButton"] button,
    .st-key-topbar_action_info div[data-testid="stPopover"] button,
    .st-key-topbar_action_refresh div[data-testid="stButton"] button,
    .st-key-topbar_action_refresh div[data-testid="stPopover"] button {
        width: 1.7rem !important; height: 1.7rem !important;
        min-width: 1.7rem !important;
        padding: 0 !important;
        border-radius: var(--radius-sm) !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-muted) !important;
        box-shadow: none !important;
        font-size: 0.8rem !important;
    }
    .st-key-topbar_action_info div[data-testid="stButton"] button p,
    .st-key-topbar_action_info div[data-testid="stPopover"] button p,
    .st-key-topbar_action_refresh div[data-testid="stButton"] button p,
    .st-key-topbar_action_refresh div[data-testid="stPopover"] button p {
        font-size: 0.8rem !important;
    }
    .st-key-topbar_action_info div[data-testid="stButton"] button:hover,
    .st-key-topbar_action_info div[data-testid="stPopover"] button:hover,
    .st-key-topbar_action_refresh div[data-testid="stButton"] button:hover,
    .st-key-topbar_action_refresh div[data-testid="stPopover"] button:hover {
        background: var(--surface-alt) !important;
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_stat_card(label, value, icon, color, sublabel=None):
    """Custom KPI tile with a colored icon badge, matching the corporate
    reference design's stat-card pattern more closely than a bare st.metric.
    Long values (e.g. a timestamp) drop to a smaller font so they don't wrap
    across several lines in a narrow card."""
    sub_html = f'<div class="stat-card-sub">{sublabel}</div>' if sublabel else ""
    value_class = "stat-card-value small" if len(str(value)) > 10 else "stat-card-value"
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-top">
                <span class="stat-card-label">{label}</span>
                <span class="stat-card-icon" style="background:{color}1a; color:{color};">{icon}</span>
            </div>
            <div class="{value_class}">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(num_label, title):
    """Breadcrumb + bold title + small icon buttons (Reference Data status,
    Refresh) on the right, matching the reference design's topbar row (used
    in place of a plain st.subheader for every window)."""
    head_col, info_col, action_col = st.columns([7, 1, 1])
    with head_col:
        st.markdown(
            f"""
            <div class="page-breadcrumb">Windows / {num_label}</div>
            <div class="page-title">{title}</div>
            """,
            unsafe_allow_html=True,
        )
    with info_col:
        with st.container(key="topbar_action_info"):
            with st.popover("🕒", help="Data freshness — when each source was last refreshed"):
                st.markdown("**Data freshness**")
                if os.path.exists(META_OUT):
                    with open(META_OUT) as f:
                        info_meta = json.load(f)
                    refreshed = datetime.fromisoformat(info_meta["last_refreshed"]).strftime("%b %d, %H:%M")
                    st.markdown(f"**Base Stock reference:** {refreshed}")
                    st.caption(f"{info_meta['base_stock_rows']:,} base-stock rows")
                else:
                    st.caption("Base Stock reference: not built yet.")

                st.divider()
                for label, path in [
                    ("Stock", STOCK_API_OUT),
                    ("Sales", SALES_API_OUT),
                    ("Sales Return", SALES_RETURN_API_OUT),
                    ("Memo Issue", MEMO_API_OUT),
                    ("Memo Return", MEMO_RETURN_API_OUT),
                ]:
                    if os.path.exists(path):
                        stamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%b %d, %H:%M")
                        st.markdown(f"**{label}:** {stamp}")
                    else:
                        st.markdown(f"**{label}:** *never fetched*")
    with action_col:
        with st.container(key="topbar_action_refresh"):
            if st.button("🔄", key=f"refresh_{num_label}", help="Refresh"):
                st.rerun()


def render_html_table(df_or_styler, max_height=480):
    """Render a dataframe (or an already-formatted/colored Styler) as a
    real HTML table instead of st.dataframe. st.dataframe's header is
    drawn on an internal canvas grid and can't be recolored with CSS —
    a real <table> can, so this is what gives every pivot table its dark
    header. Any background colors already applied via a Styler (e.g. the
    SRP % traffic-light coloring) are preserved as-is."""
    html = df_or_styler.to_html() if hasattr(df_or_styler, "to_html") else df_or_styler
    html = re.sub(r"<table[^>]*>", '<table class="html-table">', html, count=1)
    st.markdown(f'<div class="html-table-wrap" style="max-height:{max_height}px;">{html}</div>', unsafe_allow_html=True)

title_col, stock_btn_col, freshness_col = st.columns([5, 1, 2])
with title_col:
    st.markdown('<div class="home-eyebrow">Poddar Diamonds · Assortment Intelligence</div>', unsafe_allow_html=True)
    st.title("Assortment Stock & Base Stock Dashboard")
# Filled in further down, once api_calls_enabled and fetch_stock_api both
# exist — st.empty() reserves this exact spot in the layout now so the
# button renders here, next to the title, on every window.
stock_btn_slot = stock_btn_col.empty()
with freshness_col:
    freshness_rows = "".join(
        f'<div class="freshness-row"><b>{label}:</b> '
        f'{datetime.fromtimestamp(os.path.getmtime(path)).strftime("%b %d, %H:%M") if os.path.exists(path) else "never fetched"}</div>'
        for label, path in [
            ("Stock", STOCK_API_OUT),
            ("Sales", SALES_API_OUT),
            ("Sales Return", SALES_RETURN_API_OUT),
            ("Memo Issue", MEMO_API_OUT),
            ("Memo Return", MEMO_RETURN_API_OUT),
        ]
    )
    st.markdown(f'<div class="freshness-panel">{freshness_rows}</div>', unsafe_allow_html=True)

# ---------------- Window registry + sidebar nav list ----------------
# One row per window, driven by clicking a home tile or a sidebar nav
# button — both just set st.session_state["nav"] and rerun. Only the
# active window is rendered (not every pivot on every rerun).
WINDOW_TILES = [
    ("stock", "📦", "Window 1", "Stock", "Current stock pieces by store, style, and category."),
    ("basestock", "📊", "Window 2", "Base Stock", "Average monthly sell-through rate per style — what 'normal' stocking looks like."),
    ("diff", "➕➖", "Window 3", "Difference", "Stock minus Base Stock — spot surplus and shortfall at a glance."),
    ("sales_cal", "📅", "Window 4", "Sales", "Month-by-month sales count, per store and style — check old sales trends here."),
    ("flag", "🧾", "Window 5", "Issued DC", "Month-by-month pieces issued to each store on memo — stock outward to store, sent from HO to store."),
    ("no_style", "🏬", "Window 6", "Store Summary (No Style)", "Base Stock and Difference rolled up to store level, styles combined."),
    ("srp", "🎯", "Window 7", "Store by SRP %", "Sell-through % over a period you pick, colored red / yellow / green."),
    ("store_compare", "🏆", "Window 8", "Store Comparison", "SRP %, Base Stock, and Current Stock side-by-side across every store."),
    ("seasonal", "🎉", "Window 10", "Seasonal Trends", "Month and season best-sellers, mapped to India's festival calendar."),
    ("zone", "🗺️", "Window 11", "Area (Zone) Sales", "Sales performance rolled up by Zone instead of individual store."),
    ("style_lookup", "🔎", "Window 13", "Style Lookup", "Paste one Style No, see every window's numbers for it in one place."),
    ("jewel_allocate", "🧭", "Window 14", "Stock Assortment Summary", "Paste or upload Jewel Codes to find the best SRP store for each, plus their seasonal sales history for a month you pick."),
]
WINDOW_LOOKUP = {key: (icon, num, title) for key, icon, num, title, _ in WINDOW_TILES}
TILE_ACCENTS = ["#8B5CF6", "#EC4899", "#F59E0B", "#22C55E", "#3B82F6", "#F43F5E", "#14B8A6"]

if "nav" not in st.session_state:
    st.session_state["nav"] = "home"


def _go_to(key):
    st.session_state["nav"] = key


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


def _load_api_config(path):
    """Checks Streamlit Secrets first (section name = the config filename
    without .json, e.g. "stock_api_config") — this is what makes saved
    credentials survive a redeploy on Streamlit Community Cloud, where the
    local JSON file below doesn't persist (its filesystem resets on every
    restart). Falls back to the local file, which is all local dev needs."""
    section = os.path.splitext(os.path.basename(path))[0]
    try:
        if section in st.secrets:
            sec = st.secrets[section]
            return {"url": sec.get("url", ""), "auth_name": sec.get("auth_name", "AuthorizationToken"), "auth_value": sec.get("auth_value", "")}
    except Exception:
        pass  # no secrets.toml configured — normal for local dev
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_api_config(path, url, auth_name, auth_value):
    with open(path, "w") as f:
        json.dump({"url": url, "auth_name": auth_name, "auth_value": auth_value}, f)


# All 5 fetch buttons (Stock + the 4 transaction APIs) hit the same
# PoddarDiamonds backend host. Without a shared lock, two team members
# clicking Fetch around the same time — or one person double-clicking —
# could fire overlapping requests at that server. A plain st.session_state
# flag wouldn't catch this: each browser tab gets its own session state, so
# a lock has to live in a shared file on disk instead, visible to every
# session hitting this same running app.
FETCH_LOCK_FILE = "api_fetch_lock.json"
API_FETCH_COOLDOWN_SECONDS = 30


def _claim_fetch_lock(label):
    """Raises if another fetch (any of the 5) claimed the lock less than
    API_FETCH_COOLDOWN_SECONDS ago; otherwise claims it for this fetch."""
    now = datetime.now()
    if os.path.exists(FETCH_LOCK_FILE):
        with open(FETCH_LOCK_FILE) as f:
            lock = json.load(f)
        elapsed = (now - datetime.fromisoformat(lock["at"])).total_seconds()
        if elapsed < API_FETCH_COOLDOWN_SECONDS:
            wait = int(API_FETCH_COOLDOWN_SECONDS - elapsed)
            raise ValueError(
                f"{lock['label']} was just fetched {int(elapsed)}s ago — wait {wait}s before fetching "
                f"{label}, so we don't send overlapping requests to the server."
            )
    with open(FETCH_LOCK_FILE, "w") as f:
        json.dump({"label": label, "at": now.isoformat()}, f)


def fetch_transaction_api(label, config_path, out_path):
    """Shared by every date-ranged transaction API (Sales, Sales Return,
    Memo Issue, Memo Return): load its saved URL/token, fetch till date
    (API_FETCH_START_DATE through today), map, and save. Returns the row
    count fetched. Raises on any failure — the caller decides how to show
    it (inline st.error in the sidebar, a toast on the Home page, etc.)."""
    _claim_fetch_lock(label)
    cfg = _load_api_config(config_path)
    if not cfg.get("url"):
        raise ValueError("No URL saved for this API yet — enter it in the sidebar's API Settings first.")
    headers = build_api_headers(cfg.get("auth_name", ""), cfg.get("auth_value", ""), API_FETCH_START_DATE, datetime.now().date())
    raw_df = fetch_api_dataframe(cfg["url"], headers=headers)
    mapped = apply_fixed_mapping(raw_df, TRANSACTION_COLS)
    mapped.to_csv(out_path, index=False)
    return len(mapped)


def fetch_stock_api():
    """Fetch, map, save, and load the Stock API snapshot into session
    state — shared by the sidebar's Stock API button and the quick-access
    button next to the title. No date range (always a full current
    snapshot). Returns the row count fetched. Raises on failure."""
    _claim_fetch_lock("Stock")
    cfg = _load_api_config(STOCK_API_CONFIG)
    if not cfg.get("url"):
        raise ValueError("No URL saved for the Stock API yet — enter it in the sidebar's Stock API section first.")
    headers = build_api_headers(cfg.get("auth_name", ""), cfg.get("auth_value", ""))
    raw_df = fetch_api_dataframe(cfg["url"], headers=headers)
    mapped = apply_fixed_mapping(raw_df, STOCK_COLS)
    mapped.to_csv(STOCK_API_OUT, index=False)
    store_codes, store_lookup = load_store_lookup()
    st.session_state["stock_grouped"] = process_stock_file(mapped, store_codes, store_lookup)
    st.session_state["fresh_stock"] = process_fresh_stock(mapped)
    st.session_state["all_pieces"] = process_all_pieces(mapped, store_codes, store_lookup)
    st.session_state["file_key"] = f"api-fetch-{datetime.now().isoformat()}"
    return len(mapped)


# ---------------- Sidebar: reference data + upload ----------------
with st.sidebar:
    st.header("Reference Data (Base Stock)")
    if os.path.exists(META_OUT):
        with open(META_OUT) as f:
            meta = json.load(f)
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
    with st.expander("🔒 API Access", expanded=False):
        api_calls_enabled = st.toggle(
            "🔓 Allow live API calls", value=False, key="api_calls_enabled",
            help="Off by default so a stray click never hits the live system. Turn this on only when you actually want to fetch. Gates every Fetch button below, in both API sections.",
        )
        if not api_calls_enabled:
            st.caption("🔒 API calls are off — every Fetch button below is disabled. Turn the toggle on above when you're ready to fetch.")

    st.divider()
    with st.expander("⚙️ API Settings"):
        st.caption(
            "Each API's URL and token are saved to their own file next to the app (not `.env`, not committed "
            "to git) as soon as you fetch — no retyping next session. Always fetches **till date**: from a "
            f"fixed start ({API_FETCH_START_DATE.strftime('%d-%b-%Y')}) through today, no date range to pick."
        )

        # ---- Sales API ----
        st.markdown("**Sales API**")
        sales_saved = _load_api_config(SALES_API_CONFIG)
        sales_api_url = st.text_input("Sales API URL", value=sales_saved.get("url", ""), key="sales_api_url")
        sales_auth_name = st.text_input("Auth Header Name", value=sales_saved.get("auth_name", "AuthorizationToken"), key="sales_auth_name")
        sales_auth_value = st.text_input("Auth Header Value", value=sales_saved.get("auth_value", ""), key="sales_auth_value", type="password")
        if st.button("Fetch Sales from API", width="stretch", disabled=not api_calls_enabled):
            _save_api_config(SALES_API_CONFIG, sales_api_url, sales_auth_name, sales_auth_value)
            try:
                with st.spinner("Fetching Sales API..."):
                    n = fetch_transaction_api("Sales", SALES_API_CONFIG, SALES_API_OUT)
                st.success(f"Fetched and saved {n} rows. Click 'Recompute Base Stock' above to merge them in.")
                st.toast(f"✅ Sales API worked — {n} rows fetched", icon="✅")
            except Exception as e:
                st.error(f"Sales API fetch failed: {e}")
                st.toast(f"❌ Sales API failed: {e}", icon="❌")

        st.divider()

        # ---- Sales Return API ----
        st.markdown("**Sales Return API**")
        st.caption("Customer returns of already-sold items — different from Memo Return (a store handing back unsold consignment stock). Saved for reference; not yet merged into Base Stock/SRP % calculations.")
        sales_return_saved = _load_api_config(SALES_RETURN_API_CONFIG)
        sales_return_api_url = st.text_input("Sales Return API URL", value=sales_return_saved.get("url", ""), key="sales_return_api_url")
        sales_return_auth_name = st.text_input("Auth Header Name", value=sales_return_saved.get("auth_name", "AuthorizationToken"), key="sales_return_auth_name")
        sales_return_auth_value = st.text_input("Auth Header Value", value=sales_return_saved.get("auth_value", ""), key="sales_return_auth_value", type="password")
        if st.button("Fetch Sales Return from API", width="stretch", disabled=not api_calls_enabled):
            _save_api_config(SALES_RETURN_API_CONFIG, sales_return_api_url, sales_return_auth_name, sales_return_auth_value)
            try:
                with st.spinner("Fetching Sales Return API..."):
                    n = fetch_transaction_api("Sales Return", SALES_RETURN_API_CONFIG, SALES_RETURN_API_OUT)
                st.success(f"Fetched and saved {n} rows to {SALES_RETURN_API_OUT}.")
                st.toast(f"✅ Sales Return API worked — {n} rows fetched", icon="✅")
            except Exception as e:
                st.error(f"Sales Return API fetch failed: {e}")
                st.toast(f"❌ Sales Return API failed: {e}", icon="❌")

        st.divider()

        # ---- Memo Issue API ----
        st.markdown("**Memo Issue API**")
        memo_saved = _load_api_config(MEMO_API_CONFIG)
        memo_api_url = st.text_input("Memo Issue API URL", value=memo_saved.get("url", ""), key="memo_api_url")
        memo_auth_name = st.text_input("Auth Header Name", value=memo_saved.get("auth_name", "AuthorizationToken"), key="memo_auth_name")
        memo_auth_value = st.text_input("Auth Header Value", value=memo_saved.get("auth_value", ""), key="memo_auth_value", type="password")
        if st.button("Fetch Memo Issue from API", width="stretch", disabled=not api_calls_enabled):
            _save_api_config(MEMO_API_CONFIG, memo_api_url, memo_auth_name, memo_auth_value)
            try:
                with st.spinner("Fetching Memo Issue API..."):
                    n = fetch_transaction_api("Memo Issue", MEMO_API_CONFIG, MEMO_API_OUT)
                st.success(f"Fetched and saved {n} rows. Click 'Recompute Base Stock' above to merge them in.")
                st.toast(f"✅ Memo Issue API worked — {n} rows fetched", icon="✅")
            except Exception as e:
                st.error(f"Memo Issue API fetch failed: {e}")
                st.toast(f"❌ Memo Issue API failed: {e}", icon="❌")

        st.divider()

        # ---- Memo Return API ----
        st.markdown("**Memo Return API**")
        st.caption("Netted against Memo Issue everywhere SRP % is calculated: SRP % = Sales ÷ (Memo Issue − Memo Return).")
        memo_return_saved = _load_api_config(MEMO_RETURN_API_CONFIG)
        memo_return_api_url = st.text_input("Memo Return API URL", value=memo_return_saved.get("url", ""), key="memo_return_api_url")
        memo_return_auth_name = st.text_input("Auth Header Name", value=memo_return_saved.get("auth_name", "AuthorizationToken"), key="memo_return_auth_name")
        memo_return_auth_value = st.text_input("Auth Header Value", value=memo_return_saved.get("auth_value", ""), key="memo_return_auth_value", type="password")
        if st.button("Fetch Memo Return from API", width="stretch", disabled=not api_calls_enabled):
            _save_api_config(MEMO_RETURN_API_CONFIG, memo_return_api_url, memo_return_auth_name, memo_return_auth_value)
            try:
                with st.spinner("Fetching Memo Return API..."):
                    n = fetch_transaction_api("Memo Return", MEMO_RETURN_API_CONFIG, MEMO_RETURN_API_OUT)
                st.success(f"Fetched and saved {n} rows. Click 'Recompute Base Stock' above to merge them in.")
                st.toast(f"✅ Memo Return API worked — {n} rows fetched", icon="✅")
            except Exception as e:
                st.error(f"Memo Return API fetch failed: {e}")
                st.toast(f"❌ Memo Return API failed: {e}", icon="❌")

    st.divider()
    with st.expander("📦 Stock API"):
        st.caption(
            "No date field, always a full current snapshot — and no upload needed. Fetch once and it's "
            "saved to disk; every load after that (even a fresh server restart) reuses that same snapshot "
            "automatically until you click Fetch again."
        )
        if os.path.exists(STOCK_API_OUT):
            st.caption(f"Current snapshot: **{datetime.fromtimestamp(os.path.getmtime(STOCK_API_OUT)).strftime('%b %d, %H:%M')}**")

        stock_api_saved = _load_api_config(STOCK_API_CONFIG)
        stock_api_url = st.text_input("Stock API URL", value=stock_api_saved.get("url", ""), key="stock_api_url")
        stock_auth_name = st.text_input("Auth Header Name", value=stock_api_saved.get("auth_name", "AuthorizationToken"), key="stock_auth_name")
        stock_auth_value = st.text_input("Auth Header Value", value=stock_api_saved.get("auth_value", ""), key="stock_auth_value", type="password")
        if st.button("Fetch Stock from API", width="stretch", disabled=not api_calls_enabled):
            _save_api_config(STOCK_API_CONFIG, stock_api_url, stock_auth_name, stock_auth_value)
            try:
                with st.spinner("Fetching Stock API..."):
                    n = fetch_stock_api()
                st.success(f"Fetched and saved {n} rows from Stock API.")
                st.toast(f"✅ Stock API worked — {n} rows fetched", icon="✅")
                st.rerun()
            except Exception as e:
                st.error(f"Stock API fetch failed: {e}")
                st.toast(f"❌ Stock API failed: {e}", icon="❌")

    st.divider()
    with st.expander("🏪 Store Master"):
        st.caption(
            f"The single source of truth for every store this app knows about (`{STORE_DETAIL}`) — "
            "Store Name, Store Code, Grade, and Zone everywhere in the app come from here."
        )

        if not os.path.exists(STORE_DETAIL):
            st.error(f"{STORE_DETAIL} not found.")
        else:
            store_master_df = pd.read_excel(STORE_DETAIL)
            store_master_df = store_master_df.rename(columns={"Unnamed: 5": "Store Type"})

            st.markdown(f"**Registered stores ({len(store_master_df)})**")
            store_master_search = st.text_input("Search by store name or code", key="store_master_search")
            store_master_display = store_master_df
            if store_master_search:
                mask = (
                    store_master_df["store_name"].astype(str).str.contains(store_master_search, case=False, na=False)
                    | store_master_df["store_code"].astype(str).str.contains(store_master_search, case=False, na=False)
                )
                store_master_display = store_master_df[mask]
            render_html_table(store_master_display, max_height=180)
            store_master_csv = store_master_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full store list (CSV)", store_master_csv, "store_master.csv", "text/csv", key="dl_store_master",
            )

            st.divider()
            st.markdown("**Add a new store**")
            st.caption("Grade / Store Potential / Zone / Store Type default to existing values — pick '➕ Other' to type a new one.")

            def _select_or_custom(label, options, key):
                choice = st.selectbox(label, options + ["➕ Other (type new)"], key=f"{key}_select")
                if choice == "➕ Other (type new)":
                    return st.text_input(f"New {label}", key=f"{key}_custom").strip()
                return choice

            existing_grades = sorted(store_master_df["grade"].dropna().unique().tolist())
            existing_potentials = sorted(store_master_df["store_potential"].dropna().unique().tolist())
            existing_zones = sorted(store_master_df["Zone"].dropna().unique().tolist())
            existing_types = sorted(store_master_df["Store Type"].dropna().unique().tolist())

            new_code = st.text_input("Store Code *", key="new_store_code").strip().upper()
            new_name = st.text_input("Store Name *", key="new_store_name").strip()
            new_grade = _select_or_custom("Grade", existing_grades, "new_store_grade")
            new_potential = _select_or_custom("Store Potential", existing_potentials, "new_store_potential")
            new_zone = _select_or_custom("Zone", existing_zones, "new_store_zone")
            new_type = _select_or_custom("Store Type", existing_types, "new_store_type")

            if st.button("Add Store", width="stretch"):
                existing_codes_upper = store_master_df["store_code"].astype(str).str.strip().str.upper()
                if not new_code or not new_name:
                    st.error("Store Code and Store Name are required.")
                elif new_code in existing_codes_upper.values:
                    st.error(f"Store Code '{new_code}' already exists — pick a different code.")
                else:
                    new_row = pd.DataFrame([{
                        "store_code": new_code,
                        "store_name": new_name,
                        "grade": new_grade,
                        "store_potential": new_potential,
                        "Zone": new_zone,
                        "Store Type": new_type,
                    }])
                    updated_df = pd.concat([store_master_df, new_row], ignore_index=True)
                    updated_df.to_excel(STORE_DETAIL, index=False)
                    st.success(f"Added '{new_name}' ({new_code}).")
                    st.toast(f"✅ Store '{new_name}' added to the store master", icon="✅")
                    st.rerun()

    st.divider()
    if st.button("🔄 Clear cached results", width="stretch",
                  help="Forces a full reprocess of the last-fetched Stock API snapshot."):
        for key in ["file_key", "stock_grouped", "fresh_stock", "all_pieces"]:
            st.session_state.pop(key, None)
        st.rerun()

with stock_btn_slot.container():
    if st.button("📦", key="title_fetch_stock", help="Fetch Stock from API", disabled=not api_calls_enabled):
        try:
            with st.spinner("Fetching Stock API..."):
                n = fetch_stock_api()
            st.toast(f"✅ Stock API worked — {n} rows fetched", icon="✅")
            st.rerun()
        except Exception as e:
            st.toast(f"❌ Stock API failed: {e}", icon="❌")


reference_ready = os.path.exists("reference_base_stock.csv")
if not reference_ready:
    st.error("Base Stock reference not found. Use 'Recompute Base Stock' in the sidebar first.")
    st.stop()

# ---------------- Load the last-fetched Stock API snapshot, cache in session state ----------------
# There's no file upload — Stock only ever comes from the API. The most
# recently fetched snapshot is saved to STOCK_API_OUT on disk, so it's
# reused automatically (even across a server restart) until "Fetch Stock
# from API" in the sidebar is clicked again, which overwrites it.
if "stock_grouped" not in st.session_state:
    if not os.path.exists(STOCK_API_OUT):
        # Home still needs to render even with no Stock data yet — it's
        # where the Quick Fetch buttons (Sales/Sales Return/Memo Issue/Memo
        # Return, none of which need Stock) and the Stock API section live.
        # Every other window genuinely needs stock_grouped, so it still
        # hard-stops with the same message.
        if st.session_state["nav"] != "home":
            st.info("No Stock data yet — use 'Fetch Stock from API' in the sidebar to load the current snapshot.")
            st.stop()
        st.session_state["stock_grouped"] = None
        st.session_state["fresh_stock"] = None
        st.session_state["all_pieces"] = None
    else:
        with st.spinner("Loading last-fetched Stock snapshot..."):
            stock_api_df = pd.read_csv(STOCK_API_OUT)
            store_codes, store_lookup = load_store_lookup()
            st.session_state["stock_grouped"] = process_stock_file(stock_api_df, store_codes, store_lookup)
            st.session_state["fresh_stock"] = process_fresh_stock(stock_api_df)
            st.session_state["all_pieces"] = process_all_pieces(stock_api_df, store_codes, store_lookup)
            st.session_state["file_key"] = f"stock-snapshot-{os.path.getmtime(STOCK_API_OUT)}"

stock_grouped = st.session_state["stock_grouped"]
fresh_stock = st.session_state["fresh_stock"]
all_pieces = st.session_state["all_pieces"]
base_stock = pd.read_csv("reference_base_stock.csv")
sales_calendar = pd.read_csv("reference_sales_calendar.csv") if os.path.exists("reference_sales_calendar.csv") else None
memo_issue = pd.read_csv("reference_memo_issue.csv") if os.path.exists("reference_memo_issue.csv") else None
memo_return = pd.read_csv("reference_memo_return.csv") if os.path.exists("reference_memo_return.csv") else None
sales_daily = pd.read_csv("reference_sales_daily.csv", parse_dates=["Date"]) if os.path.exists("reference_sales_daily.csv") else None
memo_daily = pd.read_csv("reference_memo_daily.csv", parse_dates=["Date"]) if os.path.exists("reference_memo_daily.csv") else None
memo_return_daily = pd.read_csv("reference_memo_return_daily.csv", parse_dates=["Date"]) if os.path.exists("reference_memo_return_daily.csv") else None
sales_events = pd.read_csv("reference_sales_events.csv", parse_dates=["Date"]) if os.path.exists("reference_sales_events.csv") else None
memo_events = pd.read_csv("reference_memo_events.csv", parse_dates=["Date"]) if os.path.exists("reference_memo_events.csv") else None
memo_return_events = pd.read_csv("reference_memo_return_events.csv", parse_dates=["Date"]) if os.path.exists("reference_memo_return_events.csv") else None

ROW_KEYS = ["Store Name", "Store Code", "Style No", "Price Point", "Store Grade"]
ROW_KEYS_NO_STYLE = [k for k in ROW_KEYS if k != "Style No"]

MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
               7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
MONTH_ORDER = list(MONTH_NAMES.values())
SEASON_ORDER = ["Winter", "Spring", "Summer", "Monsoon", "Autumn"]

# Style Lookup's seasonal (current-month-only) matching needs at least this
# many qualifying years of data before it'll name a "Best Store" — a single
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
# stock_grouped is None on Home before any Stock data has been fetched —
# Home renders anyway (see the Stock API load gate above), so this has to
# tolerate that instead of assuming stock_grouped is always a DataFrame.
with st.sidebar:
    st.divider()
    if stock_grouped is not None:
        store_options = sorted(stock_grouped["Store Name"].dropna().unique().tolist())
        category_options = set(stock_grouped["Category"].dropna().unique()) | set(base_stock["Category"].dropna().unique())
        if sales_calendar is not None:
            category_options |= set(sales_calendar["Category"].dropna().unique())
        if memo_issue is not None:
            category_options |= set(memo_issue["Category"].dropna().unique())
        category_options = sorted(category_options)

        with st.expander("🔍 Filters", expanded=False):
            selected_stores = st.multiselect("Store Name", store_options, default=store_options)
            selected_categories = st.multiselect("Category", category_options, default=category_options)
    else:
        store_options, category_options = [], []
        selected_stores, selected_categories = [], []

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


def srp_pivot(sales_df, memo_df, row_keys, pivot_col, brand=None, min_months=1, memo_return_df=None):
    """SRP % = total Sales ÷ total Net Memo Issue (Memo Issue − Memo Return),
    one column per pivot_col value. Only months where that row+pivot_col had
    BOTH a sale and a memo issue count toward the totals (an inner join on
    Month+Year) — e.g. Memo 10 and Sales 5 in the same month contributes 10
    to Memo and 5 to Sales; a memo-only or sale-only month is dropped
    entirely before summing. If memo_return_df is given, it's matched on the
    same Month+Year and subtracted from Memo before dividing (a return with
    no matching Memo Issue that same month contributes nothing — there's
    nothing to net it against). Net Memo is floored at 0; a row where
    returns wipe out or exceed that month's issue (net <= 0) is left blank
    rather than a negative or infinite SRP.
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

    if memo_return_df is not None:
        return_f = memo_return_df[
            memo_return_df["Store Name"].isin(selected_stores) & memo_return_df["Category"].isin(selected_categories)
        ]
        if brand is not None:
            return_f = return_f[return_f["Brand"] == brand]
        matched = matched.merge(
            return_f[join_keys + ["JewelCodeCount"]].rename(columns={"JewelCodeCount": "Return"}),
            on=join_keys, how="left",
        )
        matched["Return"] = matched["Return"].fillna(0)
    else:
        matched["Return"] = 0

    totals = matched.groupby(row_keys + [pivot_col]).agg(
        Sales=("Sales", "sum"), Memo=("Memo", "sum"), Return=("Return", "sum"), MonthsMatched=("Sales", "size")
    ).reset_index()
    totals["NetMemo"] = (totals["Memo"] - totals["Return"]).clip(lower=0)
    totals["SRP"] = (totals["Sales"] / totals["NetMemo"] * 100).round(1)
    totals.loc[totals["NetMemo"] <= 0, "SRP"] = np.nan
    totals.loc[totals["MonthsMatched"] < min_months, "SRP"] = np.nan

    pivot = totals.pivot_table(index=row_keys, columns=pivot_col, values="SRP", aggfunc="first").reset_index()
    pivot.columns.name = None
    return pivot


def srp_lifetime_pivot(sales_df, memo_df, row_keys, pivot_col, brand=None):
    """SRP % = Total Sales ÷ Total Memo Issue, summed over ALL available
    history — one fixed, period-free number per Store + Style + Grade +
    Price Point + Category + Brand.

    Deliberately gross: Memo Return is NOT subtracted here. A Style No
    covers many different physical pieces (Jewel Codes) circulating through
    a store over time — issued, some sell, some come back unsold, and
    replacements get issued and sell in turn. A returned piece is almost
    always a *different* physical unit than one that later sold, so netting
    Return against Issue shrinks the denominator toward the pieces that
    didn't convert, even while other pieces of the same style keep selling
    fine — for a style with heavy issue/return churn this can inflate SRP %
    past 1000% (seen in practice: 17 issued, 16 returned, 13 sold → net
    Memo of 1 → 1300%), which misrepresents genuinely healthy sell-through
    as a broken number. See return_rate_pivot for the Return signal instead
    — shown alongside SRP %, not blended into it.
    A row with Memo Issue == 0 is left blank (nothing to divide by); 0
    Sales with Memo Issue > 0 correctly shows 0%, not blank."""
    sales_f = sales_df[sales_df["Store Name"].isin(selected_stores) & sales_df["Category"].isin(selected_categories)]
    memo_f = memo_df[memo_df["Store Name"].isin(selected_stores) & memo_df["Category"].isin(selected_categories)]
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


def return_rate_pivot(memo_df, memo_return_df, row_keys, pivot_col, brand=None):
    """Return % = Total Memo Return ÷ Total Memo Issue, summed over ALL
    available history — the companion signal to SRP % (see
    srp_lifetime_pivot's docstring for why Return isn't netted into SRP %
    itself). High Return % on a style/store means pieces keep bouncing back
    unsold, tracked independently of how well other pieces of the same
    style are selling elsewhere.
    A row with Memo Issue == 0 is left blank. 0 Return with Memo Issue > 0
    correctly shows 0%, not blank."""
    memo_f = memo_df[memo_df["Store Name"].isin(selected_stores) & memo_df["Category"].isin(selected_categories)]
    if brand is not None:
        memo_f = memo_f[memo_f["Brand"] == brand]
    memo_totals = memo_f.groupby(row_keys + [pivot_col])["JewelCodeCount"].sum().reset_index(name="Memo")

    if memo_return_df is not None:
        return_f = memo_return_df[
            memo_return_df["Store Name"].isin(selected_stores) & memo_return_df["Category"].isin(selected_categories)
        ]
        if brand is not None:
            return_f = return_f[return_f["Brand"] == brand]
        return_totals = return_f.groupby(row_keys + [pivot_col])["JewelCodeCount"].sum().reset_index(name="Return")
    else:
        return_totals = pd.DataFrame(columns=row_keys + [pivot_col, "Return"])

    totals = memo_totals.merge(return_totals, on=row_keys + [pivot_col], how="left")
    totals["Return"] = totals["Return"].fillna(0)
    totals["ReturnRate"] = np.where(totals["Memo"] > 0, (totals["Return"] / totals["Memo"] * 100).round(1), np.nan)

    pivot = totals.pivot_table(index=row_keys, columns=pivot_col, values="ReturnRate", aggfunc="first").reset_index()
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
    from the Stock API snapshot) summed across every Category for each
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


def month_zone_category_pivot(df, brand):
    """Like month_category_pivot, but with Zone as an extra row dimension
    alongside Month (row = Month + Zone, Jan->Dec order, Zone alphabetical
    within each month), one column per Category — lets you see whether a
    seasonal spike in a Category shows up the same way in every Zone, or is
    concentrated in just one. Every Month+Zone combination is shown even
    with 0 sales (reindexed), same as month_category_pivot's full Jan->Dec
    coverage."""
    filtered = df[
        (df["Brand"] == brand)
        & df["Store Name"].isin(selected_stores)
        & df["Category"].isin(selected_categories)
    ]
    zones = sorted(df["Zone"].dropna().unique().tolist())
    full_index = pd.MultiIndex.from_product([range(1, 13), zones], names=["Month", "Zone"])
    pivot = filtered.pivot_table(
        index=["Month", "Zone"], columns="Category", values="JewelCodeCount", aggfunc="sum", fill_value=0
    )
    pivot = pivot.reindex(full_index, fill_value=0)
    pivot = pivot.reset_index()
    pivot["Month"] = pivot["Month"].map(MONTH_NAMES)
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
            f"<span style='color:#2f5dff; font-size:0.82rem; font-weight:600;'>{label or 'Working'}… {pct}%</span>",
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


def _fmt_func(fmt):
    return lambda x: (fmt % x) if pd.notna(x) else ""


def show_pivot(pivot, key_prefix, fmt="%.2f", row_keys=None, styler=None):
    row_keys = row_keys or ROW_KEYS
    cat_cols = [c for c in pivot.columns if c not in row_keys]
    if styler is not None:
        render_html_table(styler)
    else:
        render_html_table(pivot.style.format(_fmt_func(fmt), subset=cat_cols, na_rep=""))
    csv = pivot.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{key_prefix}.csv", "text/csv", key=f"dl_{key_prefix}")


def show_store_pivot(pivot, row_keys, key_prefix, fmt="%.1f%%", best_label="Best Store", styler=None):
    """Like show_pivot, but skips number-formatting the text 'Best Store'
    column while still formatting its paired 'Best Store Value' column."""
    text_col = best_label
    numeric_cols = [c for c in pivot.columns if c not in row_keys + [text_col]]
    if styler is not None:
        render_html_table(styler)
    else:
        render_html_table(pivot.style.format(_fmt_func(fmt), subset=numeric_cols, na_rep=""))
    csv = pivot.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"{key_prefix}.csv", "text/csv", key=f"dl_{key_prefix}")


MERGE_KEYS = ROW_KEYS + ["Brand", "Category"]
if stock_grouped is None:
    # Home with no Stock data yet — nothing downstream actually reads this
    # (every other window hard-stops before reaching here), just needs to
    # exist without crashing.
    diff_long = pd.DataFrame(columns=MERGE_KEYS + ["Stock", "BaseStock", "Difference"])
else:
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
nav = st.session_state["nav"]

if nav == "home":
    st.markdown('<div class="home-eyebrow">12 windows · one dashboard</div>', unsafe_allow_html=True)

    if stock_grouped is None:
        st.info("No Stock data yet — use 'Fetch Stock from API' in the sidebar, or the Quick Fetch buttons below for the other APIs, to get started.")
    else:
        stat_cols = st.columns(4)
        with stat_cols[0]:
            render_stat_card("Stores", f"{stock_grouped['Store Code'].nunique():,}", "🏬", "#2f5dff")
        with stat_cols[1]:
            render_stat_card("Styles Tracked", f"{stock_grouped['Style No'].nunique():,}", "🎨", "#16a34a")
        with stat_cols[2]:
            render_stat_card("Total Stock Pieces", f"{int(stock_grouped['Stock'].sum()):,}", "📦", "#d97706")
        with stat_cols[3]:
            if os.path.exists(META_OUT):
                last_refreshed = datetime.fromisoformat(meta["last_refreshed"]).strftime("%b %d, %H:%M")
            else:
                last_refreshed = "—"
            render_stat_card("Refreshed", last_refreshed, "🕒", "#dc2626")

    st.write("")
    with st.container(key="home_group_expanders_other_api"):
        with st.expander("Other API", expanded=False):
            if not api_calls_enabled:
                st.caption("🔒 Turn on '🔓 Allow live API calls' in the sidebar to enable these buttons.")
            quick_fetch_specs = [
                ("Sales", SALES_API_CONFIG, SALES_API_OUT),
                ("Sales Return", SALES_RETURN_API_CONFIG, SALES_RETURN_API_OUT),
                ("Memo Issue", MEMO_API_CONFIG, MEMO_API_OUT),
                ("Memo Return", MEMO_RETURN_API_CONFIG, MEMO_RETURN_API_OUT),
            ]
            quick_fetch_cols = st.columns(4)
            for col, (label, config_path, out_path) in zip(quick_fetch_cols, quick_fetch_specs):
                with col:
                    if st.button(f"⬇️ Fetch {label}", key=f"quick_fetch_{label}", width="stretch", disabled=not api_calls_enabled):
                        try:
                            with st.spinner(f"Fetching {label} API..."):
                                n = fetch_transaction_api(label, config_path, out_path)
                            st.toast(f"✅ {label} API worked — {n} rows fetched", icon="✅")
                        except Exception as e:
                            st.toast(f"❌ {label} API failed: {e}", icon="❌")

    st.write("")
    tiles_by_key = {t[0]: t for t in WINDOW_TILES}

    def render_tile(col, tile, accent):
        key, icon, num, title, desc = tile
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

    # Featured pair, always visible above the collapsible groups.
    st.markdown('<div class="tile-group-label">Featured</div>', unsafe_allow_html=True)
    featured_cols = st.columns(2)
    for col, key, accent in zip(featured_cols, ["jewel_allocate", "srp"], TILE_ACCENTS):
        render_tile(col, tiles_by_key[key], accent)

    st.markdown('<div class="choose-window-note"><b>Choose a window</b> — click a tile to open that window\'s data. You can jump between windows any time with the nav above.</div>', unsafe_allow_html=True)

    # Everything else, grouped and collapsible.
    TILE_GROUPS = [
        ("Inventory", ["stock", "basestock", "diff"]),
        ("Sales & Performance", ["sales_cal", "store_compare", "seasonal", "zone"]),
        ("Reports & Lookup", ["flag", "no_style", "style_lookup"]),
    ]
    cols_per_row = 3
    with st.container(key="home_group_expanders_tiles"):
        for group_label, keys in TILE_GROUPS:
            with st.expander(group_label, expanded=False):
                group_tiles = [tiles_by_key[k] for k in keys]
                for row_start in range(0, len(group_tiles), cols_per_row):
                    row_tiles = group_tiles[row_start:row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for i, (col, tile) in enumerate(zip(cols, row_tiles)):
                        accent = TILE_ACCENTS[(row_start + i) % len(TILE_ACCENTS)]
                        render_tile(col, tile, accent)
else:
    icon, num, title = WINDOW_LOOKUP[nav]
    st.button("🏠 Home", key="nav_home_btn", on_click=_go_to, args=("home",))

# ---------------- Window 1: Stock pivot ----------------
if nav == "stock":
    render_page_header(num, "Stock — Item Pcs by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = total ItemPcs from the Stock API snapshot."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "1. Start from every row in the current Stock API snapshot.\n"
            "2. Drop rows with a blank Style No or Category — this removes the report's trailing "
            "'Total' row and any incomplete row, so it can't silently inflate the piece count.\n"
            "3. Keep only rows whose **Client Code** matches a real store in `store detail.xlsx` "
            "— rows with a blank Client Code are unallocated 'fresh stock' and are excluded here "
            "(they show up instead in Window 13's Style Lookup and Window 14's Stock Assortment "
            "Summary).\n"
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

    st.divider()
    st.markdown("## Current Stock — Store + Price Point (styles combined), by Category")
    st.write(
        "Same Stock numbers as above, rolled up to **Store + Price Point** instead of individual "
        "Style No — a quicker, style-agnostic read on how much quantity sits at each store in each "
        "price band, by Category."
    )
    st.markdown("### Sparkles (Gold)")
    show_pivot(
        blank_zeros(brand_pivot(stock_grouped, "Stock", "Sparkles", row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "stock_pricepoint_sparkles", fmt="%.0f", row_keys=ROW_KEYS_NO_STYLE,
    )

    st.markdown("### SparQ (Silver)")
    show_pivot(
        blank_zeros(brand_pivot(stock_grouped, "Stock", "Sparq", row_keys=ROW_KEYS_NO_STYLE), row_keys=ROW_KEYS_NO_STYLE),
        "stock_pricepoint_sparq", fmt="%.0f", row_keys=ROW_KEYS_NO_STYLE,
    )

    st.info("📝 **Note:** check the pivot view of stock by store.")

# ---------------- Window 2: Base Stock pivot ----------------
if nav == "basestock":
    render_page_header(num, "Base Stock — by Category")
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
    render_page_header(num, "Difference — Stock minus Base Stock, by Category")
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

    st.info("📝 **Note:** Difference = Stock − Base Stock — shows whether a store is carrying **extra** (positive) or **less** (negative) stock than its typical sell-through.")

# ---------------- Window 4: Sales ----------------
if nav == "sales_cal":
    render_page_header(num, "Sales — distinct Jewel Code count by Month")
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

        st.info("📝 **Note:** check old sales trends here.")

# ---------------- Window 5: Issued DC ----------------
if nav == "flag":
    render_page_header(num, "Issued DC — distinct Jewel Code count by Month")
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

        st.info("📝 **Note:** Stock Outward to store — sent from HO to store.")

# ---------------- Window 6: Base Stock + Difference, no Style No ----------------
if nav == "no_style":
    render_page_header(num, "Store Summary — Base Stock and Difference, Style No removed")
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

# ---------------- Window 7: SRP % (Sales ÷ Net Memo, lifetime) ----------------
if nav == "srp":
    render_page_header(num, "Store by SRP % — Sales ÷ Net Memo, by Category")
    st.write(
        "Row = Store Name + Store Code + Style No + Price Point + Store Grade, one column per Category. "
        "Value = SRP % = Total Sales ÷ Total Net Memo (Memo Issue − Memo Return), summed over **all "
        "history** — one fixed number, no period to pick. "
        "**Current Stock (Pcs)** = pieces of that Style physically sitting in that store right now."
    )
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "SRP % measures **sell-through**: of what was issued to a store on memo (net of what came "
            "back), how much actually sold — using the entire history on file, not a trailing window:\n\n"
            "1. Start from `reference_sales_daily.csv`, `reference_memo_daily.csv`, and "
            "`reference_memo_return_daily.csv`.\n"
            "2. Sum **Sales**, **Memo Issue**, and **Memo Return** separately for each Store + Style + "
            "Grade + Price Point + Category, across every date on file — no trailing window, no "
            "same-month requirement between a Sale and a Memo Issue.\n"
            "3. **Net Memo = Memo Issue − Memo Return**, floored at 0. Using the full lifetime for "
            "both sides (rather than a Day/Week/Month/Year window) sidesteps the trade-off a window "
            "forces: a short window can catch a Return with no matching Issue inside it (an issue and "
            "its return are often months apart), and a fixed running balance divided into a longer "
            "Sales window inflates without bound. Over the full lifetime, every Issue/Return pair is "
            "guaranteed to fall inside the same (unbounded) window, so neither problem can occur.\n"
            "4. **SRP % = (Sales ÷ Net Memo) × 100**, rounded to 1 decimal. A row with **Net Memo ≤ 0** "
            "is left blank (nothing to divide by); **0 Sales** with Net Memo > 0 correctly shows "
            "**0%**, not blank.\n"
            "5. **Current Stock (Pcs)** is Window 1's Stock (the Stock API snapshot's ItemPcs), "
            "summed across every Category for that Store + Style + Grade + Price Point — a right-now "
            "snapshot, not a historical total.\n"
            f"6. **Color** — 🟩 green: SRP % ≥ {SRP_HIGH}% *and* stock in hand (selling well, still "
            f"stocked). 🟥 red: SRP % < {SRP_LOW}% (weak sell-through — includes dead stock sitting "
            "unsold, and styles with nothing moving either way). 🟨 yellow: everything in between, "
            f"including a store selling well (SRP % ≥ {SRP_HIGH}%) but currently out of that stock — "
            "a different problem (missed sales) than dead stock, so it isn't colored red."
        )

    if sales_daily is None or memo_daily is None:
        st.error("reference_sales_daily.csv or reference_memo_daily.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        st.caption("Lifetime SRP % — Sales ÷ Memo Issue, both since the piece was first issued. Fixed, no period to pick.")
        st.caption(
            f"🟩 SRP % ≥ {SRP_HIGH}% with stock in hand &nbsp;·&nbsp; "
            f"🟨 mixed signal (moderate SRP %, or strong SRP % but stocked out) &nbsp;·&nbsp; "
            f"🟥 SRP % < {SRP_LOW}% — weak sell-through"
        )

        srp7_tick = progress_tracker(6)

        st.markdown("### Sparkles (Gold)")
        srp7_tick("Computing Sparkles SRP %")
        srp7_sparkles = srp_lifetime_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", brand="Sparkles")
        srp7_sparkles = with_total_stock(srp7_sparkles, ROW_KEYS, "Sparkles")
        srp7_tick("Rendering Sparkles table")
        show_pivot(
            srp7_sparkles, "srp_sparkles", fmt="%.1f%%", row_keys=ROW_KEYS + ["Current Stock (Pcs)"],
            styler=style_srp_with_row_stock(srp7_sparkles, ROW_KEYS),
        )
        srp7_tick("Computing Sparkles Return %")
        st.caption("Return % — Memo Return ÷ Memo Issue, lifetime. High = pieces of this style keep bouncing back unsold at this store.")
        show_pivot(
            blank_zeros(return_rate_pivot(memo_daily, memo_return_daily, ROW_KEYS, "Category", brand="Sparkles"), row_keys=ROW_KEYS),
            "return_rate_sparkles", fmt="%.1f%%", row_keys=ROW_KEYS,
        )

        st.markdown("### SparQ (Silver)")
        srp7_tick("Computing SparQ SRP %")
        srp7_sparq = srp_lifetime_pivot(sales_daily, memo_daily, ROW_KEYS, "Category", brand="Sparq")
        srp7_sparq = with_total_stock(srp7_sparq, ROW_KEYS, "Sparq")
        srp7_tick("Rendering SparQ table")
        show_pivot(
            srp7_sparq, "srp_sparq", fmt="%.1f%%", row_keys=ROW_KEYS + ["Current Stock (Pcs)"],
            styler=style_srp_with_row_stock(srp7_sparq, ROW_KEYS),
        )
        srp7_tick("Computing SparQ Return %")
        st.caption("Return % — Memo Return ÷ Memo Issue, lifetime. High = pieces of this style keep bouncing back unsold at this store.")
        show_pivot(
            blank_zeros(return_rate_pivot(memo_daily, memo_return_daily, ROW_KEYS, "Category", brand="Sparq"), row_keys=ROW_KEYS),
            "return_rate_sparq", fmt="%.1f%%", row_keys=ROW_KEYS,
        )

# ---------------- Window 8: Store Comparison (SRP % and Base Stock, Store Name as columns) ----------------
if nav == "store_compare":
    render_page_header(num, "Store Comparison — SRP %, Base Stock, and Current Stock, side by side across stores")
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
            "- **SRP % table** — Window 7's lifetime SRP % (Total Sales ÷ Total Net Memo, summed over "
            "all history, no period to pick), pivoted with Store Name as columns instead of Category.\n"
            "- **Base Stock table** — Window 2's raw Base Stock rate (Total Sales Qty ÷ Months "
            "Elapsed, *before* the blank/0/1 simplification), pivoted the same way.\n"
            "- **Current Stock table** — Window 1's raw Stock (ItemPcs from the Stock API snapshot, "
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
    st8_tick = progress_tracker(8 if srp8_ready else 4)

    st.markdown("## SRP % — best store by sell-through (Window 7 / lifetime logic)")
    if not srp8_ready:
        st.error("reference_sales_daily.csv or reference_memo_daily.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        st.caption("Lifetime SRP % — Sales ÷ Memo Issue, both since the piece was first issued. Fixed, no period to pick.")
        st.caption(
            f"🟩 SRP % ≥ {SRP_HIGH}% with stock in hand &nbsp;·&nbsp; "
            f"🟨 mixed signal (moderate SRP %, or strong SRP % but stocked out) &nbsp;·&nbsp; "
            f"🟥 SRP % < {SRP_LOW}% — weak sell-through"
        )

        st.markdown("### Sparkles (Gold)")
        st8_tick("Computing Sparkles SRP %")
        srp_store_sparkles = srp_lifetime_pivot(sales_daily, memo_daily, STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparkles")
        srp_store_sparkles = add_best_column(srp_store_sparkles, STORE_COMPARE_ROW_KEYS)
        srp8_cols_sparkles = [c for c in srp_store_sparkles.columns if c not in STORE_COMPARE_ROW_KEYS + ["Best Store", "Best Store Value"]]
        show_store_pivot(
            srp_store_sparkles, STORE_COMPARE_ROW_KEYS, "store_compare_srp_sparkles", fmt="%.1f%%",
            styler=style_srp_with_store_stock(srp_store_sparkles, stock_raw_sparkles, STORE_COMPARE_ROW_KEYS, srp8_cols_sparkles),
        )
        st8_tick("Computing Sparkles Return %")
        st.caption("Return % — Memo Return ÷ Memo Issue, lifetime, by store.")
        show_pivot(
            blank_zeros(return_rate_pivot(memo_daily, memo_return_daily, STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparkles"), row_keys=STORE_COMPARE_ROW_KEYS),
            "return_rate_store_sparkles", fmt="%.1f%%", row_keys=STORE_COMPARE_ROW_KEYS,
        )

        st.markdown("### SparQ (Silver)")
        st8_tick("Computing SparQ SRP %")
        srp_store_sparq = srp_lifetime_pivot(sales_daily, memo_daily, STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparq")
        srp_store_sparq = add_best_column(srp_store_sparq, STORE_COMPARE_ROW_KEYS)
        srp8_cols_sparq = [c for c in srp_store_sparq.columns if c not in STORE_COMPARE_ROW_KEYS + ["Best Store", "Best Store Value"]]
        show_store_pivot(
            srp_store_sparq, STORE_COMPARE_ROW_KEYS, "store_compare_srp_sparq", fmt="%.1f%%",
            styler=style_srp_with_store_stock(srp_store_sparq, stock_raw_sparq, STORE_COMPARE_ROW_KEYS, srp8_cols_sparq),
        )
        st8_tick("Computing SparQ Return %")
        st.caption("Return % — Memo Return ÷ Memo Issue, lifetime, by store.")
        show_pivot(
            blank_zeros(return_rate_pivot(memo_daily, memo_return_daily, STORE_COMPARE_ROW_KEYS, "Store Name", brand="Sparq"), row_keys=STORE_COMPARE_ROW_KEYS),
            "return_rate_store_sparq", fmt="%.1f%%", row_keys=STORE_COMPARE_ROW_KEYS,
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
    st.write("Value = current Stock (ItemPcs from the Stock API snapshot). 'Most Stocked Store' = the store holding the most pieces right now.")
    st.markdown("### Sparkles (Gold)")
    st8_tick("Computing Sparkles Current Stock")
    stock_store_sparkles = add_best_column(stock_raw_sparkles.copy(), STORE_COMPARE_ROW_KEYS, label="Most Stocked Store")
    show_store_pivot(stock_store_sparkles, STORE_COMPARE_ROW_KEYS, "store_compare_stock_sparkles", fmt="%.0f", best_label="Most Stocked Store")

    st.markdown("### SparQ (Silver)")
    st8_tick("Computing SparQ Current Stock")
    stock_store_sparq = add_best_column(stock_raw_sparq.copy(), STORE_COMPARE_ROW_KEYS, label="Most Stocked Store")
    show_store_pivot(stock_store_sparq, STORE_COMPARE_ROW_KEYS, "store_compare_stock_sparq", fmt="%.0f", best_label="Most Stocked Store")

# ---------------- Window 10: Seasonal Trends (India Festival Calendar) ----------------
if nav == "seasonal":
    render_page_header(num, "Seasonal Trends — Month-wise Best Sellers, mapped to India's Festival Calendar")
    st.write(
        "Month-wise: row = calendar Month (Jan → Dec, every year on file combined), one column per "
        "Category, plus that month's Season and Festivals from India_Culture_Events_2019-2026.csv. "
        "Month-wise by Zone: the same, with Zone added as an extra row alongside Month. "
        "Season-wise: the same sales, rolled up to the 5 seasons. "
        "**Best Category** = the highest-selling category, in that Month, Month+Zone, or Season."
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
            "Window 13's Fresh Stock pieces section (current-month-only)."
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

            st.markdown("**Month-wise by Zone**")
            st.caption("Same Month-wise numbers, split out by Zone as well — see whether a seasonal spike shows up the same way in every Zone or is concentrated in just one.")
            if "Zone" not in sales_calendar.columns:
                st.error("This reference data was built before Zone tracking was added. Click 'Recompute Base Stock' in the sidebar to rebuild it with Zone included.")
            else:
                month_zone_pivot = month_zone_category_pivot(sales_calendar, brand)
                month_zone_pivot = add_best_col_no_sort(month_zone_pivot, row_keys=["Month", "Zone"], label="Best Category")
                month_zone_pivot = month_zone_pivot.merge(
                    culture_month_lookup[["MonthName", "Season", "Festivals"]],
                    left_on="Month", right_on="MonthName", how="left",
                ).drop(columns="MonthName")
                mz_front_cols = ["Month", "Zone", "Season", "Festivals"]
                mz_other_cols = [c for c in month_zone_pivot.columns if c not in mz_front_cols]
                month_zone_pivot = month_zone_pivot[mz_front_cols + mz_other_cols]
                show_pivot(
                    blank_zeros(month_zone_pivot, row_keys=mz_front_cols + ["Best Category"]),
                    f"seasonal_month_zone_{brand.lower()}", fmt="%.0f", row_keys=mz_front_cols + ["Best Category"],
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
    render_page_header(num, "Area (Zone) Sales — which Zone sells best, by Style and by Month")
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

# ---------------- Window 13: Style Lookup — why this store? ----------------
if nav == "style_lookup":
    render_page_header(num, "Style Lookup — why should this style go to this store?")
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
            "6. **SRP %, by store** — Windows 7/8's lifetime SRP % (Total Sales ÷ Total Net Memo, no "
            "period to pick), filtered to this style.\n"
            "7. **Fresh Stock pieces** — this style's unallocated pieces and their SRP / Base Stock "
            f"Best Store recommendations, scoped to {lookup_month_name}-only history (every year on "
            "file).\n"
            f"8. **Seasonal performance, by store** — the full per-store ranking behind that "
            f"recommendation, for {lookup_month_name} across every year on file (same "
            f"{MIN_SEASONAL_SAMPLES}-year minimum-sample guard), for *every* store, not just the "
            "winner — so you can see who came 2nd and 3rd, and by how much.\n"
            "9. **Month-wise Trend** — Window 10's Jan→Dec pattern, scoped to just this style, across "
            "every store and year, so you can see if it's a seasonal style.\n"
            "10. **Zone Breakdown** — Window 11's Style-wise Zone comparison, filtered to this style.\n\n"
            f"Color coding on SRP % tables matches Windows 7/8: 🟩 SRP % ≥ {SRP_HIGH}% with stock "
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
            style13_tick = progress_tracker(11)

            # ---- Summary — Sparkles vs SparQ, at a glance ----
            style13_tick("Building Summary")
            st.markdown("### 🔎 Summary")
            summary_cols = st.columns(2, gap="large")
            for summary_col, b in zip(summary_cols, ["Sparkles", "Sparq"]):
                with summary_col:
                    stock_b = style_stock_rows[style_stock_rows["Brand"] == b]
                    bs_b = _match_style(base_stock)
                    bs_b = bs_b[bs_b["Brand"] == b]
                    sales_cal_b = _match_style(sales_calendar) if sales_calendar is not None else pd.DataFrame()
                    sales_cal_b = sales_cal_b[sales_cal_b["Brand"] == b] if not sales_cal_b.empty else sales_cal_b
                    memo_b = _match_style(memo_issue) if memo_issue is not None else pd.DataFrame()
                    memo_b = memo_b[memo_b["Brand"] == b] if not memo_b.empty else memo_b
                    memo_return_b = _match_style(memo_return) if memo_return is not None else pd.DataFrame()
                    memo_return_b = memo_return_b[memo_return_b["Brand"] == b] if not memo_return_b.empty else memo_return_b
                    fresh_b = style_fresh_rows[style_fresh_rows["Brand"] == b]

                    if stock_b.empty and bs_b.empty and sales_cal_b.empty and memo_b.empty and fresh_b.empty:
                        st.markdown(
                            f"""<div style="border:1px solid #e2e8f0; border-radius:10px;
                                padding:12px 14px; background:#f8fafc;">
                                <div style="font-weight:700; font-size:0.85rem; margin-bottom:4px; color:#1e2a3b;">{b}</div>
                                <div style="font-size:0.78rem; color:#64748b;">No data for this brand.</div>
                                </div>""",
                            unsafe_allow_html=True,
                        )
                        continue

                    total_stock = stock_b["Stock"].sum() if not stock_b.empty else 0
                    stores_with_stock = stock_b.loc[stock_b["Stock"] > 0, "Store Name"].nunique() if not stock_b.empty else 0
                    total_base_stock = bs_b["BaseStock"].sum() if not bs_b.empty else 0.0
                    total_sales = sales_cal_b["JewelCodeCount"].sum() if not sales_cal_b.empty else 0
                    total_memo = memo_b["JewelCodeCount"].sum() if not memo_b.empty else 0
                    total_memo_return = memo_return_b["JewelCodeCount"].sum() if not memo_return_b.empty else 0
                    srp_pct = (total_sales / total_memo * 100) if total_memo > 0 else None
                    return_pct = (total_memo_return / total_memo * 100) if total_memo > 0 else None
                    fresh_pieces = fresh_b["ItemPcs"].sum() if not fresh_b.empty else 0

                    best_store, best_store_stock = None, None
                    if not stock_b.empty:
                        by_store = stock_b.groupby("Store Name")["Stock"].sum()
                        if by_store.max() > 0:
                            best_store, best_store_stock = by_store.idxmax(), by_store.max()

                    best_month = None
                    if not sales_cal_b.empty:
                        by_month = sales_cal_b.groupby("Month")["JewelCodeCount"].sum()
                        if by_month.max() > 0:
                            best_month = MONTH_NAMES.get(int(by_month.idxmax()))

                    srp_display = f"{srp_pct:.1f}%" if srp_pct is not None else "—"
                    return_display = f"{return_pct:.1f}%" if return_pct is not None else "—"
                    rows_html = "".join(
                        f"""<div style="display:flex; justify-content:space-between; padding:3px 0;
                            border-bottom:1px solid #e2e8f0;">
                            <span style="color:#64748b;">{label}</span>
                            <span style="font-weight:600; font-variant-numeric:tabular-nums; color:#1e2a3b;">{value}</span>
                            </div>"""
                        for label, value in [
                            ("Current Stock", f"{total_stock:.0f}"),
                            ("Base Stock (network)", f"{total_base_stock:.2f}"),
                            ("Lifetime Sales", f"{total_sales:.0f}"),
                            ("Memo Issue", f"{total_memo:.0f}"),
                            ("Memo Return", f"{total_memo_return:.0f}"),
                            ("SRP % (Sales ÷ Memo Issue)", srp_display),
                            ("Return % (Return ÷ Memo Issue)", return_display),
                            ("Fresh Stock (unallocated)", f"{fresh_pieces:.0f}"),
                        ]
                    )
                    footer_bits = []
                    if best_store:
                        footer_bits.append(f"📍 Best store: <b>{best_store}</b> ({best_store_stock:.0f} pcs)")
                    if best_month:
                        footer_bits.append(f"🗓️ Best month: <b>{best_month}</b>")
                    footer_html = (
                        f"""<div style="margin-top:8px; font-size:0.74rem; color:#64748b; line-height:1.5;">
                            {"<br>".join(footer_bits)}</div>"""
                        if footer_bits else ""
                    )
                    st.markdown(
                        f"""<div style="border:1px solid #e2e8f0; border-radius:12px;
                            padding:14px 16px; background:#ffffff; box-shadow:0 1px 2px rgba(16,24,38,.06);">
                            <div style="font-weight:700; font-size:0.92rem; margin-bottom:8px;
                                color:#1e2a3b;">{b} <span style="color:#94a3b8; font-weight:500;
                                font-size:0.72rem;">· stock across {stores_with_stock} store(s)</span></div>
                            <div style="font-size:0.82rem;">{rows_html}</div>
                            {footer_html}
                            </div>""",
                        unsafe_allow_html=True,
                    )

            st.divider()

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
                render_html_table(style_stock_display.style.format({"Stock": _fmt_func("%.0f")}, na_rep=""))

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
            # ---- SRP %, by store (Windows 7/8 logic, lifetime) ----
            st.markdown("### 📈 6 — SRP %, by store")
            if sales_daily is None or memo_daily is None:
                st.error("reference_sales_daily.csv or reference_memo_daily.csv not found.")
            else:
                st.caption("Lifetime SRP % — Sales ÷ Memo Issue, both since the piece was first issued. Fixed, no period to pick.")
                any_period = False
                for b in ["Sparkles", "Sparq"]:
                    srp_p = srp_lifetime_pivot(sales_daily, memo_daily, match_keys, "Store Name", brand=b)
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
                    return_p = _match_style(return_rate_pivot(memo_daily, memo_return_daily, match_keys, "Store Name", brand=b))
                    if not return_p.empty:
                        st.caption(f"**{b}** Return % — Memo Return ÷ Memo Issue, lifetime, by store.")
                        show_pivot(blank_zeros(return_p, row_keys=match_keys), f"lookup_return_rate_{b.lower()}_{style_query}", fmt="%.1f%%", row_keys=match_keys)
                if not any_period:
                    st.info("No SRP % data for this style.")

            st.divider()
            # ---- Fresh Stock pieces + Best Store recommendation ----
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
                memo_return_seasonal = memo_return[memo_return["Month"] == lookup_month] if memo_return is not None else None
                srp_lookup_parts, bs_lookup_parts = [], []
                for b in ["Sparkles", "Sparq"]:
                    srp_b = srp_pivot(
                        sales_seasonal, memo_seasonal, match_keys, "Store Name", brand=b,
                        min_months=MIN_SEASONAL_SAMPLES, memo_return_df=memo_return_seasonal,
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
                render_html_table(
                    style_suggestions[display_cols].style.format({
                        "SRP Best Store Value": _fmt_func("%.1f%%"),
                        "Base Stock Best Store Value": _fmt_func("%.2f"),
                    }, na_rep="")
                )

            st.divider()
            # ---- Seasonal SRP % and Base Stock by store, full breakdown ----
            st.markdown(f"### 🗓️ 8 — Seasonal performance — {lookup_month_name} history, every store")
            st.write("Full store-by-store ranking behind the recommendation above — not just the winner.")
            if sales_calendar is None or memo_issue is None:
                st.error("reference_sales_calendar.csv or reference_memo_issue.csv not found.")
            else:
                sales_seasonal = sales_calendar[sales_calendar["Month"] == lookup_month]
                memo_seasonal = memo_issue[memo_issue["Month"] == lookup_month]
                memo_return_seasonal = memo_return[memo_return["Month"] == lookup_month] if memo_return is not None else None
                any_seasonal = False
                for b in ["Sparkles", "Sparq"]:
                    srp_b = srp_pivot(
                        sales_seasonal, memo_seasonal, match_keys, "Store Name", brand=b,
                        min_months=MIN_SEASONAL_SAMPLES, memo_return_df=memo_return_seasonal,
                    )
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

# ---------------- Window 14: Stock Assortment Summary ----------------
if nav == "jewel_allocate":
    render_page_header(num, "Stock Assortment Summary — best SRP store, plus seasonal sales history")
    with st.expander("📖 How this is calculated"):
        st.markdown(
            "1. Match each Jewel Code you provide against **every piece in the current Stock API "
            "snapshot** — fresh (blank Client Code, still at HO) and already-allocated (sitting at a "
            "store) alike; the result table's **Current Store** column shows blank for fresh pieces, "
            "or the store it's currently at. A code not found at all — sold, discontinued, or mistyped "
            "— or made of a Base Metal not tracked as Sparkles/Sparq (e.g. Alloy) is listed separately "
            "and gets no recommendation.\n"
            "2. For each matched piece's Style No + Price Point + Category + Brand, compute lifetime "
            "SRP % (Window 7's logic): Sales ÷ Memo Issue, summed over **all history** — one fixed "
            "number, no period to pick — split out **by Store Name**.\n"
            "3. **Store Name** in the result = the store with the highest SRP % for that combination — "
            "picked purely by SRP %, with no rule against it matching **Current Store**; when it does, "
            "that just confirms the piece is already well-placed and no transfer is needed. Ranking is "
            "on SRP % alone — Return % is not netted in (see Window 7's docstring on why), just shown "
            "alongside for context.\n"
            "4. **Return %**, **Avg Days to Sell**, **Avg Stock Return Count**, **Stock Available**, and **Base "
            "Stock** are then looked up for that specific winning store — Memo Return ÷ Memo Issue, a "
            "per-piece Issue → Return → Sale timeline trace (same backward-matching technique used to "
            "attribute Memo Return to a store), Window 1's Stock, and Window 2's Base Stock rate — for "
            "context, not part of the ranking itself.\n"
            "5. **Seasonal Sales History** (below the main table, split into separate Sparkles/SparQ "
            "tables) is simpler — just Sales Calendar units sold plus that store's **Current Stock** "
            "(Window 1's ItemPcs, same Style + Price Point + Category + Store), filtered to the "
            "same-calendar-month-across-years you pick, for just these matched styles, broken out by "
            "every Store + Zone that's ever sold one, not just the single best store. No SRP % here."
        )

    input_mode = st.radio("Input method", ["Paste Jewel Codes", "Upload file"], horizontal=True, key="jewel_alloc_mode")
    raw_codes = []
    if input_mode == "Paste Jewel Codes":
        pasted = st.text_area(
            "Jewel Codes — one per line, or comma/space separated", height=140, key="jewel_alloc_paste",
        )
        raw_codes = re.split(r"[\s,]+", pasted.strip()) if pasted.strip() else []
    else:
        jewel_file = st.file_uploader(
            "File with a 'Jewel Code' column (.csv or .xlsx)", type=["csv", "xlsx"], key="jewel_alloc_file",
        )
        if jewel_file is not None:
            try:
                jf_df = pd.read_csv(jewel_file) if jewel_file.name.lower().endswith(".csv") else pd.read_excel(jewel_file)
                code_col = next((c for c in jf_df.columns if str(c).strip().lower() in ("jewel code", "jewelcode")), None)
                if code_col is None:
                    st.error(f"No 'Jewel Code' column found. Columns in file: {', '.join(str(c) for c in jf_df.columns)}")
                else:
                    raw_codes = jf_df[code_col].dropna().astype(str).tolist()
            except Exception as e:
                st.error(f"Could not read file: {e}")

    input_codes = sorted({c.strip().upper() for c in raw_codes if c.strip()})

    if not input_codes:
        st.info("Paste or upload Jewel Codes above to get started.")
    elif all_pieces.empty:
        st.warning("No stock loaded — fetch the Stock API in the sidebar first.")
    elif sales_daily is None or memo_daily is None:
        st.error("reference_sales_daily.csv or reference_memo_daily.csv not found. Use 'Recompute Base Stock' in the sidebar to build them.")
    else:
        all_lookup = all_pieces.copy()
        all_lookup["Jewel Code"] = all_lookup["Jewel Code"].astype(str).str.strip().str.upper()
        matched_rows = all_lookup[all_lookup["Jewel Code"].isin(input_codes)].drop_duplicates("Jewel Code")
        not_found = sorted(set(input_codes) - set(matched_rows["Jewel Code"]))

        if not_found:
            preview = ", ".join(not_found[:30]) + (" …" if len(not_found) > 30 else "")
            st.warning(f"{len(not_found)} code(s) not found in current stock: {preview}")
            with st.expander("Why weren't these found?"):
                if not os.path.exists(STOCK_API_OUT):
                    st.caption("Raw Stock API snapshot not on disk — can't diagnose further.")
                else:
                    raw_stock = pd.read_csv(STOCK_API_OUT, dtype=str)
                    raw_stock["Jewel Code"] = raw_stock["Jewel Code"].astype(str).str.strip().str.upper()
                    reasons = []
                    for code in not_found:
                        match = raw_stock[raw_stock["Jewel Code"] == code]
                        if match.empty:
                            reason = "Not in the current Stock API snapshot at all (sold, typo, or discontinued)"
                        else:
                            base_metal = match.iloc[0]["Base Metal"]
                            brand = brand_from_base_metal(pd.Series([base_metal])).iloc[0]
                            if pd.isna(brand):
                                reason = f"Base Metal '{base_metal}' isn't Sparkles or Sparq — not tracked by brand"
                            else:
                                reason = "In the snapshot, but excluded for another reason (check Style No/Category)"
                        reasons.append((code, reason))
                    render_html_table(pd.DataFrame(reasons, columns=["Jewel Code", "Reason"]))

        if matched_rows.empty:
            st.info("None of the codes you entered match any current stock right now.")
        else:
            alloc_match_keys = ["Style No", "Price Point", "Category"]
            st.caption("Lifetime SRP % — Sales ÷ Memo Issue, both since the piece was first issued. Fixed, no period to pick.")

            lifecycle_available = sales_events is not None and memo_events is not None and memo_return_events is not None
            if lifecycle_available:
                alloc_lifecycle_anchor = memo_events["Date"].max()
                sold_pieces_alloc, _ = compute_piece_lifecycle(sales_events, memo_events, memo_return_events, alloc_lifecycle_anchor)
            else:
                st.caption("⚠️ Avg Days to Sell / Avg Stock Return Count unavailable — reference_*_events.csv not found. Click 'Recompute Base Stock' in the sidebar to build them.")

            any_brand = False
            for b in ["Sparkles", "Sparq"]:
                rows_b = matched_rows[matched_rows["Brand"] == b]
                if rows_b.empty:
                    continue
                any_brand = True

                srp_p = srp_lifetime_pivot(sales_daily, memo_daily, alloc_match_keys, "Store Name", brand=b)
                srp_p = add_best_column(srp_p, alloc_match_keys, label="Best Store")

                result = rows_b.merge(
                    srp_p[alloc_match_keys + ["Best Store", "Best Store Value"]], on=alloc_match_keys, how="left",
                )

                stock_by_store = stock_grouped[stock_grouped["Brand"] == b].groupby(
                    alloc_match_keys + ["Store Name"]
                )["Stock"].sum().reset_index()
                result = result.merge(
                    stock_by_store, left_on=alloc_match_keys + ["Best Store"],
                    right_on=alloc_match_keys + ["Store Name"], how="left",
                ).drop(columns=["Store Name"])

                base_by_store = base_stock[base_stock["Brand"] == b].groupby(
                    alloc_match_keys + ["Store Name"]
                )["BaseStock"].sum().reset_index()
                result = result.merge(
                    base_by_store, left_on=alloc_match_keys + ["Best Store"],
                    right_on=alloc_match_keys + ["Store Name"], how="left",
                ).drop(columns=["Store Name"])

                return_p = return_rate_pivot(memo_daily, memo_return_daily, alloc_match_keys, "Store Name", brand=b)
                return_long = return_p.melt(id_vars=alloc_match_keys, var_name="Store Name", value_name="ReturnRate")
                result = result.merge(
                    return_long, left_on=alloc_match_keys + ["Best Store"],
                    right_on=alloc_match_keys + ["Store Name"], how="left",
                ).drop(columns=["Store Name"])

                if lifecycle_available:
                    sold_b_alloc = sold_pieces_alloc[sold_pieces_alloc["Brand"] == b]
                    lifecycle_agg = sold_b_alloc.groupby(alloc_match_keys + ["Store Name"]).agg(
                        **{"AvgDaysToSell": ("DaysToSell", "mean"), "AvgCycleCount": ("CycleCount", "mean")}
                    ).reset_index()
                    result = result.merge(
                        lifecycle_agg, left_on=alloc_match_keys + ["Best Store"],
                        right_on=alloc_match_keys + ["Store Name"], how="left",
                    ).drop(columns=["Store Name"])
                else:
                    result["AvgDaysToSell"] = np.nan
                    result["AvgCycleCount"] = np.nan

                result["Stock"] = result["Stock"].fillna(0)
                result["BaseStock"] = result["BaseStock"].fillna(0)
                srp_col = "SRP %"
                return_col = "Return %"
                dts_col = "Avg Days to Sell"
                cyc_col = "Avg Stock Return Count"
                result = result.rename(columns={
                    "Best Store": "Recommended Store", "Best Store Value": srp_col,
                    "Stock": "Stock Available", "BaseStock": "Base Stock", "ReturnRate": return_col,
                    "AvgDaysToSell": dts_col, "AvgCycleCount": cyc_col,
                })
                display_cols = [
                    "Jewel Code", "Style No", "Category", "Price Point", "Current Store", "Recommended Store",
                    srp_col, return_col, dts_col, cyc_col, "Stock Available", "Base Stock",
                ]
                result = result[display_cols].sort_values(srp_col, ascending=False, na_position="last").reset_index(drop=True)

                st.markdown(f"### {b}")
                with st.expander("What each column means"):
                    st.markdown(
                        "- **Current Store** — where the piece is right now; blank means it's fresh, unallocated stock still at HO.\n"
                        "- **Recommended Store** — the store with the highest SRP % for this Style + Price Point + Category. "
                        "Matching Current Store just means the piece is already well-placed — no transfer needed.\n"
                        "- **SRP %** — Recommended Store's lifetime sell-through rate (Sales ÷ Memo Issue). This is what the ranking is on.\n"
                        "- **Return %**, **Avg Days to Sell**, **Avg Stock Return Count**, **Stock Available**, **Base Stock** — "
                        "all context for Recommended Store, not part of the SRP % ranking itself."
                    )
                render_html_table(
                    result.style.format({
                        srp_col: _fmt_func("%.1f%%"),
                        return_col: _fmt_func("%.1f%%"),
                        dts_col: _fmt_func("%.1f"),
                        cyc_col: _fmt_func("%.2f"),
                        "Stock Available": _fmt_func("%.0f"),
                        "Base Stock": _fmt_func("%.2f"),
                    }, na_rep="")
                )
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV", csv, f"jewel_allocation_{b.lower()}.csv", "text/csv",
                    key=f"dl_jewel_alloc_{b.lower()}",
                )

            if not any_brand:
                st.info("No matched pieces in either brand.")

            st.divider()
            st.markdown("## Seasonal Sales History")
            st.write(
                "Past sales qty for these same styles, filtered to **one calendar month across every "
                "year on file** — the same idea as Window 10's Seasonal Trends, scoped to just the "
                "styles you're allocating. Pick a month to see how each Store + Zone has performed in "
                "it historically."
            )
            if sales_calendar is None:
                st.error("reference_sales_calendar.csv not found.")
            else:
                season_month_name = st.selectbox(
                    "Month", list(MONTH_NAMES.values()), index=datetime.now().month - 1, key="alloc_season_month",
                )
                season_month_num = {v: k for k, v in MONTH_NAMES.items()}[season_month_name]

                for b in ["Sparkles", "Sparq"]:
                    st.markdown(f"### {b}")
                    rows_b = matched_rows[matched_rows["Brand"] == b]
                    if rows_b.empty:
                        st.info(f"No matched pieces for {b}.")
                        continue
                    styles_b = rows_b[alloc_match_keys].drop_duplicates()

                    sales_seasonal = sales_calendar[(sales_calendar["Month"] == season_month_num) & (sales_calendar["Brand"] == b)]
                    sales_long = sales_seasonal.merge(styles_b, on=alloc_match_keys, how="inner")

                    if sales_long.empty:
                        st.info(f"No historical sales for these {b} styles in {season_month_name}, across any year on file.")
                        continue

                    seasonal_table = sales_long.groupby(alloc_match_keys + ["Store Name", "Zone"])["JewelCodeCount"].sum().reset_index(name="Units Sold")

                    current_stock_b = (
                        stock_grouped[stock_grouped["Brand"] == b]
                        .groupby(alloc_match_keys + ["Store Name"])["Stock"].sum()
                        .reset_index(name="Current Stock")
                    )
                    seasonal_table = seasonal_table.merge(current_stock_b, on=alloc_match_keys + ["Store Name"], how="left")
                    seasonal_table["Current Stock"] = seasonal_table["Current Stock"].fillna(0).astype(int)

                    seasonal_table = seasonal_table.sort_values("Units Sold", ascending=False).reset_index(drop=True)

                    seasonal_search = st.text_input("Search by Store Name or Style No", key=f"alloc_seasonal_search_{b.lower()}")
                    seasonal_display = seasonal_table
                    if seasonal_search:
                        mask = (
                            seasonal_table["Store Name"].astype(str).str.contains(seasonal_search, case=False, na=False)
                            | seasonal_table["Style No"].astype(str).str.contains(seasonal_search, case=False, na=False)
                        )
                        seasonal_display = seasonal_table[mask]

                    seasonal_cols = ["Style No", "Store Name", "Zone", "Category", "Price Point", "Units Sold", "Current Stock"]
                    render_html_table(seasonal_display[seasonal_cols])
                    seasonal_csv = seasonal_display[seasonal_cols].to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download CSV", seasonal_csv, f"seasonal_history_{season_month_name.lower()}_{b.lower()}.csv", "text/csv",
                        key=f"dl_alloc_seasonal_{b.lower()}",
                    )

