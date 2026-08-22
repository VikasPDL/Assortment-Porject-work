import json

import pandas as pd
import requests

# Keys a JSON response's record list might be wrapped under, checked in
# order, searched recursively since some APIs nest it (e.g. {"result":
# {"data": [...]}}).
_RECORD_KEYS = ("data", "results", "records", "items")


def _find_record_list(payload, depth=0, max_depth=3):
    """Return the first list-of-dicts found in payload, searching dict
    values recursively (breadth-limited) so wrapper nesting doesn't matter."""
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            return payload
        return None

    if isinstance(payload, dict) and depth < max_depth:
        # Prefer the conventional key names first, at this level.
        for key in _RECORD_KEYS:
            if key in payload:
                found = _find_record_list(payload[key], depth + 1, max_depth)
                if found is not None:
                    return found
        # Fall back to scanning every value at this level.
        for value in payload.values():
            found = _find_record_list(value, depth + 1, max_depth)
            if found is not None:
                return found

    return None


def fetch_api_dataframe(url, headers=None, timeout=120):
    """GET a JSON API and return it as a DataFrame. Accepts a bare JSON
    array of records, or an object wrapping the records under a common key
    (data/results/records/items) at any nesting depth. Also unwraps a
    double-encoded response (the whole body is itself a JSON string).
    headers is sent as-is — some APIs use a custom auth header name (e.g.
    "AuthorizationToken" instead of "Authorization"), or expect a date
    range as headers (e.g. "FromDate"/"ToDate"), so the caller builds the
    full header dict rather than this function assuming a fixed shape."""
    resp = requests.get(url, headers=headers or {}, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            pass

    records = _find_record_list(payload)
    if records is None:
        preview = resp.text[:1000]
        raise ValueError(
            "Could not find a list of records anywhere in the API response. "
            f"Raw response (first 1000 chars): {preview}"
        )

    return pd.DataFrame(records)


# Common alternate names seen across APIs for each internal column, tried in
# order (case-insensitive) when auto-suggesting a mapping.
COLUMN_ALIASES = {
    "PartyCode": ["PartyCode", "Client", "ClientCode", "Client Code", "StoreCode", "Store Code"],
    "StyleCode": ["StyleCode", "Style No", "StyleNo", "Style Code"],
    "BaseMetal": ["BaseMetal", "Base Metal", "BaseMetalQlyCode", "BaseMetalCode"],
    "Category": ["Category", "GrpName", "Group", "GroupName"],
    "MRP": ["MRP"],
    "JewelTransDate": ["JewelTransDate", "TransactionDate", "TransDate", "Date"],
    "JewelCode": ["JewelCode", "Jewel Code"],
    "Client Code": ["Client Code", "ClientCode", "PartyCode", "Client", "StoreCode"],
    "Style No": ["Style No", "StyleCode", "StyleNo"],
    "Base Metal": ["Base Metal", "BaseMetal", "BaseMetalQlyCode"],
    "Sale Price": ["Sale Price", "MRP", "SalePrice"],
    "ItemPcs": ["ItemPcs", "Qty", "ItemPieces"],
}


def guess_column(target, available_cols):
    """Best-effort guess of which available_cols entry corresponds to
    target, using COLUMN_ALIASES (case-insensitive). Returns None if no
    alias matches anything available — the caller should still let the user
    pick manually."""
    lower_map = {c.lower(): c for c in available_cols}
    for alias in COLUMN_ALIASES.get(target, [target]):
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


# Confirmed field mapping — all three PoddarDiamonds APIs (Sales, Memo
# Issue, Stock) return the same record shape (Client, GrpName, JewelCode,
# TransactionDate, BaseMetalQlyCode, MRP, Qty, ...), so one fixed mapping
# covers all of them. No separate style-code field exists in any of them —
# it's always derived from JewelCode (the text before "/").
FIXED_FIELD_MAPPING = {
    "PartyCode": "Client",
    "BaseMetal": "BaseMetalQlyCode",
    "Category": "GrpName",
    "MRP": "MRP",
    "JewelTransDate": "TransactionDate",
    "JewelCode": "JewelCode",
    "Client Code": "Client",
    "Jewel Code": "JewelCode",
    "Base Metal": "BaseMetalQlyCode",
    "Sale Price": "MRP",
    "ItemPcs": "Qty",
}
_STYLE_TARGETS = {"StyleCode": "JewelCode", "Style No": "Jewel Code"}


def apply_fixed_mapping(raw_df, targets):
    """Maps raw_df's columns onto targets using FIXED_FIELD_MAPPING, deriving
    any Style column from the Jewel Code (text before '/'). Raises ValueError
    listing exactly what's missing if the fixed mapping doesn't fit this
    response — safer than guessing and silently mismapping a column."""
    out = pd.DataFrame()
    missing = []

    for target in targets:
        if target in _STYLE_TARGETS:
            jewel_target = _STYLE_TARGETS[target]
            jewel_src = FIXED_FIELD_MAPPING.get(jewel_target)
            if jewel_src in raw_df.columns:
                out[target] = raw_df[jewel_src].astype(str).str.split("/").str[0]
            else:
                missing.append(f"{target} (needs '{jewel_src}' to derive from)")
            continue

        src = FIXED_FIELD_MAPPING.get(target)
        if src in raw_df.columns:
            out[target] = raw_df[src]
        else:
            missing.append(f"{target} (expected source column '{src}')")

    if missing:
        raise ValueError(
            f"Fixed field mapping didn't match this response. Missing: {', '.join(missing)}. "
            f"Columns received: {', '.join(str(c) for c in raw_df.columns)}"
        )

    return out
