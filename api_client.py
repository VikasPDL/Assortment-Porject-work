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


def _flatten_nested_records(records):
    """The PoddarDiamonds transaction APIs (GetSaleData, GetSaleReturnData,
    GetBranchTransferIssue, GetBranchTransferReceive — confirmed 2026-08-31)
    return one record per transaction/invoice, with the actual piece-level
    fields (JewelCode, Category, BaseMetal, MRP, StyleCode) nested inside a
    per-transaction list (e.g. "JewelTransInward"), not at the top level —
    a transaction with 3 pieces is 1 record with a 3-item nested list, not 3
    records. Explodes each transaction's nested list into its own row,
    carrying the parent's own fields (PartyCode, JewelTransDate, ...) along
    with it — the same one-row-per-piece shape the CSV exports already use.
    If no record has a nested list of dicts, returns records unchanged (so
    a genuinely flat response — e.g. a future Stock API — isn't affected)."""
    nested_key = None
    for record in records:
        if not isinstance(record, dict):
            continue
        for key, value in record.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                nested_key = key
                break
        if nested_key:
            break

    if nested_key is None:
        return pd.DataFrame(records)

    parent_fields = [k for k in records[0].keys() if k != nested_key]
    child_fields = set(records[0][nested_key][0].keys()) if records[0].get(nested_key) else set()
    meta = [f for f in parent_fields if f not in child_fields]  # drop names that would collide (e.g. JewelTransId)
    return pd.json_normalize(records, record_path=nested_key, meta=meta)


def fetch_api_dataframe(url, headers=None, timeout=120):
    """GET a JSON API and return it as a DataFrame. Accepts a bare JSON
    array of records, or an object wrapping the records under a common key
    (data/results/records/items) at any nesting depth. Also unwraps a
    double-encoded response (the whole body is itself a JSON string).
    headers is sent as-is — some APIs use a custom auth header name (e.g.
    "AuthorizationToken" instead of "Authorization"), or expect a date
    range as headers (e.g. "FromDate"/"ToDate"), so the caller builds the
    full header dict rather than this function assuming a fixed shape.
    See _flatten_nested_records for the master-detail unwrapping this does
    before returning — needed for the PoddarDiamonds transaction APIs."""
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

    return _flatten_nested_records(records)


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


# Confirmed field mapping (verified 2026-08-31 against live GetSaleData,
# GetBranchTransferIssue, and GetBranchTransferReceive responses — these
# back the Sales, Memo Issue, and Memo Return fetch buttons respectively).
# After _flatten_nested_records unwraps the per-transaction JewelTransInward
# list, every field TRANSACTION_COLS needs is already present under its
# exact own name (PartyCode, BaseMetal, Category, MRP, JewelTransDate,
# JewelCode) — no renaming needed, so this is mostly an identity map.
#
# GetStockSummary (verified 2026-09-01) is unrelated to the transaction
# shape above — it's already flat (no nested per-transaction list) and uses
# its own field names: ClientCode (blank = unallocated fresh stock, same
# meaning as the Excel upload's Client Code), StyleNo, SalePrice, ItemPcs.
FIXED_FIELD_MAPPING = {
    "PartyCode": "PartyCode",
    "BaseMetal": "BaseMetal",
    "Category": "Category",
    "MRP": "MRP",
    "JewelTransDate": "JewelTransDate",
    "JewelCode": "JewelCode",
    "Client Code": "ClientCode",
    "Jewel Code": "JewelCode",
    "Base Metal": "BaseMetal",
    "Sale Price": "SalePrice",
    "ItemPcs": "ItemPcs",
}
_STYLE_TARGETS = {"StyleCode": "JewelCode", "Style No": "Jewel Code"}


def apply_fixed_mapping(raw_df, targets):
    """Maps raw_df's columns onto targets using FIXED_FIELD_MAPPING. A Style
    column prefers a literal "StyleCode" field when the response has one
    (confirmed present on GetSaleData/GetBranchTransferIssue/Receive after
    flattening); falls back to deriving it from Jewel Code (text before
    '/') for a response that doesn't. Raises ValueError listing exactly
    what's missing if neither the fixed mapping nor the fallback fits this
    response — safer than guessing and silently mismapping a column."""
    out = pd.DataFrame()
    missing = []

    for target in targets:
        if target in _STYLE_TARGETS:
            if "StyleCode" in raw_df.columns:
                out[target] = raw_df["StyleCode"]
                continue
            jewel_target = _STYLE_TARGETS[target]
            jewel_src = FIXED_FIELD_MAPPING.get(jewel_target)
            if jewel_src in raw_df.columns:
                out[target] = raw_df[jewel_src].astype(str).str.split("/").str[0]
            else:
                missing.append(f"{target} (no literal 'StyleCode' field, and needs '{jewel_src}' to derive from)")
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
