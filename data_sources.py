"""Data sources for exchange-rate history.

Primary source: Google Sheets' GOOGLEFINANCE function. The app writes a
GOOGLEFINANCE formula into a scratch sheet through the Sheets REST API, waits
for the sheet to calculate, then reads the values back.

Fallbacks (so the app is usable before Sheets is configured):
  * Frankfurter (ECB reference rates, free, no key)
  * Synthetic random-walk data (offline demo / testing only)
"""
from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

SHEETS_BASE = "https://sheets.googleapis.com/v4/spreadsheets"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


class DataError(Exception):
    pass


# --------------------------------------------------------------------------
# Google Sheets / GOOGLEFINANCE
# --------------------------------------------------------------------------
def _auth_headers(secrets: dict) -> dict:
    """Build an Authorization header from either an OAuth access token or a
    service-account JSON (both read from Streamlit secrets)."""
    token = secrets.get("access_token")
    if token:
        return {"Authorization": f"Bearer {token}"}

    sa = secrets.get("service_account")
    if sa:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
        except ImportError as e:  # pragma: no cover
            raise DataError("Install google-auth to use a service account.") from e
        creds = service_account.Credentials.from_service_account_info(
            dict(sa), scopes=SCOPES
        )
        creds.refresh(Request())
        return {"Authorization": f"Bearer {creds.token}"}

    raise DataError(
        "No Google credentials found. Add `access_token` or a `service_account` "
        "table under [gsheets] in .streamlit/secrets.toml."
    )


def fetch_googlefinance(base: str, quote: str, days: int, secrets: dict) -> pd.Series:
    """Daily closing rate for base->quote (1 base = X quote) from GOOGLEFINANCE."""
    sheet_id = secrets.get("spreadsheet_id")
    if not sheet_id:
        raise DataError("`spreadsheet_id` missing under [gsheets] in secrets.")
    tab = secrets.get("worksheet", "Sheet1")
    headers = _auth_headers(secrets)
    headers["Content-Type"] = "application/json"

    start = date.today() - timedelta(days=days)
    # Each call uses its own scratch columns far to the right so concurrent
    # users / repeated runs do not trample each other too badly.
    anchor = "A1"
    formula = (
        f'=GOOGLEFINANCE("CURRENCY:{base}{quote}","price",'
        f"DATE({start.year},{start.month},{start.day}),TODAY(),\"DAILY\")"
    )

    # 1) clear the scratch area, 2) write the formula
    rng = f"{tab}!A:B"
    r = requests.post(f"{SHEETS_BASE}/{sheet_id}/values/{rng}:clear",
                      headers=headers, timeout=30)
    if r.status_code >= 400:
        raise DataError(f"Sheets clear failed ({r.status_code}): {r.text[:300]}")

    r = requests.put(
        f"{SHEETS_BASE}/{sheet_id}/values/{tab}!{anchor}",
        params={"valueInputOption": "USER_ENTERED"},
        json={"values": [[formula]]},
        headers=headers,
        timeout=30,
    )
    if r.status_code >= 400:
        raise DataError(f"Sheets write failed ({r.status_code}): {r.text[:300]}")

    # 3) GOOGLEFINANCE calculates asynchronously; poll until data appears.
    rows = None
    for _ in range(12):
        time.sleep(1.5)
        r = requests.get(
            f"{SHEETS_BASE}/{sheet_id}/values/{rng}",
            params={"valueRenderOption": "UNFORMATTED_VALUE",
                    "dateTimeRenderOption": "FORMATTED_STRING"},
            headers=headers,
            timeout=30,
        )
        if r.status_code >= 400:
            raise DataError(f"Sheets read failed ({r.status_code}): {r.text[:300]}")
        vals = r.json().get("values", [])
        if len(vals) > 2 and not (isinstance(vals[0][0], str) and vals[0][0].startswith("#")):
            rows = vals
            break
        if vals and isinstance(vals[0][0], str) and vals[0][0].startswith("#"):
            raise DataError(
                f"GOOGLEFINANCE returned {vals[0][0]} for {base}/{quote}. "
                "Check that both currency codes are valid."
            )
    if rows is None:
        raise DataError("Timed out waiting for GOOGLEFINANCE to calculate.")

    df = pd.DataFrame(rows[1:], columns=["date", "rate"])  # row 0 = header
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise DataError("GOOGLEFINANCE returned no usable rows.")
    s = df.set_index(df["date"].dt.normalize())["rate"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.rename(f"{base}/{quote}")


# --------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------
def fetch_frankfurter(base: str, quote: str, days: int) -> pd.Series:
    start = date.today() - timedelta(days=days)
    url = f"https://api.frankfurter.dev/v1/{start.isoformat()}.."
    r = requests.get(url, params={"base": base, "symbols": quote}, timeout=30)
    if r.status_code >= 400:
        raise DataError(f"Frankfurter error ({r.status_code}): {r.text[:200]}")
    rates = r.json().get("rates", {})
    if not rates:
        raise DataError("Frankfurter has no data for this pair (ECB currencies only).")
    s = pd.Series({pd.Timestamp(d): v[quote] for d, v in rates.items()}).sort_index()
    return s.rename(f"{base}/{quote}")


def synthetic(base: str, quote: str, days: int, seed: int | None = None) -> pd.Series:
    """Offline demo data. NOT real rates."""
    import zlib

    h = zlib.crc32((base + quote).encode())
    rng = np.random.default_rng(seed if seed is not None else h)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=int(days * 5 / 7))
    r = rng.normal(0.0001, 0.006, len(idx))
    start = 0.5 + (h % 1000) / 200
    return pd.Series(start * np.exp(np.cumsum(r)), index=idx, name=f"{base}/{quote}")


def load_rates(source: str, base: str, quote: str, days: int, gsheets_secrets: dict) -> pd.Series:
    if base == quote:
        raise DataError("Pick two different currencies.")
    if source == "Google Sheets (GOOGLEFINANCE)":
        s = fetch_googlefinance(base, quote, days, gsheets_secrets)
    elif source == "Frankfurter (ECB)":
        s = fetch_frankfurter(base, quote, days)
    else:
        s = synthetic(base, quote, days)
    if len(s) < 120:
        raise DataError(f"Only {len(s)} data points available; need at least 120 for a forecast.")
    return s