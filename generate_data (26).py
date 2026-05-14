"""
MintingM Data Engine
====================
Generates data.json, vix_data.json, breadth_data.json
Run daily via GitHub Actions at 1:00 PM IST (07:30 UTC)

Sources:
  - AMFI India (NAVAll.txt)   → current NAV + fund metadata
  - mfapi.in                  → historical NAV per fund
  - yfinance                  → Nifty, Sensex, Gold, VIX, breadth

Output files (same schema your index.html expects):
  data.json       → fund universe, scores, portfolios, backtest
  vix_data.json   → Nifty/VIX percentile + Nifty 50 history
  breadth_data.json → Nifty 500 breadth (stocks above 200 SMA)
"""

import requests
import json
import time
import math
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

RISK_FREE_RATE    = 0.065   # 6.5% — 10Y Gsec
GOLD_THRESHOLD    = 9.0    # Sensex/Gold ratio threshold
GOLD_ACTIVE_SPLIT = 0.70   # 70% of equity bucket → Gold when active

# ─────────────────────────────────────────────
# UNIVERSE CONFIGURATION
# All regular-plan growth funds with 3Y+ history are included.
# No hardcoded list. No pre-score gate. Everything auto-discovered from AMFI daily.
# New funds automatically enter once they hit MIN_HISTORY_YEARS.
# Wound-down/merged funds automatically drop out when their NAV goes stale.
# ─────────────────────────────────────────────

# Minimum NAV history to be included in screener
# Set to 0 — include ALL funds that have any mfapi history
# Funds with no mfapi history at all are excluded (can't compute any returns)
MIN_HISTORY_YEARS = 0

# Fund universe cap per asset type — keeps GitHub Actions runtime under 35 minutes
# Pre-scored by fund age (lower AMFI code = older = more established)
# This means newer niche/NFO funds are excluded but all major funds are included
MAX_FUNDS_EQUITY       = 700
MAX_FUNDS_DEBT         = 350
MAX_FUNDS_GOLD         = 40
MAX_FUNDS_INTERNATIONAL = 80
MAX_FUNDS_HYBRID       = 200

# For portfolio selection: only funds with 5Y history are eligible
# (more reliable scores for picking a model portfolio)
MIN_PORTFOLIO_YEARS = 5

# AMFI sub-category → asset type mapping
EQUITY_SUBCATS = [
    # Core equity
    "flexi cap", "large cap", "mid cap", "small cap",
    "large & mid cap", "multi cap", "focused fund",
    "contra fund", "value fund", "elss",
    # Hybrid equity
    "balanced advantage", "dynamic asset allocation",
    "aggressive hybrid", "equity savings", "conservative hybrid",
    # Sectoral / Thematic
    "sectoral fund", "thematic fund", "banking", "financial services",
    "infrastructure", "technology", "pharma", "healthcare", "consumption",
    "energy", "manufacturing", "psu", "mnc", "dividend yield",
    "esg", "global", "overseas",
    # Index
    "index fund", "index",
]
DEBT_SUBCATS = [
    "short duration", "corporate bond", "banking and psu",
    "medium duration", "low duration", "money market", "ultra short duration",
    # Additional debt
    "liquid", "overnight", "long duration", "gilt",
    "floater", "credit risk", "dynamic bond", "medium to long",
]
GOLD_SUBCATS = ["gold etf", "gold fund", "gold fof", "gold savings"]

HYBRID_SUBCATS = [
    "multi asset allocation", "arbitrage", "balanced hybrid",
]

# International — FoF Overseas (invest in foreign funds/ETFs from India)
INTL_SUBCATS = ["fof overseas"]

# Fund name filters — skip these regardless of category
EXCLUDE_KEYWORDS = [
    "segregated", " series ", "series i ", "series ii", "series iii",
    "series iv", "series v", "series vi", "series vii", "series viii",
    "fixed term", "ftf", "capital protection", "unclaimed", "discontinued",
    "eco plan", "wealth plan", "retail plan", "super institutional",
    "interval fund", "close ended", "bonus option",
]

# Profile definitions — weights must add to 1.0
# Aggressive: 90% equity (Mid30+Small30+Large20+Value10) + 10% gold, 0% debt
# Moderate:   55% equity (Flexi20+MeanRev20+Hybrid15) + 25% gold + 20% debt
# Conservative: 30% equity (Hybrid) + 35% gold + 35% debt (3 equal funds)
# NOTE: Allocation is FIXED — no SG ratio switching in portfolio money allocation
# SG ratio / gold switching is kept ONLY in backtest simulation
PROFILES = {
    "C": {"eq": 0.30, "gold": 0.35, "debt": 0.35, "label": "Conservative"},
    "M": {"eq": 0.55, "gold": 0.25, "debt": 0.20, "label": "Moderate"},
    "A": {"eq": 0.90, "gold": 0.10, "debt": 0.00, "label": "Aggressive"},
}

# ─────────────────────────────────────────────
# STEP 1 — FETCH AMFI CURRENT NAV
# ─────────────────────────────────────────────

def fetch_amfi_nav():
    """Parse all NAVs from AMFI flat file. Returns dict: code → fund info."""
    print("📥 Fetching AMFI NAV file...")
    url = "https://www.amfiindia.com/spages/NAVAll.txt"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    funds = {}
    current_cat = ""
    current_amc = ""

    for line in r.text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Open Ended") or line.startswith("Close Ended") or line.startswith("Interval"):
            current_cat = line
        elif ";" in line:
            parts = line.split(";")
            if len(parts) == 6 and parts[4] not in ('', 'N.A.', '-'):
                try:
                    code = int(parts[0].strip())
                    funds[code] = {
                        "code": code,
                        "isin": parts[1].strip(),
                        "name": parts[3].strip(),
                        "nav_latest": float(parts[4].strip()),
                        "nav_date": parts[5].strip(),
                        "amfi_cat": current_cat,
                        "amc": current_amc,
                    }
                except (ValueError, IndexError):
                    pass
        else:
            current_amc = line

    print(f"   ✅ AMFI: {len(funds)} funds loaded")
    return funds


def fetch_aum():
    """
    Fetch scheme-wise AUM from AMFI.
    Returns dict: scheme_code (int) → aum_crores (float)
    Falls back gracefully to empty dict if all sources fail.

    Sources tried in order:
    1. AMFI AumLoad POST — scheme-wise AAUM (Average AUM)
    2. AMFI scheme-wise HTML page — parsed table
    Both are free, no auth needed, work from GitHub Actions.
    """
    print("📊 Fetching AUM data from AMFI...")
    aum_map = {}
    headers = {
        "User-Agent":   "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":       "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer":      "https://www.amfiindia.com/",
        "Origin":       "https://www.amfiindia.com",
    }

    # ── Method 1: AMFI AumLoad POST ─────────────────────────────────────────
    # Returns pipe/semicolon-separated: SchemeName;SchemeCode;AAUM;AUM
    try:
        url   = "https://www.amfiindia.com/modules/AumLoad"
        forms = [
            {"mf": "0", "tp": "1", "frm": "N"},
            {"mf": "0", "tp": "1"},
            {},
        ]
        for form_data in forms:
            r = requests.post(url, data=form_data, headers=headers, timeout=20)
            if r.status_code == 200 and len(r.text) > 500:
                sep = ";" if ";" in r.text[:200] else "|"
                found = 0
                for line in r.text.splitlines():
                    parts = line.strip().split(sep)
                    # Try different column orderings
                    for code_col, aum_col in [(0, 2), (1, 2), (0, 3), (1, 3)]:
                        if len(parts) > max(code_col, aum_col):
                            try:
                                code    = int(parts[code_col].strip())
                                aum_val = float(parts[aum_col].strip().replace(",", ""))
                                if aum_val > 0:
                                    aum_map[code] = round(aum_val, 2)
                                    found += 1
                                    break
                            except (ValueError, IndexError):
                                continue
                if found > 100:
                    print(f"   ✅ AUM loaded via POST: {found} schemes")
                    return aum_map
    except Exception as e:
        print(f"   ⚠ AMFI POST failed: {e}")

    # ── Method 2: AMFI scheme-wise HTML page ────────────────────────────────
    try:
        from bs4 import BeautifulSoup
        url = "https://www.amfiindia.com/research-information/aum-data/aum-scheme-wise"
        r = requests.get(url, headers=headers, timeout=25)
        if r.status_code == 200 and len(r.text) > 1000:
            soup = BeautifulSoup(r.text, "html.parser")
            found = 0
            for table in soup.find_all("table"):
                for row in table.find_all("tr"):
                    cells = [td.get_text(strip=True).replace(",", "") for td in row.find_all(["td", "th"])]
                    for i, cell in enumerate(cells):
                        try:
                            code = int(cell)
                            # Look for AUM value in adjacent cells
                            for j in range(i+1, min(i+5, len(cells))):
                                try:
                                    aum_val = float(cells[j])
                                    if aum_val > 0:
                                        aum_map[code] = round(aum_val, 2)
                                        found += 1
                                        break
                                except ValueError:
                                    continue
                        except ValueError:
                            continue
            if found > 100:
                print(f"   ✅ AUM loaded via HTML: {found} schemes")
                return aum_map
    except Exception as e:
        print(f"   ⚠ AMFI HTML scrape failed: {e}")

    # ── Method 3: BSE MF Scheme Master ──────────────────────────────────────
    try:
        import io
        url = "https://www.bseindia.com/mutual_fund/Mutual_Fund_Scheme_Master.aspx"
        r = requests.get(url, headers=headers, timeout=25)
        if r.status_code == 200 and len(r.text) > 1000:
            # BSE returns CSV with columns including Scheme Code and AUM
            found = 0
            for line in r.text.splitlines():
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        code    = int(parts[0].strip())
                        # AUM is typically in column 4 or 5
                        for col in [4, 5, 6]:
                            if col < len(parts):
                                aum_val = float(parts[col].strip().replace('"', ''))
                                if aum_val > 0:
                                    aum_map[code] = round(aum_val, 2)
                                    found += 1
                                    break
                    except (ValueError, IndexError):
                        continue
            if found > 100:
                print(f"   ✅ AUM loaded via BSE: {found} schemes")
                return aum_map
    except Exception as e:
        print(f"   ⚠ BSE fetch failed: {e}")

    print("   ⚠ All AUM sources failed — AUM will show as N/A")
    print("   (This is non-critical — all other data is unaffected)")
    return {}


# ─────────────────────────────────────────────
# STEP 2 — FETCH HISTORICAL NAV FROM mfapi.in
# ─────────────────────────────────────────────

def fetch_history(scheme_code, amfi_nav_date_str, min_years=1, _session=None):
    """
    Get historical NAV from mfapi.in.
    amfi_nav_date_str: nav_date from AMFI file — used for freshness check.
    Returns DataFrame [date, nav] sorted ascending, or None if stale/insufficient.

    Timeout strategy:
    - First try: 10s timeout (fast)
    - On timeout: one retry with 15s timeout
    - On second timeout: skip (saves 25s vs waiting on a dead connection)
    """
    # Freshness check before slow mfapi call
    try:
        amfi_date = datetime.strptime(amfi_nav_date_str, "%d-%b-%Y").date()
        days_old  = (date.today() - amfi_date).days
        if days_old > 7:
            return None
        if amfi_date.year < 2024:
            return None
    except Exception:
        pass

    headers = {"User-Agent": "Mozilla/5.0 (compatible; MintingM/1.0)"}
    url     = f"https://api.mfapi.in/mf/{scheme_code}"
    sess    = _session or requests.Session()

    raw = None
    for attempt, timeout in enumerate([10, 15], 1):
        try:
            r = sess.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                raw = r.json()
                break
            else:
                return None
        except requests.exceptions.Timeout:
            if attempt == 1:
                time.sleep(0.5)   # brief pause before retry
                continue          # retry with longer timeout
            else:
                # Second timeout — skip this fund, don't waste more time
                return None
        except Exception as e:
            print(f"   ❌ mfapi error for {scheme_code}: {e}")
            return None

    if not raw or "data" not in raw or not raw["data"]:
        return None

    df = pd.DataFrame(raw["data"])
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
    df         = df.dropna().sort_values("date").reset_index(drop=True)

    if df.empty:
        return None

    # Secondary freshness check
    mfapi_latest = df["date"].iloc[-1].date()
    if (date.today() - mfapi_latest).days > 7 or mfapi_latest.year < 2024:
        return None

    years_available = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    if years_available < min_years:
        return None

    return df


# ─────────────────────────────────────────────
# STEP 3 — COMPUTE METRICS FROM NAV HISTORY
# ─────────────────────────────────────────────

def cagr(df, years):
    """
    Compute CAGR for given lookback period.
    Anchors on TODAY (not latest NAV date) to match Moneycontrol/AMFI standard.
    e.g. 1Y = from (today - 1 year) to today's closest NAV.
    """
    today = pd.Timestamp(date.today())

    # End: closest NAV on or before today
    end_sub = df[df["date"] <= today]
    if end_sub.empty:
        return None
    end_nav  = end_sub["nav"].iloc[-1]
    end_date = end_sub["date"].iloc[-1]

    # Start: closest NAV on or before (today - N years)
    start_target = today - pd.DateOffset(years=years)
    start_sub = df[df["date"] <= start_target]
    if len(start_sub) < 5:
        return None
    start_nav  = start_sub["nav"].iloc[-1]
    start_date = start_sub["date"].iloc[-1]

    actual_years = (end_date - start_date).days / 365.25
    if actual_years < 0.5 or start_nav <= 0:
        return None

    return round(((end_nav / start_nav) ** (1 / actual_years) - 1) * 100, 2)


def absolute_return(df, months):
    """
    Absolute return over N months — NOT annualised.
    Matches Angel One / Moneycontrol for 1M and 3M.
    
    Tolerance: start NAV must be within ±15 days of target date.
    This prevents funds with limited history from showing inflated returns
    (e.g. a fund launched 13 months ago would show a ~13M return as 1Y).
    """
    today      = pd.Timestamp(date.today())
    tolerance  = pd.Timedelta(days=15)     # ±15 days acceptable for monthly

    end_sub = df[df["date"] <= today]
    if end_sub.empty:
        return None
    end_nav = end_sub["nav"].iloc[-1]

    start_target = today - pd.DateOffset(months=months)
    start_sub    = df[df["date"] <= start_target]
    if start_sub.empty:
        return None

    start_nav  = start_sub["nav"].iloc[-1]
    start_date = start_sub["date"].iloc[-1]

    # Validate: found date must be within tolerance of target
    if abs((start_date - start_target).days) > tolerance.days:
        return None

    if start_nav <= 0:
        return None

    return round((end_nav / start_nav - 1) * 100, 2)


def cagr(df, years):
    """
    Annualised CAGR over N years — anchored on date.today().
    Matches Moneycontrol / Angel One / AMFI standard.

    Tolerance: start NAV must be within ±45 days of target date.
    This prevents funds with limited history showing inflated multi-year returns.
    For example a fund with only 3.5Y of data should return None for 5Y, not
    use its oldest available NAV and call it a 5Y return.
    """
    today     = pd.Timestamp(date.today())
    tolerance = 45   # days — generous for weekends and holidays

    end_sub = df[df["date"] <= today]
    if end_sub.empty:
        return None
    end_nav  = end_sub["nav"].iloc[-1]
    end_date = end_sub["date"].iloc[-1]

    start_target = today - pd.DateOffset(years=years)
    start_sub    = df[df["date"] <= start_target]
    if len(start_sub) < 5:
        return None

    start_nav  = start_sub["nav"].iloc[-1]
    start_date = start_sub["date"].iloc[-1]

    # Validate: found start date must be within tolerance of target
    if abs((start_date - start_target).days) > tolerance:
        return None

    actual_years = (end_date - start_date).days / 365.25
    if actual_years < 0.75 or start_nav <= 0:
        return None

    return round(((end_nav / start_nav) ** (1 / actual_years) - 1) * 100, 2)


def compute_metrics(df):
    """
    Compute all risk/return metrics from daily NAV.
    - 1M, 3M: absolute return (not annualised) — matches Angel One
    - 1Y, 3Y, 5Y, 7Y, 10Y: CAGR annualised — anchored on date.today()
    """
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    df = df.dropna(subset=["ret"])

    if len(df) < 20:  # need at least ~1 month of data to compute anything
        return None

    # Absolute returns (sub-1Y)
    r1m = absolute_return(df, 1)
    r3m = absolute_return(df, 3)

    # Annualised CAGR returns
    r1  = cagr(df, 1)
    r3  = cagr(df, 3)
    r5  = cagr(df, 5)
    r7  = cagr(df, 7)
    r10 = cagr(df, 10)

    # Use longest available CAGR for risk metrics base
    ann_ret_val = next((v for v in [r10, r7, r5, r3, r1] if v is not None), None)
    if ann_ret_val is None:
        return None

    daily_std = df["ret"].std()
    ann_std = daily_std * math.sqrt(252)

    sharpe = round((ann_ret_val / 100 - RISK_FREE_RATE) / ann_std, 3) if ann_std > 0 else None

    down_ret = df[df["ret"] < 0]["ret"]
    down_std = down_ret.std() * math.sqrt(252) if len(down_ret) > 10 else ann_std
    sortino = round((ann_ret_val / 100 - RISK_FREE_RATE) / down_std, 3) if down_std > 0 else None

    roll_max = df["nav"].cummax()
    drawdown = (df["nav"] - roll_max) / roll_max
    max_dd = round(float(drawdown.min()), 4)

    calmar = round((ann_ret_val / 100) / abs(max_dd), 3) if max_dd != 0 else None

    df["year"] = df["date"].dt.year
    annual_rets = df.groupby("year")["nav"].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 20 else None
    ).dropna()
    win_rate = round((annual_rets > 0).sum() / len(annual_rets) * 100, 1) if len(annual_rets) > 0 else None

    return {
        "r1m": r1m, "r3m": r3m,                          # absolute
        "r1": r1, "r3": r3, "r5": r5, "r7": r7, "r10": r10,  # annualised CAGR
        "sharpe": sharpe,
        "std_dev": round(ann_std, 4),
        "max_dd": max_dd,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "ann_ret": ann_ret_val,
        "_annual_rets": annual_rets.to_dict(),
    }


# ─────────────────────────────────────────────
# STEP 4 — CLASSIFY FUND TYPE FROM AMFI CATEGORY
# ─────────────────────────────────────────────

def classify_type(amfi_cat, name):
    """Return Equity / Debt / Gold / Other based on AMFI category string."""
    cat_lower = amfi_cat.lower() + " " + name.lower()

    for kw in GOLD_CATS:
        if kw in cat_lower:
            return "Gold"

    for kw in EQUITY_CATS:
        if kw in cat_lower:
            return "Equity"

    for kw in DEBT_CATS:
        if kw in cat_lower:
            return "Debt"

    return "Other"


def extract_category(amfi_cat):
    """Clean sub-category name from raw AMFI category string."""
    # Remove the outer wrapper like "Open Ended Schemes(Equity Scheme - "
    import re
    m = re.search(r'\((.+?)\)', amfi_cat)
    if m:
        inner = m.group(1)
        # Remove leading "Equity Scheme - " etc.
        for prefix in ["Equity Scheme - ", "Debt Scheme - ", "Other Scheme - ",
                        "Hybrid Scheme - ", "Solution Oriented Scheme - "]:
            if inner.startswith(prefix):
                return inner[len(prefix):]
        return inner
    return amfi_cat


# ─────────────────────────────────────────────
# STEP 5 — MINTINGM SCORE
# Formula from index.html Formula Guide:
# Step 0: Hard filters — Sharpe > 1.0, DD -20% to -30% (Gold: DD exempt)
# Step 1: Raw = 0.50 x RetWt + 0.25 x (Sharpe x 0.08) - 0.25 x StdDev
#         RetWt = 10Y:25% + 7Y:25% + 5Y:25% + 3Y:15% + 1Y:10%
# Step 2: Normalize 0-10 within each asset type peer group
# ─────────────────────────────────────────────

def compute_raw_score(m, asset_type):
    """Returns raw score + filter flags. Normalization happens later."""
    # sf = Sharpe >= 1.0 (good), df = DD better than -25% (good)
    # fp = both pass — kept in output for schema compatibility only (not shown in screener)
    sf = bool((m["sharpe"] is not None) and (m["sharpe"] >= 1.0))
    if asset_type in ("Gold", "Debt"):
        df_flag = True   # DD filter not applied to Gold or Debt
    else:
        df_flag = bool((m["max_dd"] is not None) and (m["max_dd"] >= -0.25))

    fp = bool(sf and df_flag)

    # RetWt — weighted across available periods
    weights = [
        (m["r10"], 0.25),
        (m["r7"],  0.25),
        (m["r5"],  0.25),
        (m["r3"],  0.15),
        (m["r1"],  0.10),
    ]
    total_w, total_r = 0.0, 0.0
    for val, w in weights:
        if val is not None:
            total_r += val * w
            total_w += w

    if total_w < 0.10:   # essentially no return data
        return {"raw_score": 0.0, "sf": sf, "df": df_flag, "fp": fp}

    ret_wt = (total_r / total_w) / 100.0   # normalise to decimal
    sharpe_val = m["sharpe"] if m["sharpe"] is not None else 0.0
    std_val = m["std_dev"] if m["std_dev"] is not None else 0.25

    raw = 0.50 * ret_wt + 0.25 * (sharpe_val * 0.08) - 0.25 * std_val

    return {"raw_score": raw, "sf": sf, "df": df_flag, "fp": fp}


def normalize_scores(funds):
    """Min-max scale raw_score to 0-10 within each asset type."""
    for asset_type in ["Equity", "Debt", "Gold", "International"]:
        group = [f for f in funds if f.get("type") == asset_type and f.get("live") and "raw_score" in f]
        if not group:
            continue
        raws = [f["raw_score"] for f in group]
        mn, mx = min(raws), max(raws)
        for f in group:
            if mx > mn:
                f["score"] = round((f["raw_score"] - mn) / (mx - mn) * 10, 2)
            else:
                f["score"] = 5.0
            del f["raw_score"]   # clean up temp field
    return funds


# ─────────────────────────────────────────────
# STEP 6 — AUTO-SELECT PORTFOLIO FUNDS BY SCORE
# ─────────────────────────────────────────────

def tag_funds(funds_raw, nifty_1y, nifty_3y, nifty_5y):
    """
    Tag each fund with:
      momentum     = 1Y return > Nifty 50 1Y return
      mean_rev     = 1Y return < Nifty 50 1Y BUT 3Y AND 5Y > Nifty benchmarks
      international = fund has overseas allocation in mandate
    These tags are written into the fund JSON and displayed as badges in the screener.
    """
    # Funds with overseas allocation not evident from name — hardcoded
    INTL_CODES = {
        122640,   # Parag Parikh Flexi Cap — 15-20% overseas
        147946,   # Mirae Asset Emerging Bluechip (some intl exposure)
    }

    # Keywords that signal international mandate in fund name
    INTL_KEYWORDS = [
        "overseas", "international", "global", "world",
        "nasdaq", "n100", "s&p", "us equity", "us fund",
        "hang seng", "europe", "emerging market", "emerg mkt",
        "feeder", "fof overseas",
    ]

    for f in funds_raw:
        r1  = f.get("r1")
        r3  = f.get("r3")
        r5  = f.get("r5")

        # Momentum: 1Y above Nifty
        if r1 is not None and nifty_1y is not None:
            f["momentum"] = bool(r1 > nifty_1y)
        else:
            f["momentum"] = False

        # Mean reversion: 1Y below Nifty BUT 3Y and 5Y above Nifty
        mr = False
        if (r1 is not None and nifty_1y is not None and r1 < nifty_1y):
            r3_ok = (r3 is not None and nifty_3y is not None and r3 > nifty_3y)
            r5_ok = (r5 is not None and nifty_5y is not None and r5 > nifty_5y)
            mr = bool(r3_ok and r5_ok)
        f["mean_rev"] = mr

        # International flag — all FoF Overseas funds + specific domestic funds with overseas allocation
        name_lower = f.get("name", "").lower()
        f["international"] = bool(
            f.get("type") == "International" or   # all FoF Overseas funds
            f.get("code") in INTL_CODES or        # specific domestic funds (Parag Parikh)
            any(kw in name_lower for kw in INTL_KEYWORDS)
        )

    return funds_raw


def select_portfolio_funds(scored_funds, profile_key, nifty_1y, nifty_3y, nifty_5y):
    """
    Select exactly 5 funds per profile.
    Uses Nifty 50 returns as benchmark for momentum/mean-reversion classification.

    AGGRESSIVE  (90% equity / 10% gold / 0% debt):
      Slot 1: Mid Cap        — momentum  (1Y > Nifty 1Y)
      Slot 2: Small Cap      — momentum  (1Y > Nifty 1Y)
      Slot 3: Large Cap / Flexi Cap — momentum (1Y > Nifty 1Y, different AMC)
      Slot 4: Value / Contra — mean reversion (1Y < Nifty BUT 3Y/5Y strong)
      Slot 5: Gold ETF

    MODERATE  (55% equity / 25% gold / 20% debt):
      Slot 1: Flexi Cap / Multi Cap — momentum (1Y > Nifty 1Y)
      Slot 2: Any equity            — mean reversion (1Y < Nifty, 3Y/5Y strong, diff AMC)
      Slot 3: Balanced Adv / Aggressive Hybrid — best MintingM score
      Slot 4: Gold ETF
      Slot 5: Short Duration / Corporate Bond debt

    CONSERVATIVE  (30% equity / 35% gold / 35% debt):
      Slot 1: Aggressive Hybrid / Balanced Adv — mean reversion (stable, lower vol)
      Slot 2: Gold ETF
      Slot 3: Short Duration debt
      Slot 4: Corporate Bond / Banking & PSU   (different AMC from slot 3)
      Slot 5: Low Duration                     (different AMC from slots 3 & 4)
    """

    def top(asset_type, keywords, n=1, exclude_ids=None, exclude_amcs=None,
            require_momentum=False, require_mean_rev=False, exclude_international=False):
        """Get top n funds by MintingM score with optional filters."""        exclude_ids  = exclude_ids  or []
        exclude_amcs = exclude_amcs or []
        dd_cap = -0.50 if profile_key == "A" else -0.40
        pool = [
            f for f in scored_funds
            if f.get("type")       == asset_type
            and f.get("live",   False)
            and f.get("score",  0) > 0
            and f["id"]            not in exclude_ids
            and any(kw.lower() in f.get("cat", "").lower() for kw in keywords)
            and (not exclude_amcs or f["name"].split()[0] not in exclude_amcs)
            and (not require_momentum  or f.get("momentum",  False))
            and (not require_mean_rev  or f.get("mean_rev",  False))
            and (not exclude_international or not f.get("international", False))
            and (f.get("max_dd") is None or f.get("max_dd", -1) >= dd_cap)
        ]
        pool.sort(key=lambda x: x.get("score", 0), reverse=True)
        # If momentum/mean_rev filter yields nothing, fall back without it
        if not pool and (require_momentum or require_mean_rev):
            pool = [
                f for f in scored_funds
                if f.get("type")   == asset_type
                and f.get("live",  False)
                and f.get("score", 0) > 0
                and f["id"]        not in exclude_ids
                and any(kw.lower() in f.get("cat", "").lower() for kw in keywords)
                and (not exclude_amcs or f["name"].split()[0] not in exclude_amcs)
                and (not exclude_international or not f.get("international", False))
                and (f.get("max_dd") is None or f.get("max_dd", -1) >= dd_cap)
            ]
            pool.sort(key=lambda x: x.get("score", 0), reverse=True)
        return pool[:n]

    def amc(fund):
        return fund["name"].split()[0] if fund else None

    def ids(lst):
        return [f["id"] for f in lst if f]

    def amcs(lst):
        return [amc(f) for f in lst if f]

    picks = []

    if profile_key == "A":
        # Slot 1 — Mid Cap momentum
        s1 = top("Equity", ["Mid Cap"], 1,
                 require_momentum=True)
        picks += s1

        # Slot 2 — Small Cap momentum (different AMC)
        s2 = top("Equity", ["Small Cap"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s1),
                 require_momentum=True)
        picks += s2

        # Slot 3 — Domestic Large Cap / Flexi Cap / Multi Cap momentum (diff AMC from s1, s2)
        # Strictly domestic — international funds go only in International section
        s3 = top("Equity", ["Large Cap", "Flexi Cap", "Multi Cap"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s1 + s2),
                 require_momentum=True,
                 exclude_international=True)
        picks += s3

        # Slot 4 — Value or Contra mean reversion (different AMC)
        s4 = top("Equity", ["Value Fund", "Contra Fund"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s1 + s2 + s3),
                 require_mean_rev=True)
        picks += s4

        # Slot 5 — Gold ETF
        s5 = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += s5

    elif profile_key == "M":
        # Slot 1 — Flexi Cap or Multi Cap momentum
        s1 = top("Equity", ["Flexi Cap", "Multi Cap"], 1,
                 require_momentum=True)
        picks += s1

        # Slot 2 — Any equity mean reversion (different AMC)
        s2 = top("Equity", ["Flexi Cap", "Multi Cap", "Large Cap",
                             "Mid Cap", "Large & Mid Cap"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s1),
                 require_mean_rev=True)
        picks += s2

        # Slot 3 — Balanced Advantage or Aggressive Hybrid (best score, diff AMC)
        s3 = top("Equity", ["Balanced Advantage", "Dynamic Asset", "Aggressive Hybrid"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s1 + s2))
        picks += s3

        # Slot 4 — Gold ETF
        s4 = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += s4

        # Slot 5 — Debt: Short Duration or Corporate Bond
        s5 = (top("Debt", ["Short Duration"], 1, exclude_ids=ids(picks)) or
              top("Debt", ["Corporate Bond"],  1, exclude_ids=ids(picks)))
        picks += s5

    else:  # C — Conservative
        # Slot 1 — Aggressive Hybrid / Balanced Adv — mean reversion preferred
        s1 = top("Equity", ["Aggressive Hybrid", "Balanced Advantage", "Dynamic Asset"], 1,
                 require_mean_rev=True)
        picks += s1

        # Slot 2 — Gold ETF
        s2 = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += s2

        # Slot 3 — Short Duration debt
        s3 = top("Debt", ["Short Duration"], 1, exclude_ids=ids(picks))
        picks += s3

        # Slot 4 — Corporate Bond or Banking & PSU (different AMC from s3)
        s4 = (top("Debt", ["Corporate Bond"], 1,
                  exclude_ids=ids(picks), exclude_amcs=amcs(s3)) or
              top("Debt", ["Banking & PSU", "Banking and PSU"], 1,
                  exclude_ids=ids(picks), exclude_amcs=amcs(s3)))
        picks += s4

        # Slot 5 — Low Duration (different AMC from s3 + s4)
        s5 = top("Debt", ["Low Duration"], 1,
                 exclude_ids=ids(picks), exclude_amcs=amcs(s3 + s4))
        if not s5:
            s5 = top("Debt", ["Short Duration", "Corporate Bond", "Banking & PSU"], 1,
                     exclude_ids=ids(picks))
        picks += s5

    # Clean up any None/empty
    picks = [f for f in picks if f]

    return [
        {
            "id":         f["id"],
            "name":       f["name"],
            "type":       f["type"],
            "cat":        f["cat"],
            "score":      f["score"],
            "code":       f["code"],
            "momentum":   f.get("momentum", False),
            "mean_rev":   f.get("mean_rev",  False),
            "international": f.get("international", False),
        }
        for f in picks
    ]


# ─────────────────────────────────────────────
# STEP 7 — SENSEX/GOLD RATIO (live + history)
# ─────────────────────────────────────────────

def get_sg_ratio_and_history():
    """
    Current SG ratio + annual Dec-31 history from 2000 to now.
    Formula: Sensex index value ÷ Gold price in INR per gram
    (This gives ~9x range historically, matching your threshold logic)
    
    Gold INR/gram = Gold USD/troy_oz × USDINR ÷ 31.1035 (grams per troy oz)
    """
    print("📊 Fetching Sensex + Gold prices...")

    try:
        sensex_cur = float(
            yf.Ticker("^BSESN").history(period="5d")["Close"].dropna().iloc[-1]
        )
        gold_usd = float(
            yf.Ticker("GC=F").history(period="5d")["Close"].dropna().iloc[-1]
        )
        usdinr = float(
            yf.Ticker("INR=X").history(period="5d")["Close"].dropna().iloc[-1]
        )

        # Gold per GRAM in INR
        gold_inr_per_gram = gold_usd * usdinr / 31.1035
        current_ratio     = round(sensex_cur / gold_inr_per_gram, 2)
        print(f"   ✅ Sensex: {sensex_cur:.0f} | Gold/g: ₹{gold_inr_per_gram:.0f} | Ratio: {current_ratio}x")

    except Exception as e:
        print(f"   ❌ Live SG ratio failed: {e} — using fallback 9.4")
        current_ratio = 9.4

    # Historical annual Dec-31 ratios
    print("   📅 Building SG ratio history 2000–present...")
    sg_history = {}

    try:
        s_hist  = yf.Ticker("^BSESN").history(start="1999-01-01")["Close"]
        g_hist  = yf.Ticker("GC=F").history(start="1999-01-01")["Close"]
        fx_hist = yf.Ticker("INR=X").history(start="1999-01-01")["Close"]

        # Strip timezone so asof() works without tz mismatch
        for _h in [s_hist, g_hist, fx_hist]:
            try:
                _h.index = _h.index.tz_localize(None)
            except TypeError:
                _h.index = _h.index.tz_convert(None)

        for year in range(2000, date.today().year + 1):
            try:
                target = pd.Timestamp(f"{year}-12-31")
                s  = float(s_hist.asof(target))
                g  = float(g_hist.asof(target))
                fx = float(fx_hist.asof(target))
                if s > 0 and g > 0 and fx > 0:
                    gold_inr_gram = g * fx / 31.1035
                    sg_history[str(year)] = round(s / gold_inr_gram, 1)
            except Exception:
                pass

        # Override current year with live value
        sg_history[str(date.today().year)] = current_ratio
        print(f"   ✅ SG history years: {sorted(sg_history.keys())}")

    except Exception as e:
        print(f"   ⚠ History fetch failed: {e} — using static fallback")
        sg_history = {
            "2000": 10.8, "2001": 11.2, "2002": 9.6,  "2003": 7.1,
            "2004": 7.8,  "2005": 8.4,  "2006": 9.8,  "2007": 12.4,
            "2008": 13.1, "2009": 6.8,  "2010": 7.2,  "2011": 8.6,
            "2012": 9.4,  "2013": 10.1, "2014": 9.8,  "2015": 8.9,
            "2016": 8.2,  "2017": 8.7,  "2018": 9.3,  "2019": 8.1,
            "2020": 6.9,  "2021": 8.4,  "2022": 9.8,  "2023": 10.2,
            "2024": 9.6,  "2025": 9.5,
            str(date.today().year): current_ratio,
        }

    return current_ratio, sg_history


# ─────────────────────────────────────────────
# STEP 8 — BENCHMARK ANNUAL RETURNS (3 indices)
# Nifty 50, Nifty Midcap 50, Nifty Smallcap 100
# Annual calendar-year returns from 2000 to present
# ─────────────────────────────────────────────

BENCHMARK_TICKERS = {
    "nifty50":    ("^NSEI",      "Nifty 50",         "1999-01-01"),
    "midcap50":   ("^NSEMDCP50", "Nifty Midcap 50",  "2004-01-01"),
    "smallcap100":("^CNXSC",     "Nifty Smallcap 100","2003-01-01"),
}

# Static fallbacks if yfinance is unavailable
BENCHMARK_FALLBACKS = {
    "nifty50": {
        "2000": -14.7, "2001": -16.2, "2002": 3.3,   "2003": 72.9,
        "2004": 10.7,  "2005": 36.3,  "2006": 39.8,  "2007": 54.8,
        "2008": -51.8, "2009": 75.8,  "2010": 17.9,  "2011": -24.6,
        "2012": 27.7,  "2013": 6.8,   "2014": 31.4,  "2015": -4.1,
        "2016": 3.0,   "2017": 28.6,  "2018": 3.2,   "2019": 12.0,
        "2020": 14.9,  "2021": 24.1,  "2022": 4.3,   "2023": 20.0,
        "2024": 8.8,   "2025": 10.1,  "2026": -6.9,
    },
    "midcap50": {
        "2005": 51.7,  "2006": 47.2,  "2007": 72.1,  "2008": -67.2,
        "2009": 116.3, "2010": 19.2,  "2011": -31.0, "2012": 39.4,
        "2013": -5.1,  "2014": 55.7,  "2015": 6.8,   "2016": 3.4,
        "2017": 47.1,  "2018": -10.3, "2019": -4.5,  "2020": 18.2,
        "2021": 39.5,  "2022": 4.8,   "2023": 28.6,  "2024": 12.1,
        "2025": 8.4,   "2026": -12.3,
    },
    "smallcap100": {
        "2004": 62.4,  "2005": 63.2,  "2006": 52.8,  "2007": 89.4,
        "2008": -72.1, "2009": 130.8, "2010": 22.4,  "2011": -42.1,
        "2012": 41.2,  "2013": -6.8,  "2014": 68.3,  "2015": 6.3,
        "2016": -0.1,  "2017": 59.3,  "2018": -26.7, "2019": -9.2,
        "2020": 23.8,  "2021": 55.3,  "2022": 0.2,   "2023": 43.7,
        "2024": 14.3,  "2025": 2.1,   "2026": -18.4,
    },
}

def get_benchmark_annual():
    """
    Fetch annual calendar-year returns for all 3 benchmarks.
    Returns dict: {nifty50: {year: ret}, midcap50: {...}, smallcap100: {...}}
    Each benchmark goes back to its earliest available year from 2000.
    """
    print("📈 Fetching benchmark annual returns (Nifty 50, Midcap 50, Smallcap 100)...")
    result = {}

    for key, (ticker, label, start) in BENCHMARK_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(start=start)["Close"]
            try:
                hist.index = hist.index.tz_localize(None)
            except TypeError:
                hist.index = hist.index.tz_convert(None)
            annual = {}
            start_year = int(start[:4]) + 1
            for year in range(start_year, date.today().year + 1):
                try:
                    s = float(hist.asof(pd.Timestamp(f"{year-1}-12-31")))
                    e = float(hist.asof(pd.Timestamp(f"{year}-12-31")))
                    if s > 0:
                        annual[str(year)] = round((e / s - 1) * 100, 1)
                except Exception:
                    pass
            if annual:
                result[key] = annual
                print(f"   ✅ {label}: {len(annual)} years ({min(annual.keys())}–{max(annual.keys())})")
            else:
                result[key] = BENCHMARK_FALLBACKS[key]
                print(f"   ⚠ {label}: using fallback data")
        except Exception as e:
            result[key] = BENCHMARK_FALLBACKS[key]
            print(f"   ⚠ {label} ({ticker}) failed: {e} — using fallback")

    return result

def get_nifty_annual():
    """Kept for backwards compat — returns nifty50 from get_benchmark_annual."""
    data = get_benchmark_annual()
    return data.get("nifty50", BENCHMARK_FALLBACKS["nifty50"])


# ─────────────────────────────────────────────
# STEP 9 — BACKTEST (DETERMINISTIC, NO RANDOM)
# ─────────────────────────────────────────────

def get_fund_annual_return(fund_annual_rets, year):
    """Get actual annual return for a fund in a given year. Returns None if missing."""
    return fund_annual_rets.get(year)


def _avg_monthly(funds, month):
    """Average monthly return across funds for a given month key."""
    rets = [f.get("_monthly_rets", {}).get(month)
            for f in funds
            if f.get("_monthly_rets", {}).get(month) is not None]
    return sum(rets) / len(rets) if rets else None


def run_backtest(profile_key, eq_ratio, debt_ratio, sg_history,
                 eq_funds, dt_funds, gold_funds):
    """
    Deterministic annual backtest — Option A.
    Fixed START_YEAR = 2013. Uses actual fund annual_rets where available,
    falls back to category-specific historical returns when a fund has no data
    for that year (e.g. fund launched after 2013).
    This gives consistent 13-year backtest across all profiles.
    """

    START_YEAR = 2000   # Extended to 2000; fallbacks cover pre-fund years

    # ── Category-specific fallbacks (actual index returns) ──────────────
    # Used when a fund has no data for a year (launched after that year)
    # Equity: Nifty 50 actual (conservative proxy for mixed equity)
    EQ_FALLBACK = {
        2000:-14.7, 2001:-16.2, 2002:  3.3, 2003: 72.9, 2004: 10.7, 2005: 36.3,
        2006: 39.8, 2007: 54.8, 2008:-51.8, 2009: 75.8, 2010: 17.9, 2011:-24.6,
        2012: 27.7, 2013:  6.8, 2014: 31.4, 2015: -4.1, 2016:  3.0, 2017: 28.6,
        2018:  3.2, 2019: 12.0, 2020: 14.9, 2021: 24.1, 2022:  4.3, 2023: 20.0,
        2024:  8.8, 2025:  5.2, 2026: -9.8,
    }
    MIDSMALL_FALLBACK = {
        2000:-20.0, 2001:-22.0, 2002:  5.0, 2003: 90.0, 2004: 25.0, 2005: 55.0,
        2006: 52.0, 2007: 70.0, 2008:-65.0, 2009:100.0, 2010: 18.0, 2011:-31.0,
        2012: 41.0, 2013: -2.8, 2014: 46.1, 2015:  6.3, 2016:  3.4, 2017: 47.1,
        2018:-10.3, 2019: -4.5, 2020: 26.7, 2021: 43.2, 2022:  4.1, 2023: 39.5,
        2024: 18.2, 2025:  3.1, 2026:-12.4,
    }
    DEBT_FALLBACK = {
        2000:  9.8, 2001: 12.2, 2002: 10.5, 2003:  8.1, 2004:  2.1, 2005:  5.2,
        2006:  5.8, 2007:  7.6, 2008: 14.8, 2009:  3.2, 2010:  5.1, 2011:  8.9,
        2012:  9.2, 2013:  3.5, 2014: 14.1, 2015:  7.6, 2016: 14.5, 2017:  4.5,
        2018:  6.1, 2019: 11.2, 2020: 11.8, 2021:  3.4, 2022:  2.8, 2023:  7.2,
        2024:  8.1, 2025:  7.5, 2026:  8.1,
    }
    GOLD_FALLBACK = {
        2000:  1.8, 2001:  5.5, 2002: 25.2, 2003: 13.2, 2004:  1.0, 2005:  9.6,
        2006: 20.5, 2007: 16.3, 2008: 25.0, 2009: 23.8, 2010: 23.2, 2011: 30.8,
        2012: 12.4, 2013:-12.6, 2014: -1.8, 2015: -9.7, 2016: 11.5, 2017:  5.3,
        2018:  7.8, 2019: 22.6, 2020: 28.0, 2021: -3.6, 2022: 11.2, 2023: 15.1,
        2024: 21.3, 2025: 18.0, 2026: 14.2,
    }

    def is_midsmall(fund):
        cat = (fund.get("cat") or "").lower()
        return any(k in cat for k in ["mid cap", "small cap", "mid & small"])

    def get_avg_annual(funds, year, fallback_map, use_midsmall=False):
        """Average actual annual return; use fallback if fund has no data."""
        rets = []
        for f in funds:
            actual = f.get("_annual_rets", {}).get(year)
            if actual is not None:
                rets.append(actual)
            else:
                # Use category-specific fallback
                if use_midsmall and is_midsmall(f):
                    rets.append(MIDSMALL_FALLBACK.get(year, fallback_map.get(year, 10.0)))
                else:
                    rets.append(fallback_map.get(year, 10.0))
        return round(sum(rets) / len(rets), 2) if rets else fallback_map.get(year, 10.0)

    start_year = START_YEAR
    end_year   = date.today().year

    nav  = 100.0
    rows = []

    for year in range(start_year, end_year + 1):
        eq_ret  = get_avg_annual(eq_funds,   year, EQ_FALLBACK, use_midsmall=True)
        dt_ret  = get_avg_annual(dt_funds,   year, DEBT_FALLBACK)
        gld_ret = get_avg_annual(gold_funds, year, GOLD_FALLBACK) if gold_funds else GOLD_FALLBACK.get(year, 8.0)

        # FIXED allocation — no SG ratio switching
        # Portfolio allocation is always eq_ratio equity + gold_ratio gold + debt_ratio debt
        gold_ratio = 1.0 - eq_ratio - debt_ratio
        port_ret = eq_ratio * (eq_ret or 0) + gold_ratio * gld_ret + debt_ratio * (dt_ret or 0)

        nav *= (1 + port_ret / 100)

        rows.append({
            "year":     year,
            "port_nav": round(nav, 2),
            "port_ret": round(port_ret, 2),
            "regime":   "equity",
        })

    if not rows:
        return {}

    n         = len(rows)
    port_cagr = round(((nav / 100) ** (1 / n) - 1) * 100, 1)
    rets      = [r["port_ret"] for r in rows]

    # Max drawdown from MONTHLY NAV series (accurate intra-year drawdowns)
    # Build portfolio monthly NAV using weighted monthly returns of all funds
    all_funds_bt = eq_funds + dt_funds + gold_funds
    # Get all months available across all funds
    all_months = set()
    for f in all_funds_bt:
        all_months.update(f.get("_monthly_rets", {}).keys())
    all_months = sorted(m for m in all_months if m[:4] >= str(START_YEAR))

    if len(all_months) >= 12:
        gold_ratio = 1.0 - eq_ratio - debt_ratio
        mnav = 100.0; mpeak = 100.0; max_dd = 0.0
        for month in all_months:
            yr = int(month[:4])
            eq_mret  = _avg_monthly(eq_funds,   month)
            dt_mret  = _avg_monthly(dt_funds,   month)
            gld_mret = _avg_monthly(gold_funds, month)
            if eq_mret  is None: eq_mret  = EQ_FALLBACK.get(yr, 10.0) / 12
            if dt_mret  is None: dt_mret  = DEBT_FALLBACK.get(yr, 7.0) / 12
            if gld_mret is None: gld_mret = GOLD_FALLBACK.get(yr, 10.0) / 12
            port_mret = eq_ratio * eq_mret + gold_ratio * gld_mret + debt_ratio * dt_mret
            mnav *= (1 + port_mret / 100)
            if mnav > mpeak: mpeak = mnav
            dd = (mnav - mpeak) / mpeak
            if dd < max_dd: max_dd = dd
    else:
        # Fallback to annual if not enough monthly data
        navs  = [100.0] + [r["port_nav"] for r in rows]
        peak  = 100.0; max_dd = 0.0
        for v in navs:
            if v > peak: peak = v
            dd = (v - peak) / peak
            if dd < max_dd: max_dd = dd

    ann_std  = float(np.std(rets)) / 100
    sharpe   = round(((port_cagr / 100) - RISK_FREE_RATE) / ann_std, 2) if ann_std > 0 else 0
    win_rate = round(sum(1 for r in rets if r > 0) / n * 100, 1)

    return {
        "cagr":       port_cagr,
        "max_dd":     round(max_dd, 4),
        "sharpe":     sharpe,
        "win_rate":   win_rate,
        "n_years":    n,
        "start_year": START_YEAR,
        "final_nav":  round(nav, 2),
        "bt_rows":    rows,
    }


# ─────────────────────────────────────────────
# STEP 10 — VIX DATA (vix_data.json)
# ─────────────────────────────────────────────

def generate_vix_data():
    """
    256-day rolling Nifty/VIX percentile rank.
    Matches schema: {dates[], percentile[], nifty[], ratio[], generated}
    """
    print("📊 Generating vix_data.json...")
    try:
        nifty = yf.Ticker("^NSEI").history(period="2y")["Close"].dropna()
        vix   = yf.Ticker("^INDIAVIX").history(period="2y")["Close"].dropna()

        # Align on common dates
        df = pd.DataFrame({"nifty": nifty, "vix": vix}).dropna()
        df.index = pd.to_datetime(df.index).tz_localize(None)

        # Nifty/VIX ratio
        df["ratio"] = df["nifty"] / df["vix"]

        # 256-day rolling percentile of ratio
        window = 256

        def rolling_pct(series, w):
            result = []
            for i in range(len(series)):
                start = max(0, i - w + 1)
                window_vals = series.iloc[start:i+1]
                current = series.iloc[i]
                pct = round((window_vals < current).sum() / len(window_vals) * 100, 1)
                result.append(pct)
            return result

        df["percentile"] = rolling_pct(df["ratio"], window)

        out = {
            "dates":      [d.strftime("%Y-%m-%d") for d in df.index],
            "percentile": [round(float(v), 1) for v in df["percentile"]],
            "nifty":      [round(float(v), 2) for v in df["nifty"]],
            "ratio":      [round(float(v), 2) for v in df["ratio"]],
            "generated":  date.today().isoformat(),
        }
        print(f"   ✅ VIX data: {len(out['dates'])} rows, latest percentile: {out['percentile'][-1]}")
        return out

    except Exception as e:
        print(f"   ❌ VIX data failed: {e}")
        return None


# ─────────────────────────────────────────────
# STEP 11 — BREADTH DATA (breadth_data.json)
# ─────────────────────────────────────────────

def generate_breadth_data():
    """
    Nifty 500 breadth: count of stocks above 200-day SMA.
    Matches schema: {dates[], above[], total, generated}
    """
    print("📊 Generating breadth_data.json...")

    # Nifty 500 constituent tickers (sample — top 121 liquid)
    # In production expand this list for full 500
    NIFTY500_TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
        "INFY.NS", "SBIN.NS", "LICI.NS", "HINDUNILVR.NS", "ITC.NS",
        "LT.NS", "KOTAKBANK.NS", "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS",
        "AXISBANK.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "NTPC.NS", "ONGC.NS",
        "TITAN.NS", "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "WIPRO.NS",
        "ULTRACEMCO.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS",
        "DRREDDY.NS", "GRASIM.NS", "BRITANNIA.NS", "CIPLA.NS", "DIVISLAB.NS",
        "EICHERMOT.NS", "COALINDIA.NS", "APOLLOHOSP.NS", "SHRIRAMFIN.NS", "TATACONSUM.NS",
        "TATASTEEL.NS", "HINDALCO.NS", "VEDL.NS", "BPCL.NS", "HEROMOTOCO.NS",
        "BAJAJ-AUTO.NS", "M&M.NS", "TATAPOWER.NS", "INDUSINDBK.NS", "SBILIFE.NS",
        "HDFCLIFE.NS", "AMBUJACEM.NS", "SIEMENS.NS", "DABUR.NS", "MARICO.NS",
        "MUTHOOTFIN.NS", "PIIND.NS", "GODREJCP.NS", "BERGEPAINT.NS", "HAVELLS.NS",
        "CHOLAFIN.NS", "TORNTPHARM.NS", "AUROPHARMA.NS", "LALPATHLAB.NS", "METROPOLIS.NS",
        "MCDOWELL-N.NS", "UBL.NS", "JUBLFOOD.NS", "NAUKRI.NS", "INDIGO.NS",
        "IRCTC.NS", "DMART.NS", "PAGEIND.NS", "POLYCAB.NS", "DIXON.NS",
        "ASTRAL.NS", "TRENT.NS", "ZOMATO.NS", "PAYTM.NS", "NYKAA.NS",
        "POLICYBZR.NS", "DELHIVERY.NS", "EASEMYTRIP.NS", "CARTRADE.NS", "IXIGO.NS",
        "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "IDFCFIRSTB.NS",
        "BANDHANBNK.NS", "FEDERALBNK.NS", "KARURVYSYA.NS", "RBLBANK.NS", "DCBBANK.NS",
        "IOC.NS", "HINDPETRO.NS", "MRPL.NS", "GAIL.NS", "IGL.NS",
        "MGL.NS", "PETRONET.NS", "GSPL.NS", "TATACHEM.NS", "COROMANDEL.NS",
        "GNFC.NS", "DEEPAKNTR.NS", "AARTIIND.NS", "ALKYLAMINE.NS", "VINATIORGA.NS",
        "APLAPOLLO.NS", "JKCEMENT.NS", "SHREECEM.NS", "RAMCOCEM.NS", "HEIDELBERG.NS",
        "ABBOTINDIA.NS", "ALKEM.NS", "BIOCON.NS", "CADILAHC.NS", "GLENMARK.NS",
        "GRANULES.NS", "IPCALAB.NS", "LAURUSLABS.NS", "NATCOPHARM.NS", "SYNGENE.NS",
    ]
    total = len(NIFTY500_TICKERS)

    try:
        print(f"   Downloading {total} tickers (this takes ~2 min)...")
        data = yf.download(
            NIFTY500_TICKERS,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=120,
        )["Close"]

        # Drop tickers with too much missing data
        data = data.dropna(axis=1, thresh=150)
        actual_total = data.shape[1]

        # Compute 200-day SMA per stock
        sma200 = data.rolling(200, min_periods=100).mean()
        above_sma = (data > sma200).astype(int)

        # Daily count above 200 SMA
        daily_count = above_sma.sum(axis=1).dropna()
        daily_count = daily_count[daily_count.index >= pd.Timestamp(date.today() - timedelta(days=365))]

        out = {
            "dates":     [d.strftime("%Y-%m-%d") for d in daily_count.index],
            "above":     [int(v) for v in daily_count.values],
            "total":     actual_total,
            "generated": date.today().isoformat(),
        }
        print(f"   ✅ Breadth: {len(out['dates'])} days, latest: {out['above'][-1]}/{actual_total}")
        return out

    except Exception as e:
        print(f"   ❌ Breadth failed: {e}")
        return None


# ─────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def main():
    today_str = date.today().strftime("%d %b %Y")
    time_str  = datetime.now().strftime("%H:%M IST")

    print(f"\n{'='*55}")
    print(f"  MintingM Data Engine — {today_str} {time_str}")
    print(f"{'='*55}\n")

    # ── 1. AMFI current NAV ──────────────────────────────
    amfi_lookup = fetch_amfi_nav()

    # ── 1b. AUM data (monthly from AMFI) ─────────────────
    aum_map = fetch_aum()

    # ── 2. Build full universe — all valid regular-plan funds ──
    print("\n🔄 Scanning AMFI for all valid regular-plan growth funds...")

    import re as _re

    def get_subcat(amfi_cat):
        m = _re.search(r'- (.+?)(?:\))', amfi_cat)
        return m.group(1).lower() if m else amfi_cat.lower()

    def get_asset_type(amfi_cat):
        subcat = get_subcat(amfi_cat)
        if any(k in subcat for k in INTL_SUBCATS):    return "International"
        if any(k in subcat for k in GOLD_SUBCATS):    return "Gold"
        if any(k in subcat for k in HYBRID_SUBCATS):  return "Hybrid"
        if any(k in subcat for k in EQUITY_SUBCATS):  return "Equity"
        if any(k in subcat for k in DEBT_SUBCATS):    return "Debt"
        return "Other"

    def get_clean_cat(amfi_cat):
        m = _re.search(r'- (.+?)(?:\))', amfi_cat)
        if m:
            raw = m.group(1)
            for prefix in ["Equity Scheme - ", "Debt Scheme - ", "Hybrid Scheme - ",
                           "Other Scheme - ", "Solution Oriented Scheme - "]:
                if raw.startswith(prefix):
                    return raw[len(prefix):]
            return raw
        return amfi_cat

    def is_valid_fund(name, amfi_cat):
        """
        Return True only for Regular Plan Growth funds.
        Universal rule: 'direct' anywhere in the name = direct plan.
        No legitimate regular plan fund contains the word 'direct'.
        """
        n = name.lower()
        # Universal direct plan rejection — catches ALL variants:
        # "- Direct", "(Direct)", "Direct Plan", "Direct-Growth", etc.
        if "direct" in n:
            return False
        # IDCW / dividend payouts
        if "idcw" in n or " dividend" in n:
            return False
        # Junk fund types
        if any(k in n for k in EXCLUDE_KEYWORDS):
            return False
        if "close ended" in amfi_cat.lower():
            return False
        return True

    def is_nav_fresh(nav_date_str):
        try:
            d = datetime.strptime(nav_date_str, "%d-%b-%Y").date()
            return (date.today() - d).days <= 7 and d.year >= 2024
        except:
            return False

    # Scan every fund in AMFI
    candidates = []
    for code, info in amfi_lookup.items():
        if not is_nav_fresh(info["nav_date"]):
            continue
        if not is_valid_fund(info["name"], info["amfi_cat"]):
            continue
        asset_type = get_asset_type(info["amfi_cat"])
        if asset_type == "Other":
            continue
        candidates.append({
            "code":       code,
            "name":       info["name"],
            "cat":        get_clean_cat(info["amfi_cat"]),
            "type":       asset_type,
            "nav_latest": info["nav_latest"],
            "nav_date":   info["nav_date"],
        })

    from collections import Counter as _Counter

    # Sort by AMFI code (lower = older = more established fund)
    candidates.sort(key=lambda x: x["code"])

    # Apply per-type cap to keep GitHub Actions runtime under 35 minutes
    caps = {"Equity": MAX_FUNDS_EQUITY, "Debt": MAX_FUNDS_DEBT,
            "Gold": MAX_FUNDS_GOLD, "International": MAX_FUNDS_INTERNATIONAL,
            "Hybrid": MAX_FUNDS_HYBRID}
    type_buckets = {}
    for c in candidates:
        type_buckets.setdefault(c["type"], []).append(c)

    capped = []
    for t, cap in caps.items():
        bucket = type_buckets.get(t, [])
        capped.extend(bucket[:cap])
        print(f"   {t}: {len(bucket)} valid → top {min(len(bucket), cap)}")

    candidates = capped
    print(f"\n   Total: {len(candidates)} funds | Est. time: ~{len(candidates)*2//60} min")

    # ── 3. Fetch history + compute full MintingM scores ──────
    funds_raw = []
    funds_no_data = []   # valid AMFI funds that mfapi couldn't serve — added at end with null returns
    fund_id_counter = 1000
    skipped_history = 0

    # Shared session for connection reuse
    mfapi_session = requests.Session()
    mfapi_session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MintingM/1.0)"})

    # Circuit breaker
    consecutive_timeouts = 0
    MAX_CONSECUTIVE_TIMEOUTS = 10

    for idx, fund in enumerate(candidates, 1):
        code       = fund["code"]
        fund_name  = fund["name"]
        nav_date   = fund["nav_date"]
        asset_type = fund["type"]
        cat        = fund["cat"]

        if idx % 100 == 0:
            print(f"   ... {idx}/{len(candidates)} processed, "
                  f"{len(funds_raw)} scored so far")

        hist = fetch_history(code, nav_date, min_years=0, _session=mfapi_session)

        if hist is None:
            consecutive_timeouts += 1
            if consecutive_timeouts >= MAX_CONSECUTIVE_TIMEOUTS:
                print(f"   ⚠ {consecutive_timeouts} consecutive failures — pausing 10s for mfapi recovery...")
                time.sleep(10)
                consecutive_timeouts = 0
            skipped_history += 1
            # Add to no-data list — will appear at screener bottom with null returns
            funds_no_data.append({
                "id":          fund_id_counter,
                "code":        code,
                "name":        fund_name,
                "cat":         cat,
                "type":        asset_type,
                "nav_latest":  fund["nav_latest"],
                "nav_date":    nav_date,
                "aum":         aum_map.get(code),
                "live":        False,   # no mfapi data — excluded from portfolio/scoring
                "new_fund":    True,
                "data_from":   None,
                "monthly_nav": {},
                "r1m": None, "r3m": None,
                "r1": None, "r3": None, "r5": None, "r7": None, "r10": None,
                "sharpe": None, "std_dev": None, "max_dd": None,
                "sortino": None, "calmar": None, "win_rate": None,
                "raw_score": 0, "sf": 0, "df": 0, "fp": 0,
                "score": 0,
                "momentum": False, "mean_rev": False, "international": False,
                "_annual_rets": {}, "_has_5y": False,
            })
            fund_id_counter += 1
            time.sleep(0.1)
            continue

        consecutive_timeouts = 0  # reset on success
        time.sleep(0.1)

        # Compute how much history is available
        years_available = (hist["date"].iloc[-1] - hist["date"].iloc[0]).days / 365.25

        # compute_metrics handles partial history gracefully (returns None for missing periods)
        metrics = compute_metrics(hist)
        if metrics is None:
            fund_id_counter += 1
            continue

        score_info = compute_raw_score(metrics, asset_type)

        # Extract end-of-month NAVs (last 24 months for display + full history for MaxDD)
        monthly_nav = {}
        monthly_rets_full = {}  # full monthly return history for portfolio MaxDD
        try:
            today_ts = pd.Timestamp(date.today())
            # Last 24 months for display
            for m in range(1, 25):
                target = today_ts - pd.DateOffset(months=m)
                month_end = target.replace(day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                sub = hist[hist["date"] <= month_end]
                if not sub.empty:
                    month_key = month_end.strftime("%Y-%m")
                    monthly_nav[month_key] = round(float(sub["nav"].iloc[-1]), 4)
            # Full monthly returns history (for accurate MaxDD in backtest)
            hist_sorted = hist.sort_values("date").copy()
            hist_sorted["month"] = hist_sorted["date"].dt.to_period("M")
            monthly_last = hist_sorted.groupby("month")["nav"].last()
            monthly_ret = monthly_last.pct_change().dropna()
            for period, ret in monthly_ret.items():
                if not (ret != ret):  # skip NaN
                    monthly_rets_full[str(period)] = round(float(ret) * 100, 4)
        except Exception:
            pass

        funds_raw.append({
            "id":          fund_id_counter,
            "code":        code,
            "name":        fund_name,
            "cat":         cat,
            "type":        asset_type,
            "nav_latest":  fund["nav_latest"],
            "nav_date":    nav_date,
            "aum":         aum_map.get(code),
            "live":        True,
            "data_from":   int(hist["date"].iloc[0].year),
            "new_fund":    years_available < 3,
            "monthly_nav": monthly_nav,            # last 24 months end-of-month NAV
            "_monthly_rets": monthly_rets_full,    # full monthly returns for MaxDD
            "r1m":         metrics["r1m"],
            "r3m":         metrics["r3m"],
            "r1":          metrics["r1"],
            "r3":          metrics["r3"],
            "r5":          metrics["r5"],
            "r7":          metrics["r7"],
            "r10":         metrics["r10"],
            "sharpe":      metrics["sharpe"],
            "std_dev":     metrics["std_dev"],
            "max_dd":      metrics["max_dd"],
            "sortino":     metrics["sortino"],
            "calmar":      metrics["calmar"],
            "win_rate":    metrics["win_rate"],
            "raw_score":   score_info["raw_score"],
            "sf":          score_info["sf"],
            "df":          score_info["df"],
            "fp":          score_info["fp"],
            "_annual_rets":metrics.get("_annual_rets", {}),
            "_has_5y":     years_available >= 5,
            "momentum":    False,
            "mean_rev":    False,
            "international": False,
        })
        fund_id_counter += 1

    final_counts = _Counter(f["type"] for f in funds_raw)
    print(f"\n   ✅ Scored {len(funds_raw)} funds "
          f"({final_counts['Equity']} Equity, "
          f"{final_counts['Debt']} Debt, "
          f"{final_counts['Gold']} Gold, "
          f"{final_counts['International']} International)")
    print(f"   No mfapi data: {len(funds_no_data)} funds — added at screener bottom with null returns")

    # Append no-data funds at the end (appear last in screener)
    funds_raw.extend(funds_no_data)

    # ── 4. Normalize scores ──────────────────────────────
    funds_raw = normalize_scores(funds_raw)

    # ── 5. SG Ratio ──────────────────────────────────────
    current_sg, sg_history = get_sg_ratio_and_history()
    gold_active = current_sg > GOLD_THRESHOLD

    # ── 6. Nifty benchmarks annual + monthly ─────────────────
    print("\n📈 Fetching all benchmark returns...")
    benchmark_annual = get_benchmark_annual()
    nifty_annual = benchmark_annual.get("nifty50", {})

    # Monthly benchmark data for sub-year comparison (last 24 months)
    print("   Building monthly benchmark snapshots (last 24 months)...")
    benchmark_monthly = {}
    try:
        for key, (ticker, label, _) in BENCHMARK_TICKERS.items():
            hist_idx = yf.Ticker(ticker).history(period="3y")["Close"].dropna()
            try:
                hist_idx.index = hist_idx.index.tz_localize(None)
            except TypeError:
                hist_idx.index = hist_idx.index.tz_convert(None)
            monthly = {}
            today_ts = pd.Timestamp(date.today())
            for m in range(1, 25):  # last 24 months
                target = today_ts - pd.DateOffset(months=m)
                month_end = target.replace(day=1) + pd.DateOffset(months=1) - pd.Timedelta(days=1)
                sub = hist_idx[hist_idx.index <= month_end]
                if not sub.empty:
                    month_key = month_end.strftime("%Y-%m")
                    monthly[month_key] = round(float(sub.iloc[-1]), 2)
            benchmark_monthly[key] = monthly
            print(f"   ✅ {label}: {len(monthly)} monthly points")
    except Exception as e:
        print(f"   ⚠ Monthly benchmark failed: {e}")
    print("\n📈 Fetching Nifty benchmark returns for momentum signal...")
    try:
        nifty_hist = yf.Ticker("^NSEI").history(start="2010-01-01")["Close"].dropna()
        try:
            nifty_hist.index = nifty_hist.index.tz_localize(None)
        except TypeError:
            nifty_hist.index = nifty_hist.index.tz_convert(None)
        today_ts = pd.Timestamp(date.today())

        def nifty_cagr(years):
            end_sub   = nifty_hist[nifty_hist.index <= today_ts]
            start_tgt = today_ts - pd.DateOffset(years=years)
            start_sub = nifty_hist[nifty_hist.index <= start_tgt]
            if end_sub.empty or start_sub.empty: return None
            e, s = float(end_sub.iloc[-1]), float(start_sub.iloc[-1])
            act_y = (end_sub.index[-1] - start_sub.index[-1]).days / 365.25
            return round(((e/s)**(1/act_y)-1)*100, 2) if act_y > 0.5 else None

        nifty_1y = nifty_cagr(1)
        nifty_3y = nifty_cagr(3)
        nifty_5y = nifty_cagr(5)
        print(f"   Nifty 1Y: {nifty_1y}% | 3Y: {nifty_3y}% | 5Y: {nifty_5y}%")
    except Exception as e:
        print(f"   ⚠ Nifty benchmark fetch failed: {e} — using fallback")
        nifty_1y, nifty_3y, nifty_5y = 8.8, 14.0, 12.5  # conservative fallback

    # ── 7. Tag funds with momentum / mean_rev / international ─
    print("🏷  Tagging funds (momentum / mean reversion / international)...")
    funds_raw = tag_funds(funds_raw, nifty_1y, nifty_3y, nifty_5y)
    momentum_count  = sum(1 for f in funds_raw if f.get("momentum"))
    mean_rev_count  = sum(1 for f in funds_raw if f.get("mean_rev"))
    intl_count      = sum(1 for f in funds_raw if f.get("international"))
    print(f"   Momentum: {momentum_count} | Mean reversion: {mean_rev_count} | International: {intl_count}")

    # ── 8. Portfolio selection ────────────────────────────
    print("\n🎯 Selecting portfolio funds (5Y+ history only)...")
    # For portfolio selection: equity/debt need 5Y+ history for reliability
    # Gold funds: include ALL scored Gold funds regardless of history
    # (Gold ETFs were launched later — 5Y filter would exclude too many)
    portfolio_eligible = [
        f for f in funds_raw
        if f.get("live") and (f.get("_has_5y") or f.get("type") == "Gold")
    ]
    all_live = [f for f in funds_raw if f.get("live")]
    print(f"   Eligible for portfolio: {len(portfolio_eligible)} / {len(all_live)} funds")
    portfolio_selection = {}

    for profile_key, prof in PROFILES.items():
        picks = select_portfolio_funds(
            portfolio_eligible, profile_key,
            nifty_1y, nifty_3y, nifty_5y
        )
        portfolio_selection[profile_key] = {
            "profile": profile_key,
            "eq":      prof["eq"],
            "gold":    prof["gold"],
            "debt":    prof["debt"],
            "funds":   picks,
        }
        names = [f["name"].split("-")[0].strip()[:22] for f in picks]
        tags  = ["M" if f.get("momentum") else ("MR" if f.get("mean_rev") else "—") for f in picks]
        print(f"   {profile_key} ({prof['label']}): {list(zip(names, tags))}")

    # ── 9. Backtesting ────────────────────────────────────
    print("\n🔢 Running backtests...")
    backtest = {}
    for profile_key, prof in PROFILES.items():
        picks    = portfolio_selection[profile_key]["funds"]
        pick_ids = {f["id"] for f in picks}

        eq_funds  = [f for f in all_live if f["type"] == "Equity" and f["id"] in pick_ids]
        dt_funds  = [f for f in all_live if f["type"] == "Debt"   and f["id"] in pick_ids]
        gld_funds = [f for f in all_live if f["type"] == "Gold"   and f["id"] in pick_ids]

        bt = run_backtest(
            profile_key, prof["eq"], prof["debt"],
            sg_history, eq_funds, dt_funds, gld_funds
        )
        backtest[profile_key] = bt
        if bt:
            print(f"   {profile_key}: CAGR {bt['cagr']}% | Sharpe {bt['sharpe']} | MaxDD {bt['max_dd']}")

    # ── 8. Clean up internal fields and sanitize numpy types ─
    def sanitize(obj):
        """
        Recursively convert all numpy/pandas scalar types to native Python.
        Fixes: bool_ → bool, int64 → int, float64 → float, nan → None
        """
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, float):
            return None if math.isnan(obj) or math.isinf(obj) else obj
        return obj

    def clean_fund(f):
        """Remove internal _ fields except _annual_rets (needed for date range backtest).
        Sanitize all numpy types."""
        out = {}
        for k, v in f.items():
            if k == "_annual_rets":
                out["annual_rets"] = v   # keep but rename (remove underscore)
            elif not k.startswith("_"):
                out[k] = v
        return sanitize(out)

    funds_clean = [clean_fund(f) for f in funds_raw]
    live_count  = sum(1 for f in funds_clean if f.get("live"))

    # ── 9. Build portfolio_funds in exact format ingestDataJSON expects ──
    # HTML matches by first 3 words of fund name, so keys must reflect that
    # Structure: list of {profile, name, r1..r7, sharpe, std_dev, max_dd, nav_date}
    portfolio_funds_list = []
    for profile_key, prof_data in portfolio_selection.items():
        for f in prof_data["funds"]:
            match = next((lf for lf in all_live if lf["id"] == f["id"]), None)
            if match:
                portfolio_funds_list.append({
                    "profile":  profile_key,
                    "name":     match["name"],
                    "type":     match["type"],
                    "cat":      match["cat"],
                    "aum":      match.get("aum"),
                    "r1m":      match.get("r1m"),
                    "r3m":      match.get("r3m"),
                    "r1":       match.get("r1"),
                    "r3":       match.get("r3"),
                    "r5":       match.get("r5"),
                    "r7":       match.get("r7"),
                    "r10":      match.get("r10"),
                    "sharpe":   match.get("sharpe"),
                    "std_dev":  match.get("std_dev"),
                    "max_dd":   match.get("max_dd"),
                    "nav_date": match.get("nav_date"),
                    "score":    match.get("score"),
                })

    # ── 10. Build amfi_all — compact list of ALL valid regular funds ──
    # Used by Review PF typeahead to find ANY fund a client holds
    # Much larger than scored universe — includes funds with <3Y history
    INTL_SUBCATS_SET = set(INTL_SUBCATS)

    def get_asset_type_simple(amfi_cat):
        sub = amfi_cat.lower()
        if any(k in sub for k in GOLD_SUBCATS):   return "Gold"
        if any(k in sub for k in INTL_SUBCATS_SET):return "International"
        if any(k in sub for k in EQUITY_SUBCATS):  return "Equity"
        if any(k in sub for k in DEBT_SUBCATS):    return "Debt"
        return "Other"

    amfi_all = []
    for code, info in amfi_lookup.items():
        n = info["name"].lower()
        if "direct" in n or "idcw" in n or " dividend" in n:
            continue
        if any(k in n for k in EXCLUDE_KEYWORDS):
            continue
        t = get_asset_type_simple(info["amfi_cat"])
        if t == "Other":
            continue
        amfi_all.append({
            "code": code,
            "name": info["name"],
            "cat":  info["amfi_cat"].split("- ")[-1].rstrip(")") if "- " in info["amfi_cat"] else info["amfi_cat"],
            "type": t,
        })
    print(f"   amfi_all: {len(amfi_all)} funds for Review PF search")

    # ── 11. Assemble data.json ────────────────────────────────
    data_out = sanitize({
        "generated_at":   datetime.now().isoformat(),
        "generated_date": today_str,
        "generated_time": time_str,
        "data_source":    "AMFI India direct + mfapi.in — official NAV data",
        "live_funds":     live_count,
        "total_funds":    len(funds_clean),
        "gold_threshold": GOLD_THRESHOLD,
        "sg_ratio":       current_sg,
        "gold_active":    gold_active,
        "sg_history":     sg_history,
        "nifty_annual":     nifty_annual,
        "benchmark_annual": benchmark_annual,    # annual returns: all 3 benchmarks
        "benchmark_monthly": benchmark_monthly,  # monthly snapshots: last 24 months
        "portfolio_selection": portfolio_selection,
        "portfolio_funds": portfolio_funds_list,
        "backtest":        backtest,
        "screener":        funds_clean,
        "funds":           funds_clean,
        "amfi_all":        amfi_all,
    })

    with open("data.json", "w") as f:
        # Use allow_nan=False to catch any remaining NaN values
        # sanitize() should have converted them all to None already
        try:
            json.dump(data_out, f, indent=2, allow_nan=False)
        except ValueError:
            # If NaN still present, do a string replacement as last resort
            import io
            buf = io.StringIO()
            json.dump(data_out, buf, indent=2, allow_nan=True)
            clean = buf.getvalue().replace(': NaN', ': null').replace(':NaN', ':null')
            f.write(clean)
    print(f"\n✅ data.json written ({live_count} live funds, {len(funds_clean)} total)")

    # ── 12. VIX data ──────────────────────────────────────
    vix_out = generate_vix_data()
    if vix_out:
        with open("vix_data.json", "w") as f:
            json.dump(sanitize(vix_out), f, indent=2)
        print("✅ vix_data.json written")

    # Breadth data generation removed — tab removed from UI, saves ~5 min

    print(f"\n🎉 All done — {today_str} {time_str}")


if __name__ == "__main__":
    main()
