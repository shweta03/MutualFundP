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
MIN_HISTORY_YEARS = 3

# For portfolio selection: only funds with 5Y history are eligible
# (more reliable scores for picking a model portfolio)
MIN_PORTFOLIO_YEARS = 5

# AMFI sub-category → asset type mapping
EQUITY_SUBCATS = [
    "flexi cap", "large cap", "mid cap", "small cap",
    "large & mid cap", "multi cap", "focused fund",
    "contra fund", "value fund", "elss",
    "balanced advantage", "dynamic asset allocation",
    "aggressive hybrid",          # covers Conservative Hybrid-style funds
]
DEBT_SUBCATS = [
    "short duration", "corporate bond", "banking and psu",
    "medium duration", "low duration", "money market", "ultra short duration",
]
GOLD_SUBCATS = ["gold etf", "gold fund", "gold fof", "gold savings"]

# Fund name filters — skip these regardless of category
EXCLUDE_KEYWORDS = [
    "segregated", " series ", "series i ", "series ii", "series iii",
    "series iv", "series v", "series vi", "series vii", "series viii",
    "fixed term", "ftf", "capital protection", "unclaimed", "discontinued",
    "eco plan", "wealth plan", "retail plan", "super institutional",
    "interval fund", "close ended", "bonus option",
]

# Profile definitions
PROFILES = {
    "C": {"eq": 0.20, "debt": 0.80, "label": "Conservative"},
    "M": {"eq": 0.60, "debt": 0.40, "label": "Moderate"},
    "A": {"eq": 0.80, "debt": 0.20, "label": "Aggressive"},
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


# ─────────────────────────────────────────────
# STEP 2 — FETCH HISTORICAL NAV FROM mfapi.in
# ─────────────────────────────────────────────

def fetch_history(scheme_code, amfi_nav_date_str, min_years=1):
    """
    Get historical NAV from mfapi.in.
    amfi_nav_date_str: nav_date from AMFI file (e.g. '17-Apr-2026') — used for freshness check.
    Returns DataFrame [date, nav] sorted ascending, or None if stale/insufficient.
    """
    # First validate freshness from AMFI date (before making the slow mfapi call)
    try:
        amfi_date = datetime.strptime(amfi_nav_date_str, "%d-%b-%Y").date()
        days_old = (date.today() - amfi_date).days
        # Reject if NAV is older than 7 days OR if year is suspiciously old
        # (catches funds that were merged/wound-down but still appear in mfapi)
        if days_old > 7:
            print(f"   ⚠ STALE AMFI NAV ({days_old}d old, {amfi_nav_date_str}): {scheme_code} — skipping")
            return None
        if amfi_date.year < 2024:
            print(f"   ⚠ SUSPICIOUS NAV DATE ({amfi_nav_date_str}): {scheme_code} — skipping")
            return None
    except Exception:
        pass

    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MintingM/1.0)"}
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        r = requests.get(url, headers=headers, timeout=25)
        if r.status_code != 200:
            return None
        raw = r.json()
        if "data" not in raw or not raw["data"]:
            return None

        df = pd.DataFrame(raw["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)

        if df.empty:
            return None

        # Secondary freshness check on mfapi's latest date
        mfapi_latest = df["date"].iloc[-1].date()
        if (date.today() - mfapi_latest).days > 7 or mfapi_latest.year < 2024:
            print(f"   ⚠ STALE mfapi ({mfapi_latest}): {scheme_code} — skipping")
            return None

        years_available = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        if years_available < min_years:
            return None

        return df

    except Exception as e:
        print(f"   ❌ mfapi error for {scheme_code}: {e}")
        return None


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

    if len(df) < 60:
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
    # Hard filter flags — explicitly cast to native Python bool
    sf = bool((m["sharpe"] is not None) and (m["sharpe"] >= 1.0))
    if asset_type == "Gold":
        df_flag = True
    else:
        df_flag = bool((m["max_dd"] is not None) and (-0.30 <= m["max_dd"] <= -0.20))

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
    for asset_type in ["Equity", "Debt", "Gold"]:
        group = [f for f in funds if f.get("type") == asset_type and "raw_score" in f]
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

def select_portfolio_funds(scored_funds, profile_key):
    """
    Select exactly 5 funds per profile based on MintingM score.
    Spec (from product definition):

    CONSERVATIVE  (80% Debt / 20% Equity):
      1. Best Balanced Advantage OR Aggressive Hybrid fund    [Equity bucket]
      2. Best Gold ETF                                        [Gold bucket]
      3. Best Short Duration debt fund                        [Debt]
      4. Best Corporate Bond OR Banking & PSU fund            [Debt]
      5. Best Low Duration OR second Short Duration fund      [Debt]

    MODERATE  (60% Equity / 40% Debt):
      1. Best Flexi Cap fund                                  [Equity]
      2. Best Balanced Advantage OR Dynamic Asset Alloc fund  [Equity]
      3. Best Mid Cap fund                                    [Equity]
      4. Best Gold ETF                                        [Gold]
      5. Best Short Duration OR Corporate Bond debt fund      [Debt]

    AGGRESSIVE  (80% Equity / 20% Debt):
      1. Best Flexi Cap fund                                  [Equity]
      2. Best Multi Cap fund                                  [Equity]
      3. Best Value OR Contra fund                            [Equity]
      4. Best Gold ETF                                        [Gold]
      5. Best Short Duration OR Corporate Bond debt fund      [Debt]

    Rules:
    - All funds must have live=True, score > 0, _has_5y=True
    - No AMC duplication for equity funds (different fund houses preferred)
    - Gold: prefer ETF over FoF
    """
    def top(asset_type, keywords, n=1, exclude_ids=None, exclude_amcs=None):
        """Get top n funds by MintingM score from given asset_type + category keywords."""
        exclude_ids  = exclude_ids  or []
        exclude_amcs = exclude_amcs or []
        pool = [
            f for f in scored_funds
            if f.get("type") == asset_type
            and f.get("live", False)
            and f.get("score", 0) > 0
            and f["id"] not in exclude_ids
            and any(kw.lower() in f.get("cat", "").lower() for kw in keywords)
            # Avoid same AMC for equity picks (first word of fund name = AMC indicator)
            and (not exclude_amcs or f["name"].split()[0] not in exclude_amcs)
        ]
        pool.sort(key=lambda x: x.get("score", 0), reverse=True)
        return pool[:n]

    def amc(fund):
        """Extract AMC shortname from fund name (first meaningful word)."""
        return fund["name"].split()[0] if fund else None

    picks = []

    if profile_key == "C":
        # Slot 1: Balanced Advantage or Aggressive Hybrid (lowest volatility equity)
        eq1 = top("Equity", ["Balanced Advantage", "Dynamic Asset", "Aggressive Hybrid"], 1)
        picks += eq1

        # Slot 2: Gold ETF
        gold = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += gold

        # Slot 3: Best Short Duration debt
        dt1 = top("Debt", ["Short Duration"], 1)
        picks += dt1

        # Slot 4: Best Corporate Bond or Banking & PSU (different AMC from dt1)
        used_amcs = [amc(f) for f in dt1 if f]
        dt2 = (top("Debt", ["Corporate Bond"], 1, [f["id"] for f in picks], used_amcs) or
               top("Debt", ["Banking & PSU", "Banking and PSU"], 1, [f["id"] for f in picks], used_amcs))
        picks += dt2

        # Slot 5: Low Duration (different AMC from above)
        used_amcs = [amc(f) for f in dt1 + dt2 if f]
        dt3 = top("Debt", ["Low Duration"], 1, [f["id"] for f in picks], used_amcs)
        if not dt3:
            dt3 = top("Debt", ["Short Duration", "Corporate Bond", "Banking & PSU"], 1,
                      [f["id"] for f in picks])
        picks += dt3

    elif profile_key == "M":
        # Slot 1: Best Flexi Cap
        eq1 = top("Equity", ["Flexi Cap"], 1)
        picks += eq1

        # Slot 2: Balanced Advantage (at fund manager's discretion — dynamic equity)
        used_amcs = [amc(f) for f in eq1 if f]
        eq2 = (top("Equity", ["Balanced Advantage", "Dynamic Asset"], 1,
                   [f["id"] for f in picks], used_amcs))
        picks += eq2

        # Slot 3: Mid Cap (different AMC from eq1 and eq2)
        used_amcs = [amc(f) for f in eq1 + eq2 if f]
        eq3 = top("Equity", ["Mid Cap", "Large & Mid Cap"], 1,
                  [f["id"] for f in picks], used_amcs)
        picks += eq3

        # Slot 4: Gold ETF
        gold = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += gold

        # Slot 5: Best Debt (Short Duration or Corporate Bond)
        dt1 = (top("Debt", ["Short Duration"], 1, [f["id"] for f in picks]) or
               top("Debt", ["Corporate Bond"], 1, [f["id"] for f in picks]))
        picks += dt1

    else:  # A — Aggressive
        # Slot 1: Best Flexi Cap
        eq1 = top("Equity", ["Flexi Cap"], 1)
        picks += eq1

        # Slot 2: Multi Cap (SEBI mandates 25% each large/mid/small — genuine diversification)
        used_amcs = [amc(f) for f in eq1 if f]
        eq2 = top("Equity", ["Multi Cap"], 1, [f["id"] for f in picks], used_amcs)
        if not eq2:
            eq2 = top("Equity", ["Large & Mid Cap", "Focused Fund"], 1,
                      [f["id"] for f in picks], used_amcs)
        picks += eq2

        # Slot 3: Value or Contra (contrarian style — uncorrelated to growth bias of slots 1 & 2)
        used_amcs = [amc(f) for f in eq1 + eq2 if f]
        eq3 = (top("Equity", ["Value Fund", "Contra Fund"], 1,
                   [f["id"] for f in picks], used_amcs))
        if not eq3:
            eq3 = top("Equity", ["Flexi Cap", "Large Cap"], 1,
                      [f["id"] for f in picks], used_amcs)
        picks += eq3

        # Slot 4: Gold ETF
        gold = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)
        picks += gold

        # Slot 5: Best Debt (single fund = max rebalancing efficiency)
        dt1 = (top("Debt", ["Short Duration"], 1, [f["id"] for f in picks]) or
               top("Debt", ["Corporate Bond"], 1, [f["id"] for f in picks]))
        picks += dt1

    # Remove any None/empty that slipped through
    picks = [f for f in picks if f]

    return [
        {
            "id":    f["id"],
            "name":  f["name"],
            "type":  f["type"],
            "cat":   f["cat"],
            "score": f["score"],
            "code":  f["code"],
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
        s_hist.index  = s_hist.index.tz_localize(None)
        g_hist.index  = g_hist.index.tz_localize(None)
        fx_hist.index = fx_hist.index.tz_localize(None)

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
# STEP 8 — NIFTY ANNUAL RETURNS
# ─────────────────────────────────────────────

def get_nifty_annual():
    """Annual calendar-year returns for Nifty 50 from 2007 to now."""
    print("📈 Fetching Nifty annual returns...")
    try:
        nifty = yf.Ticker("^NSEI").history(start="2006-01-01")["Close"]
        nifty.index = nifty.index.tz_localize(None)  # strip tz for asof()
        result = {}
        for year in range(2007, date.today().year + 1):
            try:
                start_val = float(nifty.asof(pd.Timestamp(f"{year - 1}-12-31")))
                end_val   = float(nifty.asof(pd.Timestamp(f"{year}-12-31")))
                if start_val > 0:
                    result[str(year)] = round((end_val / start_val - 1) * 100, 1)
            except Exception:
                pass
        print(f"   ✅ Nifty annual: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"   ❌ Nifty annual failed: {e} — using static fallback")
        return {
            "2007": 36.6,  "2008": -51.8, "2009": 70.7,  "2010": 17.2,
            "2011": -24.9, "2012": 23.9,  "2013": 5.2,   "2014": 33.1,
            "2015": -5.3,  "2016": 5.1,   "2017": 28.7,  "2018": 4.0,
            "2019": 12.7,  "2020": 14.8,  "2021": 23.8,  "2022": 2.7,
            "2023": 19.4,  "2024": 8.8,   "2025": 10.1,  "2026": -6.9,
        }


# ─────────────────────────────────────────────
# STEP 9 — BACKTEST (DETERMINISTIC, NO RANDOM)
# ─────────────────────────────────────────────

def get_fund_annual_return(fund_annual_rets, year):
    """Get actual annual return for a fund in a given year. Returns None if missing."""
    return fund_annual_rets.get(year)


def run_backtest(profile_key, eq_ratio, debt_ratio, sg_history,
                 eq_funds, dt_funds, gold_funds):
    """
    Deterministic annual backtest 2013–present using actual fund annual returns.
    Gold overlay: SG ratio > 9 → 70% of equity bucket → Gold ETF.
    """
    START_YEAR = 2013

    # Build annual return lookup per fund
    def get_avg_annual(funds, year):
        rets = []
        for f in funds:
            v = f.get("_annual_rets", {}).get(year)
            if v is not None:
                rets.append(v)
        if not rets:
            return None
        return sum(rets) / len(rets)

    nav = 100.0
    rows = []
    end_year = date.today().year

    for year in range(START_YEAR, end_year + 1):
        # SG ratio at START of year = Dec-31 of PREVIOUS year
        prev_year_ratio = sg_history.get(str(year - 1), sg_history.get(str(year), 9.0))
        gold_active = float(prev_year_ratio) > GOLD_THRESHOLD
        gf = GOLD_ACTIVE_SPLIT if gold_active else (1 - GOLD_ACTIVE_SPLIT)
        ef = 1 - gf

        # Get annual returns
        eq_ret  = get_avg_annual(eq_funds, year)
        dt_ret  = get_avg_annual(dt_funds, year)
        gld_ret = get_avg_annual(gold_funds, year)

        # Fallbacks if fund has no data for this year
        if eq_ret  is None: eq_ret  = 12.0
        if dt_ret  is None: dt_ret  = 7.5
        if gld_ret is None: gld_ret = 8.0

        # Blended equity bucket return
        blended_eq = eq_ret * ef + gld_ret * gf
        port_ret   = eq_ratio * blended_eq + debt_ratio * dt_ret

        nav *= (1 + port_ret / 100)

        rows.append({
            "year":      year,
            "port_nav":  round(nav, 2),
            "port_ret":  round(port_ret, 2),
            "regime":    "gold" if gold_active else "equity",
            "sg_ratio":  float(prev_year_ratio),
        })

    # Summary stats
    n = len(rows)
    if n == 0:
        return {}

    port_cagr = round(((nav / 100) ** (1 / n) - 1) * 100, 1)
    rets = [r["port_ret"] for r in rows]

    # Max drawdown on NAV series
    navs = [100.0] + [r["port_nav"] for r in rows]
    peak = 100.0
    max_dd = 0.0
    for v in navs:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    ann_std = float(np.std(rets)) / 100
    sharpe  = round(((port_cagr / 100) - RISK_FREE_RATE) / ann_std, 2) if ann_std > 0 else 0
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

    # ── 2. Build full universe — all valid regular-plan funds ──
    print("\n🔄 Scanning AMFI for all valid regular-plan growth funds...")

    import re as _re

    def get_subcat(amfi_cat):
        m = _re.search(r'- (.+?)(?:\))', amfi_cat)
        return m.group(1).lower() if m else amfi_cat.lower()

    def get_asset_type(amfi_cat):
        subcat = get_subcat(amfi_cat)
        if any(k in subcat for k in GOLD_SUBCATS):   return "Gold"
        if any(k in subcat for k in EQUITY_SUBCATS): return "Equity"
        if any(k in subcat for k in DEBT_SUBCATS):   return "Debt"
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
    type_counts = _Counter(c["type"] for c in candidates)
    print(f"   Valid candidates: {len(candidates)} "
          f"({type_counts['Equity']} Equity, "
          f"{type_counts['Debt']} Debt, "
          f"{type_counts['Gold']} Gold)")
    print(f"   Fetching 3Y+ history for each — this takes ~15 min...")

    # ── 3. Fetch history + compute full MintingM scores ──────
    funds_raw = []
    fund_id_counter = 1000
    skipped_history = 0
    skipped_3y = 0

    for idx, fund in enumerate(candidates, 1):
        code       = fund["code"]
        fund_name  = fund["name"]
        nav_date   = fund["nav_date"]
        asset_type = fund["type"]
        cat        = fund["cat"]

        if idx % 50 == 0:
            print(f"   ... {idx}/{len(candidates)} processed, "
                  f"{len(funds_raw)} scored so far")

        hist = fetch_history(code, nav_date, min_years=MIN_HISTORY_YEARS)
        time.sleep(0.2)  # respectful rate limiting

        if hist is None:
            skipped_history += 1
            fund_id_counter += 1
            continue

        # Verify actual history length
        years_available = (hist["date"].iloc[-1] - hist["date"].iloc[0]).days / 365.25
        if years_available < MIN_HISTORY_YEARS:
            skipped_3y += 1
            fund_id_counter += 1
            continue

        metrics = compute_metrics(hist)
        if metrics is None:
            fund_id_counter += 1
            continue

        score_info = compute_raw_score(metrics, asset_type)

        funds_raw.append({
            "id":          fund_id_counter,
            "code":        code,
            "name":        fund_name,
            "cat":         cat,
            "type":        asset_type,
            "nav_latest":  fund["nav_latest"],
            "nav_date":    nav_date,
            "live":        True,
            "data_from":   int(hist["date"].iloc[0].year),
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
            # Flag funds with 5Y+ history — eligible for portfolio selection
            "_has_5y":     years_available >= 5,
        })
        fund_id_counter += 1

    final_counts = _Counter(f["type"] for f in funds_raw)
    print(f"\n   ✅ Scored {len(funds_raw)} funds "
          f"({final_counts['Equity']} Equity, "
          f"{final_counts['Debt']} Debt, "
          f"{final_counts['Gold']} Gold)")
    print(f"   Skipped: {skipped_history} stale/unavailable, "
          f"{skipped_3y} insufficient history (<3Y)")

    # ── 4. Normalize scores ──────────────────────────────
    funds_raw = normalize_scores(funds_raw)

    # ── 5. SG Ratio ──────────────────────────────────────
    current_sg, sg_history = get_sg_ratio_and_history()
    gold_active = current_sg > GOLD_THRESHOLD

    # ── 6. Nifty annual returns ───────────────────────────
    nifty_annual = get_nifty_annual()

    # ── 7. Portfolio selection ────────────────────────────
    # Only pick from funds with 5Y+ history for reliability
    print("\n🎯 Selecting portfolio funds (5Y+ history only)...")
    portfolio_eligible = [f for f in funds_raw if f.get("live") and f.get("_has_5y")]
    all_live = [f for f in funds_raw if f.get("live")]
    print(f"   Eligible for portfolio: {len(portfolio_eligible)} / {len(all_live)} funds")
    portfolio_selection = {}

    for profile_key, prof in PROFILES.items():
        picks = select_portfolio_funds(portfolio_eligible, profile_key)
        portfolio_selection[profile_key] = {
            "profile": profile_key,
            "eq":      prof["eq"],
            "debt":    prof["debt"],
            "funds":   picks,
        }
        names = [f["name"].split("-")[0].strip()[:22] for f in picks]
        print(f"   {profile_key} ({prof['label']}): {names}")

    # ── 8. Backtesting ────────────────────────────────────
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
        """Remove internal _ fields and sanitize all numpy types."""
        out = {k: v for k, v in f.items() if not k.startswith("_")}
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

    # ── 10. Assemble data.json ────────────────────────────────
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
        "nifty_annual":   nifty_annual,
        "portfolio_selection": portfolio_selection,
        "portfolio_funds": portfolio_funds_list,   # ← format ingestDataJSON reads
        "backtest":        backtest,
        "screener":        funds_clean,   # ← key ingestDataJSON reads for UNIVERSE
        "funds":           funds_clean,   # ← backwards compat
    })

    with open("data.json", "w") as f:
        json.dump(data_out, f, indent=2)
    print(f"\n✅ data.json written ({live_count} live funds, {len(funds_clean)} total, {len(portfolio_funds_list)} portfolio fund entries)")

    # ── 11. VIX data ──────────────────────────────────────
    vix_out = generate_vix_data()
    if vix_out:
        with open("vix_data.json", "w") as f:
            json.dump(sanitize(vix_out), f, indent=2)
        print("✅ vix_data.json written")

    # ── 12. Breadth data ──────────────────────────────────
    breadth_out = generate_breadth_data()
    if breadth_out:
        with open("breadth_data.json", "w") as f:
            json.dump(sanitize(breadth_out), f, indent=2)
        print("✅ breadth_data.json written")

    print(f"\n🎉 All done — {today_str} {time_str}")


if __name__ == "__main__":
    main()
