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

RISK_FREE_RATE = 0.065          # 6.5% — 10Y Gsec approximate
GOLD_THRESHOLD = 9.0            # SG Ratio threshold for Gold Active
GOLD_ACTIVE_SPLIT = 0.70        # 70% of equity bucket → Gold when active

# Which AMFI categories map to which asset type
EQUITY_CATS = [
    "flexi cap", "large cap", "mid cap", "small cap",
    "large & mid cap", "multi cap", "focused fund",
    "contra fund", "value fund", "elss"
]
DEBT_CATS = [
    "short duration", "corporate bond", "banking and psu",
    "banking & psu", "medium duration", "medium to long duration",
    "low duration", "money market", "ultra short duration"
]
GOLD_CATS = ["gold etf", "gold fund", "gold fof"]

# Fund universe: AMFI scheme codes to include
# Add/remove codes here to expand or shrink the universe
# These are well-known, liquid, regular-plan growth funds
FUND_UNIVERSE_CODES = [
    # ── EQUITY ──
    120403,  # Kotak Flexi Cap - Regular Growth
    118989,  # HDFC Mid Cap Opportunities - Regular Growth
    122639,  # Parag Parikh Flexi Cap - Regular Growth
    119597,  # SBI Bluechip Fund - Regular Growth
    120505,  # ICICI Pru Bluechip Fund - Regular Growth  (also debt below, code differs)
    118825,  # Mirae Asset Large Cap - Regular Growth
    119364,  # Axis Flexi Cap Fund - Regular Growth
    120465,  # DSP Flexi Cap Fund - Regular Growth
    118778,  # Franklin India Flexi Cap - Regular Growth
    148931,  # Canara Robeco Flexi Cap - Regular Growth
    120594,  # ICICI Pru Midcap Fund - Regular Growth
    119176,  # Nippon India Growth Fund (Mid) - Regular Growth
    120816,  # Axis Midcap Fund - Regular Growth
    119598,  # SBI Magnum Midcap Fund - Regular Growth
    120503,  # Kotak Emerging Equity - Regular Growth (Mid)
    # ── DEBT ──
    118560,  # HDFC Short Term Debt - Regular Growth
    120505,  # ICICI Pru Banking & PSU Debt - Regular Growth
    119533,  # Aditya BSL Corporate Bond - Regular Growth
    119062,  # SBI Short Term Debt - Regular Growth
    118954,  # Nippon India Low Duration - Regular Growth
    119305,  # HDFC Banking & PSU Debt - Regular Growth
    119527,  # Axis Corporate Bond - Regular Growth
    120503,  # Kotak Corporate Bond - Regular Growth
    # ── GOLD ──
    120684,  # Nippon India ETF Gold BeES - Regular Growth
    118701,  # Nippon India Gold Savings Fund - Regular Growth
    119063,  # SBI Gold Fund - Regular Growth
    120082,  # Kotak Gold Fund - Regular Growth
    118548,  # HDFC Gold ETF - Regular Growth
    118547,  # HDFC Gold Fund - Regular Growth
    119527,  # Axis Gold Fund - Regular Growth
]
# De-duplicate
FUND_UNIVERSE_CODES = list(dict.fromkeys(FUND_UNIVERSE_CODES))

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

def fetch_history(scheme_code, min_years=1):
    """
    Get historical NAV from mfapi.in.
    Returns DataFrame with [date, nav] sorted ascending.
    Returns None if data is stale or insufficient.
    """
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        r = requests.get(url, timeout=25)
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

        # Validate freshness: reject if latest NAV is >7 calendar days old
        latest_date = df["date"].iloc[-1].date()
        days_old = (date.today() - latest_date).days
        if days_old > 7:
            print(f"   ⚠ STALE ({days_old}d old): {scheme_code} — skipping")
            return None

        # Need minimum data
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
    """Compute CAGR for given lookback period in years."""
    end_date = df["date"].iloc[-1]
    start_date = end_date - pd.DateOffset(years=years)
    sub = df[df["date"] >= start_date].copy()
    if len(sub) < 30:
        return None
    start_nav = sub["nav"].iloc[0]
    end_nav = sub["nav"].iloc[-1]
    actual_years = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days / 365.25
    if actual_years < 0.5 or start_nav <= 0:
        return None
    return round(((end_nav / start_nav) ** (1 / actual_years) - 1) * 100, 2)


def compute_metrics(df):
    """
    Compute all risk/return metrics from daily NAV.
    Formula exactly matching the Formula Guide in index.html.
    """
    df = df.copy()
    df["ret"] = df["nav"].pct_change()
    df = df.dropna(subset=["ret"])

    if len(df) < 60:
        return None

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

    # Sharpe = (AnnRet - RFR) / AnnStd
    sharpe = round((ann_ret_val / 100 - RISK_FREE_RATE) / ann_std, 3) if ann_std > 0 else None

    # Sortino — downside std only
    down_ret = df[df["ret"] < 0]["ret"]
    down_std = down_ret.std() * math.sqrt(252) if len(down_ret) > 10 else ann_std
    sortino = round((ann_ret_val / 100 - RISK_FREE_RATE) / down_std, 3) if down_std > 0 else None

    # Max Drawdown
    roll_max = df["nav"].cummax()
    drawdown = (df["nav"] - roll_max) / roll_max
    max_dd = round(float(drawdown.min()), 4)

    # Calmar
    calmar = round((ann_ret_val / 100) / abs(max_dd), 3) if max_dd != 0 else None

    # Annual win rate
    df["year"] = df["date"].dt.year
    annual_rets = df.groupby("year")["nav"].apply(
        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 20 else None
    ).dropna()
    win_rate = round((annual_rets > 0).sum() / len(annual_rets) * 100, 1) if len(annual_rets) > 0 else None

    return {
        "r1": r1, "r3": r3, "r5": r5, "r7": r7, "r10": r10,
        "sharpe": sharpe,
        "std_dev": round(ann_std, 4),
        "max_dd": max_dd,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "ann_ret": ann_ret_val,
        "_annual_rets": annual_rets.to_dict(),   # used for backtest
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
    # Hard filter flags
    sf = (m["sharpe"] is not None) and (m["sharpe"] >= 1.0)
    if asset_type == "Gold":
        df_flag = True   # Gold exempt from DD filter
    else:
        df_flag = (m["max_dd"] is not None) and (-0.30 <= m["max_dd"] <= -0.20)

    fp = sf and df_flag  # fully passing

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
    Pick best-scoring funds per asset type for each profile.
    Conservative:  1 Flexi Cap equity + 1 Gold ETF + 2 best Debt
    Moderate:      1 Flexi + 1 Mid Cap + 1 Gold ETF + 2 Debt
    Aggressive:    1 Flexi + 1 Mid Cap + 1 extra Equity + 1 Gold ETF + 1 Debt
    All funds must have score > 0 and be live.
    """
    def top(asset_type, keywords, n, exclude_ids=None):
        exclude_ids = exclude_ids or []
        pool = [
            f for f in scored_funds
            if f.get("type") == asset_type
            and f.get("live", False)
            and f.get("score", 0) > 0
            and f["id"] not in exclude_ids
            and any(kw.lower() in f.get("cat", "").lower() for kw in keywords)
        ]
        pool.sort(key=lambda x: x.get("score", 0), reverse=True)
        return pool[:n]

    # Gold: prefer ETF over FoF
    gold = top("Gold", ["Gold ETF"], 1) or top("Gold", ["Gold"], 1)

    if profile_key == "C":
        eq_picks  = top("Equity", ["Flexi Cap", "Large Cap"], 1)
        dt_picks  = top("Debt", ["Short Duration", "Corporate Bond", "Banking & PSU", "Banking and PSU"], 2)
    elif profile_key == "M":
        flexi     = top("Equity", ["Flexi Cap", "Large Cap"], 1)
        mid       = top("Equity", ["Mid Cap"], 1, [f["id"] for f in flexi])
        eq_picks  = flexi + mid
        dt_picks  = top("Debt", ["Short Duration", "Corporate Bond", "Banking & PSU", "Banking and PSU"], 2)
    else:  # A
        flexi     = top("Equity", ["Flexi Cap", "Large Cap"], 1)
        mid       = top("Equity", ["Mid Cap"], 1, [f["id"] for f in flexi])
        extra     = top("Equity", ["Flexi Cap", "Large & Mid", "Multi Cap"], 1,
                        [f["id"] for f in flexi + mid])
        eq_picks  = flexi + mid + extra
        dt_picks  = top("Debt", ["Short Duration", "Corporate Bond", "Banking & PSU", "Banking and PSU"], 1)

    all_picks = eq_picks + gold + dt_picks

    return [
        {
            "id":    f["id"],
            "name":  f["name"],
            "type":  f["type"],
            "cat":   f["cat"],
            "score": f["score"],
            "code":  f["code"],
        }
        for f in all_picks
    ]


# ─────────────────────────────────────────────
# STEP 7 — SENSEX/GOLD RATIO (live + history)
# ─────────────────────────────────────────────

def get_sg_ratio_and_history():
    """
    Current SG ratio + annual Dec-31 history from 2000 to now.
    Sensex / (Gold price in INR per 10g)
    """
    print("📊 Fetching Sensex + Gold prices...")

    try:
        # Current values
        sensex_t  = yf.Ticker("^BSESN")
        gold_t    = yf.Ticker("GC=F")
        usdinr_t  = yf.Ticker("INR=X")

        sensex_cur  = float(sensex_t.history(period="5d")["Close"].dropna().iloc[-1])
        gold_usd    = float(gold_t.history(period="5d")["Close"].dropna().iloc[-1])
        usdinr      = float(usdinr_t.history(period="5d")["Close"].dropna().iloc[-1])

        # Gold in INR per 10g (1 troy oz = 31.1035g → 10g = 10/31.1035 oz)
        gold_inr_per_10g = gold_usd * (10 / 31.1035) * usdinr
        current_ratio    = round(sensex_cur / gold_inr_per_10g, 2)
        print(f"   ✅ Sensex: {sensex_cur:.0f} | Gold/10g: ₹{gold_inr_per_10g:.0f} | Ratio: {current_ratio}x")

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

        for year in range(2000, date.today().year + 1):
            try:
                # Use last trading day of December (or closest before)
                target = pd.Timestamp(f"{year}-12-31")
                s  = float(s_hist.asof(target))
                g  = float(g_hist.asof(target))
                fx = float(fx_hist.asof(target))
                if s > 0 and g > 0 and fx > 0:
                    gold_inr = g * (10 / 31.1035) * fx
                    sg_history[str(year)] = round(s / gold_inr, 1)
            except Exception:
                pass

        # Override current year with live value
        sg_history[str(date.today().year)] = current_ratio

    except Exception as e:
        print(f"   ⚠ History fetch failed: {e} — using static fallback")
        # Fallback static history (from your existing data.json)
        sg_history = {
            "2000": 10.8, "2001": 11.2, "2002": 9.6,  "2003": 7.1,
            "2004": 7.8,  "2005": 8.4,  "2006": 9.8,  "2007": 12.4,
            "2008": 13.1, "2009": 6.8,  "2010": 7.2,  "2011": 8.6,
            "2012": 9.4,  "2013": 10.1, "2014": 9.8,  "2015": 8.9,
            "2016": 8.2,  "2017": 8.7,  "2018": 9.3,  "2019": 8.1,
            "2020": 6.9,  "2021": 8.4,  "2022": 9.8,  "2023": 10.2,
            "2024": 9.6,  "2025": 9.5,  str(date.today().year): current_ratio,
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
        result = {}
        for year in range(2007, date.today().year + 1):
            try:
                start = float(nifty.asof(pd.Timestamp(f"{year - 1}-12-31")))
                end   = float(nifty.asof(pd.Timestamp(f"{year}-12-31")))
                if start > 0:
                    result[str(year)] = round((end / start - 1) * 100, 1)
            except Exception:
                pass
        print(f"   ✅ Nifty annual: {list(result.keys())}")
        return result
    except Exception as e:
        print(f"   ❌ Nifty annual failed: {e}")
        return {}


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

    # ── 2. Build fund universe ───────────────────────────
    print("\n🔄 Processing fund universe...")
    funds_raw = []
    fund_id_counter = 1000

    for code in FUND_UNIVERSE_CODES:
        amfi_info = amfi_lookup.get(code)
        if not amfi_info:
            print(f"   ⚠ Code {code} not found in AMFI — skipping")
            continue

        asset_type = classify_type(amfi_info["amfi_cat"], amfi_info["name"])
        if asset_type == "Other":
            continue

        cat = extract_category(amfi_info["amfi_cat"])

        print(f"   [{asset_type}] {amfi_info['name'][:45]}...")
        hist = fetch_history(code, min_years=1)
        time.sleep(0.3)  # be polite to mfapi.in

        if hist is None:
            # Still include fund but mark as not live (no fresh data)
            funds_raw.append({
                "id": fund_id_counter,
                "code": code,
                "name": amfi_info["name"],
                "cat": cat,
                "type": asset_type,
                "nav_latest": amfi_info["nav_latest"],
                "nav_date": amfi_info["nav_date"],
                "live": False,
                "score": 0.0,
                "sf": False, "df": False, "fp": False,
                "data_from": None,
                **{k: None for k in ["r1","r3","r5","r7","r10",
                                      "sharpe","std_dev","max_dd",
                                      "sortino","calmar","win_rate"]},
                "_annual_rets": {},
            })
        else:
            metrics = compute_metrics(hist)
            if metrics is None:
                continue
            score_info = compute_raw_score(metrics, asset_type)
            data_from = int(hist["date"].iloc[0].year)

            funds_raw.append({
                "id":         fund_id_counter,
                "code":       code,
                "name":       amfi_info["name"],
                "cat":        cat,
                "type":       asset_type,
                "nav_latest": amfi_info["nav_latest"],
                "nav_date":   amfi_info["nav_date"],
                "live":       True,
                "data_from":  data_from,
                "r1":         metrics["r1"],
                "r3":         metrics["r3"],
                "r5":         metrics["r5"],
                "r7":         metrics["r7"],
                "r10":        metrics["r10"],
                "sharpe":     metrics["sharpe"],
                "std_dev":    metrics["std_dev"],
                "max_dd":     metrics["max_dd"],
                "sortino":    metrics["sortino"],
                "calmar":     metrics["calmar"],
                "win_rate":   metrics["win_rate"],
                "raw_score":  score_info["raw_score"],
                "sf":         score_info["sf"],
                "df":         score_info["df"],
                "fp":         score_info["fp"],
                "_annual_rets": metrics.get("_annual_rets", {}),
            })

        fund_id_counter += 1

    # ── 3. Normalize scores ──────────────────────────────
    funds_raw = normalize_scores(funds_raw)

    # ── 4. SG Ratio ──────────────────────────────────────
    current_sg, sg_history = get_sg_ratio_and_history()
    gold_active = current_sg > GOLD_THRESHOLD

    # ── 5. Nifty annual returns ───────────────────────────
    nifty_annual = get_nifty_annual()

    # ── 6. Portfolio selection ────────────────────────────
    print("\n🎯 Selecting portfolio funds...")
    live_funds = [f for f in funds_raw if f.get("live")]
    portfolio_selection = {}

    for profile_key, prof in PROFILES.items():
        picks = select_portfolio_funds(live_funds, profile_key)
        portfolio_selection[profile_key] = {
            "profile": profile_key,
            "eq":      prof["eq"],
            "debt":    prof["debt"],
            "funds":   picks,
        }
        names = [f["name"].split("-")[0].strip()[:20] for f in picks]
        print(f"   {profile_key} ({prof['label']}): {names}")

    # ── 7. Backtesting ────────────────────────────────────
    print("\n🔢 Running backtests...")
    backtest = {}
    for profile_key, prof in PROFILES.items():
        picks   = portfolio_selection[profile_key]["funds"]
        pick_ids = {f["id"] for f in picks}

        eq_funds  = [f for f in live_funds if f["type"] == "Equity"  and f["id"] in pick_ids]
        dt_funds  = [f for f in live_funds if f["type"] == "Debt"    and f["id"] in pick_ids]
        gld_funds = [f for f in live_funds if f["type"] == "Gold"    and f["id"] in pick_ids]

        bt = run_backtest(
            profile_key, prof["eq"], prof["debt"],
            sg_history, eq_funds, dt_funds, gld_funds
        )
        backtest[profile_key] = bt
        if bt:
            print(f"   {profile_key}: CAGR {bt['cagr']}% | Sharpe {bt['sharpe']} | MaxDD {bt['max_dd']}")

    # ── 8. Clean up internal fields before writing ────────
    def clean_fund(f):
        out = {k: v for k, v in f.items() if not k.startswith("_")}
        return out

    funds_clean = [clean_fund(f) for f in funds_raw]
    live_count  = sum(1 for f in funds_clean if f.get("live"))

    # ── 9. Assemble data.json ─────────────────────────────
    data_out = {
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
        "backtest":       backtest,
        "funds":          funds_clean,
    }

    with open("data.json", "w") as f:
        json.dump(data_out, f, indent=2)
    print(f"\n✅ data.json written ({live_count} live funds, {len(funds_clean)} total)")

    # ── 10. VIX data ──────────────────────────────────────
    vix_out = generate_vix_data()
    if vix_out:
        with open("vix_data.json", "w") as f:
            json.dump(vix_out, f, indent=2)
        print("✅ vix_data.json written")

    # ── 11. Breadth data ──────────────────────────────────
    breadth_out = generate_breadth_data()
    if breadth_out:
        with open("breadth_data.json", "w") as f:
            json.dump(breadth_out, f, indent=2)
        print("✅ breadth_data.json written")

    print(f"\n🎉 All done — {today_str} {time_str}")


if __name__ == "__main__":
    main()
