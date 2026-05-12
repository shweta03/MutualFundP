"""
MintingM Universe Expander
===========================
Runs WEEKLY (Sunday 1 AM IST) via GitHub Actions.

Takes the existing data.json (from daily generate_data.py run)
and adds ALL valid AMFI regular-plan funds that are NOT already in it.

New funds (<1Y history) get:
  - Basic AMFI metadata (name, category, NAV)
  - Whatever mfapi returns (1M, 3M returns if available)
  - score: null, new_fund: true, live: false (excluded from portfolio)
  - NEW badge in screener, shown at bottom after scored funds

This keeps the daily run fast (only established funds)
while ensuring all new launches appear in the screener weekly.
"""

import requests
import json
import time
from datetime import datetime, date

EXCLUDE_KEYWORDS = [
    "segregated", " series ", "series i ", "series ii", "series iii",
    "series iv", "series v", "fixed term", "ftf", "capital protection",
    "unclaimed", "discontinued", "eco plan", "wealth plan",
    "interval fund", "close ended", "bonus option",
]

EQUITY_SUBCATS   = ["flexi cap","large cap","mid cap","small cap","large & mid cap",
                    "multi cap","focused fund","contra fund","value fund","elss",
                    "balanced advantage","dynamic asset allocation","aggressive hybrid"]
DEBT_SUBCATS     = ["short duration","corporate bond","banking and psu","medium duration",
                    "low duration","money market","ultra short duration"]
GOLD_SUBCATS     = ["gold etf","gold fund","gold fof","gold savings"]
INTL_SUBCATS     = ["fof overseas"]

def get_asset_type(amfi_cat):
    sub = amfi_cat.lower()
    if any(k in sub for k in GOLD_SUBCATS):   return "Gold"
    if any(k in sub for k in INTL_SUBCATS):   return "International"
    if any(k in sub for k in EQUITY_SUBCATS): return "Equity"
    if any(k in sub for k in DEBT_SUBCATS):   return "Debt"
    return "Other"

def get_clean_cat(amfi_cat):
    import re
    m = re.search(r'- (.+?)(?:\))', amfi_cat)
    if m:
        raw = m.group(1)
        for prefix in ["Equity Scheme - ","Debt Scheme - ","Hybrid Scheme - ",
                       "Other Scheme - ","Solution Oriented Scheme - "]:
            if raw.startswith(prefix):
                return raw[len(prefix):]
        return raw
    return amfi_cat

def is_valid(name, amfi_cat):
    n = name.lower()
    if "direct" in n: return False
    if "idcw" in n or " dividend" in n: return False
    if any(k in n for k in EXCLUDE_KEYWORDS): return False
    if "close ended" in amfi_cat.lower(): return False
    return True

def is_fresh(nav_date_str):
    try:
        d = datetime.strptime(nav_date_str, "%d-%b-%Y").date()
        return (date.today() - d).days <= 7 and d.year >= 2024
    except:
        return False

def fetch_quick_history(code):
    """
    Fetch last 3 months of NAV from mfapi.
    Returns basic returns dict or None.
    """
    import pandas as pd
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MintingM-Universe/1.0)"}
    url = f"https://api.mfapi.in/mf/{code}"
    try:
        r = requests.get(url, headers=headers, timeout=8)
        if r.status_code != 200: return None
        raw = r.json()
        if "data" not in raw or not raw["data"]: return None

        df = pd.DataFrame(raw["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        if df.empty: return None

        # Freshness check
        latest = df["date"].iloc[-1].date()
        if (date.today() - latest).days > 7 or latest.year < 2024:
            return None

        today_ts = pd.Timestamp(date.today())
        end_nav  = df["nav"].iloc[-1]

        def abs_ret(months):
            tgt  = today_ts - pd.DateOffset(months=months)
            sub  = df[df["date"] <= tgt]
            if sub.empty: return None
            sd   = sub["date"].iloc[-1]
            if abs((sd - tgt).days) > 20: return None
            return round((end_nav / sub["nav"].iloc[-1] - 1) * 100, 2)

        years_avail = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25

        return {
            "r1m": abs_ret(1),
            "r3m": abs_ret(3),
            "years": round(years_avail, 2),
            "data_from": int(df["date"].iloc[0].year),
        }
    except:
        return None


def main():
    print("\n" + "="*50)
    print(f"  MintingM Universe Expander — {date.today()}")
    print("="*50)

    # ── Load existing data.json ──
    print("\n📂 Loading existing data.json...")
    try:
        with open("data.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ data.json not found — run generate_data.py first")
        return

    existing_codes = set()
    screener = data.get("screener", data.get("funds", []))
    for f in screener:
        if f.get("code"):
            existing_codes.add(f["code"])
    print(f"   Existing scored funds: {len(existing_codes)}")

    # ── Fetch AMFI ──
    print("\n📥 Fetching AMFI NAV file...")
    r = requests.get("https://www.amfiindia.com/spages/NAVAll.txt", timeout=30)
    r.raise_for_status()

    amfi_all = {}
    current_cat = ""
    for line in r.text.splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith("Open Ended") or line.startswith("Close Ended"):
            current_cat = line
        elif ";" in line:
            parts = line.split(";")
            if len(parts) == 6 and parts[4] not in ('', 'N.A.', '-'):
                try:
                    code = int(parts[0].strip())
                    amfi_all[code] = {
                        "name":     parts[3].strip(),
                        "nav":      float(parts[4].strip()),
                        "nav_date": parts[5].strip(),
                        "amfi_cat": current_cat,
                    }
                except: pass

    print(f"   AMFI total: {len(amfi_all)} funds")

    # ── Find new funds not in existing data ──
    new_candidates = []
    for code, info in amfi_all.items():
        if code in existing_codes: continue
        if not is_fresh(info["nav_date"]): continue
        if not is_valid(info["name"], info["amfi_cat"]): continue
        asset_type = get_asset_type(info["amfi_cat"])
        if asset_type == "Other": continue
        new_candidates.append({
            "code":      code,
            "name":      info["name"],
            "cat":       get_clean_cat(info["amfi_cat"]),
            "type":      asset_type,
            "nav_latest":info["nav"],
            "nav_date":  info["nav_date"],
        })

    print(f"   New candidates (not in screener): {len(new_candidates)}")

    # ── Fetch quick history for new funds ──
    new_funds = []
    session = requests.Session()

    for idx, fund in enumerate(new_candidates, 1):
        if idx % 50 == 0:
            print(f"   ... {idx}/{len(new_candidates)}")

        hist = fetch_quick_history(fund["code"])
        time.sleep(0.1)

        if hist is None:
            continue  # no mfapi data at all — skip

        new_funds.append({
            "id":          90000 + idx,
            "code":        fund["code"],
            "name":        fund["name"],
            "cat":         fund["cat"],
            "type":        fund["type"],
            "nav_latest":  fund["nav_latest"],
            "nav_date":    fund["nav_date"],
            "aum":         None,
            "live":        False,  # not in portfolio selection
            "new_fund":    True,
            "data_from":   hist.get("data_from"),
            "monthly_nav": {},
            "r1m":   hist.get("r1m"),
            "r3m":   hist.get("r3m"),
            "r1":    None, "r3": None, "r5": None, "r7": None, "r10": None,
            "sharpe":  None, "std_dev": None, "max_dd": None,
            "sortino": None, "calmar":  None, "win_rate": None,
            "score":   None,  # N/A for new funds
            "raw_score": 0, "sf": 0, "df": 0, "fp": 0,
            "momentum": False, "mean_rev": False, "international": False,
            "annual_rets": {},
        })

    print(f"\n   ✅ {len(new_funds)} new funds added")

    if not new_funds:
        print("   No new funds to add — data.json unchanged")
        return

    # ── Append to screener array and save ──
    data["screener"] = screener + new_funds
    data["funds"]    = data["screener"]
    data["total_funds"] = len(data["screener"])
    data["universe_expanded"] = date.today().isoformat()

    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"   data.json updated: {len(screener)} scored + {len(new_funds)} new = {len(data['screener'])} total")
    print(f"\n🎉 Universe expansion done — {date.today()}")


if __name__ == "__main__":
    main()
