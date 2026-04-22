# MintingM — Setup Guide

## What this repo contains

```
mintingm/
├── index.html              ← Your frontend (no changes needed)
├── data.json               ← Auto-generated daily at 1 PM IST
├── vix_data.json           ← Auto-generated daily at 1 PM IST
├── breadth_data.json       ← Auto-generated daily at 1 PM IST
├── generate_data.py        ← The data engine (runs via GitHub Actions)
├── requirements.txt        ← Python dependencies
└── .github/
    └── workflows/
        └── daily_refresh.yml  ← GitHub Actions schedule
```

---

## One-time setup (15 minutes)

### Step 1 — Create the repo

1. Go to [github.com/new](https://github.com/new)
2. Name: `mintingm` (or anything you want)
3. Set to **Private** (recommended for B2B)
4. Click **Create repository**

### Step 2 — Upload all files

Upload these files to the root of your repo:
- `index.html`
- `generate_data.py`
- `requirements.txt`
- `data.json` ← your current one (will be overwritten by Actions)
- `vix_data.json`
- `breadth_data.json`

Upload the folder:
- `.github/workflows/daily_refresh.yml`
  *(On GitHub: create file → type `.github/workflows/daily_refresh.yml` → paste content)*

### Step 3 — Enable GitHub Pages

1. Go to your repo → **Settings** → **Pages**
2. Under **Source** → select **Deploy from a branch**
3. Branch: `main` | Folder: `/ (root)`
4. Click **Save**
5. After ~2 minutes your URL appears:
   `https://yourusername.github.io/mintingm/`

That's the link you share with distributors. ✅

### Step 4 — Run data engine manually (first time)

1. Go to your repo → **Actions** tab
2. Click **Daily Data Refresh** in the left sidebar
3. Click **Run workflow** → **Run workflow**
4. Wait ~5 minutes for it to complete
5. Check your repo — `data.json`, `vix_data.json`, `breadth_data.json` will be updated
6. Open your GitHub Pages URL — all data is now live and correct

From tomorrow onwards it runs automatically at 1:00 PM IST every weekday.

---

## How the daily refresh works

```
1:00 PM IST every weekday
        ↓
GitHub Actions starts
        ↓
generate_data.py runs:
  1. Fetches all NAVs from AMFI (amfiindia.com)
  2. Fetches 10Y history per fund from mfapi.in
  3. Computes: returns, Sharpe, Sortino, Calmar, MaxDD, WinRate
  4. Computes MintingM Score (exact formula from your UI)
  5. Auto-selects best funds per profile (C/M/A)
  6. Fetches live Sensex/Gold ratio from Yahoo Finance
  7. Runs deterministic backtest (no random numbers)
  8. Fetches Nifty/VIX data → vix_data.json
  9. Computes breadth (stocks above 200 SMA) → breadth_data.json
        ↓
Commits updated JSON files to repo
        ↓
GitHub Pages serves updated index.html automatically
```

---

## What was fixed vs your old setup

| Issue | Old behaviour | Fixed behaviour |
|---|---|---|
| Backtest returns | Random (`Math.random()`) every load | Deterministic from actual fund annual returns |
| Portfolio funds | Hardcoded in JSON manually | Auto-selected by highest MintingM Score per category |
| MintingM Score | Inconsistent / manual | Exact formula from your Formula Guide, normalized 0–10 |
| SG Ratio | Static number typed manually | Live Sensex ÷ Gold(INR) from Yahoo Finance |
| Stale NAV | No validation (2015 dates showing) | Rejects any fund with NAV >7 days old |
| Returns wrong | Fetched from unreliable third parties | Computed from actual daily NAV history (mfapi → AMFI) |

---

## Adding or removing funds from the universe

Open `generate_data.py` and edit `FUND_UNIVERSE_CODES`:

```python
FUND_UNIVERSE_CODES = [
    120403,  # Kotak Flexi Cap
    118989,  # HDFC Mid Cap
    # Add any AMFI scheme code here
    # Find codes at: amfiindia.com → NAV History → search fund name
]
```

After editing, commit the file. GitHub Actions will use the new list next run.

---

## Manual run anytime

Go to: **Repo → Actions → Daily Data Refresh → Run workflow**

Useful when:
- You add new funds to the universe
- You want to refresh data mid-day
- You need to debug a failed run

---

## Troubleshooting

**Actions workflow fails:**
- Click the failed run → expand the step that failed
- Most common cause: mfapi.in timeout → re-run manually, it usually works
- yfinance occasionally returns empty data for Indian tickers on holidays → next day run is fine

**Fund shows score 0.0:**
- The fund's NAV data is stale (>7 days old) — it will auto-recover when AMFI updates

**GitHub Pages shows old data:**
- Hard refresh: Ctrl+Shift+R (the JSON files are cached by browser)
- Or append `?v=DATE` to the URL when sharing with clients

---

## Cost

| Component | Cost |
|---|---|
| GitHub repo (private) | Free |
| GitHub Actions (2000 min/month free) | Free |
| GitHub Pages hosting | Free |
| AMFI data | Free |
| mfapi.in | Free |
| Yahoo Finance (yfinance) | Free |
| **Total** | **₹0/month** |
