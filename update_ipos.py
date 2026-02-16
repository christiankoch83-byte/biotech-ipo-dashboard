#!/usr/bin/env python3
"""
update_ipos.py â Fetch new biotech S-1 filings and update deals.json

Scans SEC EDGAR for new biotech S-1 filings, checks pending IPOs for trading data,
and updates deals.json with active/pending status.
"""

import json
import os
import sys
import subprocess
import requests
import yfinance as yf
from datetime import datetime, timedelta
from urllib.parse import urlencode

# Configuration
SEC_EDGAR_API = "https://efts.sec.gov/LATEST/search-index"
DEALS_FILE = "deals.json"
USER_AGENT = "BiotechIPODashboard/1.0 cko@bellevue.ch"
SEC_REQUEST_DELAY = 0.1  # 10 req/sec = max 0.1s between requests

# Biotech keywords for filtering
BIOTECH_KEYWORDS = [
    "biotechnology", "pharmaceutical", "biopharmaceutical",
    "gene therapy", "cell therapy", "oncology", "immunotherapy",
    "vaccine", "antibody", "drug discovery", "diagnostic"
]


def log_msg(msg):
    """Log with timestamp."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def load_deals():
    """Load deals from JSON file."""
    try:
        if os.path.exists(DEALS_FILE):
            with open(DEALS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        log_msg(f"ERROR loading deals.json: {e}")
    return []


def save_deals(deals):
    """Save deals to JSON file."""
    try:
        with open(DEALS_FILE, "w") as f:
            json.dump(deals, f, indent=2)
        log_msg(f"Saved {len(deals)} deals to {DEALS_FILE}")
        return True
    except Exception as e:
        log_msg(f"ERROR saving deals.json: {e}")
        return False


def search_sec_edgar(start_date, end_date):
    """
    Search SEC EDGAR for new biotech S-1 filings.
    Returns list of (company_name, cik) tuples.
    """
    log_msg(f"Searching SEC EDGAR for S-1 filings ({start_date} to {end_date})...")
    
    try:
        # Build query with biotech keywords
        keywords = " OR ".join([f'"{kw}"' for kw in BIOTECH_KEYWORDS[:5]])
        
        params = {
            "q": keywords,
            "forms": "S-1",
            "dateRange": "custom",
            "startdt": start_date,
            "enddt": end_date,
            "count": 100
        }
        
        headers = {"User-Agent": USER_AGENT}
        url = f"{SEC_EDGAR_API}?{urlencode(params)}"
        
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        
        data = resp.json()
        filings = data.get("results", [])
        
        log_msg(f"Found {len(filings)} potential biotech S-1 filings")
        
        results = []
        for filing in filings[:10]:  # Limit to avoid too many new entries
            try:
                company_name = filing.get("entity_name", "Unknown")
                cik = filing.get("cik_str", "")
                if company_name and cik:
                    results.append((company_name, str(cik)))
            except Exception:
                pass
        
        return results
    
    except requests.RequestException as e:
        log_msg(f"ERROR fetching SEC EDGAR: {e}")
        return []
    except Exception as e:
        log_msg(f"ERROR parsing SEC response: {e}")
        return []


def lookup_ticker(company_name, cik):
    """
    Try to find ticker for a company.
    Uses yfinance Ticker search (limited functionality).
    Returns ticker string or None.
    """
    try:
        # Try simple lookup
        ticker = yf.Ticker(company_name)
        if ticker.info and ticker.info.get("symbol"):
            return ticker.info["symbol"]
    except Exception:
        pass
    return None


def check_trading_data(ticker):
    """
    Check if ticker has trading data (i.e., is actively trading).
    Returns dict with price data if found, else None.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            latest_close = float(hist["Close"].iloc[-1])
            return {
                "current_price": latest_close,
                "date_fetched": datetime.now().isoformat()
            }
    except Exception:
        pass
    return None


def promote_pending_to_active(deals):
    """
    For each pending IPO, check if it has trading data.
    If yes, promote to active and fill in available fields.
    Returns number of promotions.
    """
    promoted = 0
    for deal in deals:
        if deal.get("ipo_status") == "pending":
            ticker = deal.get("ticker")
            if ticker:
                log_msg(f"Checking trading data for pending IPO: {ticker}")
                trading_data = check_trading_data(ticker)
                if trading_data:
                    deal["ipo_status"] = "active"
                    if deal.get("current_price") is None:
                        deal["current_price"] = trading_data["current_price"]
                    log_msg(f"Promoted {ticker} to active")
                    promoted += 1
    return promoted


def add_pending_ipo(deals, company_name, ticker, cik):
    """
    Add a new pending IPO to deals list if not already present.
    Returns True if added, False if already exists.
    """
    # Check if ticker already exists
    for deal in deals:
        if deal.get("ticker") == ticker:
            return False
    
    # Add new pending entry
    new_deal = {
        "year": datetime.now().year,
        "company": company_name,
        "ticker": ticker,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "area": "Unknown",
        "offer": None,
        "proceeds": None,
        "pricing": "Unknown",
        "status": "Unknown",
        "phase": "Preclinical",
        "range_lo": None,
        "range_hi": None,
        "cik": cik,
        "ipo_status": "pending"
    }
    deals.append(new_deal)
    log_msg(f"Added pending IPO: {ticker} ({company_name})")
    return True


def git_commit_if_changed(filename):
    """
    If file has changed, commit and push to git.
    Requires git credentials to be configured.
    """
    try:
        # Check if file changed
        result = subprocess.run(
            ["git", "status", "--short", filename],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if not result.stdout.strip():
            log_msg(f"No changes to {filename}, skipping git commit")
            return True
        
        log_msg(f"Changes detected in {filename}, committing...")
        
        # Configure git user (in case not set)
        subprocess.run(
            ["git", "config", "user.email", "automation@biotech-dashboard.local"],
            capture_output=True,
            timeout=5
        )
        subprocess.run(
            ["git", "config", "user.name", "Biotech IPO Dashboard Bot"],
            capture_output=True,
            timeout=5
        )
        
        # Add, commit, push
        subprocess.run(["git", "add", filename], capture_output=True, timeout=5)
        subprocess.run(
            ["git", "commit", "-m", f"Update {filename} with new IPO data"],
            capture_output=True,
            timeout=5
        )
        subprocess.run(["git", "push"], capture_output=True, timeout=10)
        
        log_msg(f"Successfully pushed changes to git")
        return True
    
    except subprocess.TimeoutExpired:
        log_msg("WARNING: Git operation timed out")
        return False
    except Exception as e:
        log_msg(f"WARNING: Git error: {e}")
        return False


def main():
    """Main update flow."""
    log_msg("Starting IPO dashboard update...")
    
    # Load current deals
    deals = load_deals()
    log_msg(f"Loaded {len(deals)} existing deals")
    
    # Scan SEC EDGAR for new filings (last 7 days)
    today = datetime.now().strftime("%Y-%m-%d")
    week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    new_filings = search_sec_edgar(week_ago, today)
    
    added_count = 0
    for company_name, cik in new_filings:
        # Try to find ticker
        ticker = lookup_ticker(company_name, cik)
        if ticker:
            if add_pending_ipo(deals, company_name, ticker, cik):
                added_count += 1
    
    log_msg(f"Added {added_count} new pending IPOs")
    
    # Check pending IPOs for trading data
    promoted_count = promote_pending_to_active(deals)
    log_msg(f"Promoted {promoted_count} pending IPOs to active")
    
    # Save if anything changed
    if added_count > 0 or promoted_count > 0:
        if save_deals(deals):
            git_commit_if_changed(DEALS_FILE)
    else:
        log_msg("No changes to save")
    
    log_msg("IPO dashboard update complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
