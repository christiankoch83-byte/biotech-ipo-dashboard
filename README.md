# Biotech IPO Dashboard — 2025-2026

Real-time biotech IPO tracker with live stock prices, performance windows, and risk appetite scoring.

## Quick Deploy (5 minutes)

### Option 1: Streamlit Community Cloud (Free — Recommended)

1. **Push to GitHub**
   - Create a new repo (e.g., `biotech-ipo-dashboard`)
   - Push all files in this folder to the repo

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repo, branch `main`, and file `app.py`
   - Click "Deploy"
   - Your dashboard will be live at `https://your-app.streamlit.app`

3. **Share with your team**
   - Send them the URL — no login required
   - Prices auto-refresh every 15 minutes
   - Click "Refresh prices" button for on-demand updates

### Option 2: Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Share on your network via `streamlit run app.py --server.address 0.0.0.0`.

## Adding New IPOs

Edit the `DEALS` list in `app.py`. Each deal is a dictionary:

```python
dict(
    year=2026,
    company="Company Name",
    ticker="TICK",
    date="2026-03-15",          # IPO date
    area="Therapeutic Area",
    offer=18.00,                 # Offer price
    proceeds=250.0,              # Gross proceeds in $M
    pricing="Above",             # Above / At Top / At / Below
    status="Upsized",           # Upsized / Inline
    phase="Phase 2",
    range_lo=16,                # Filing range low (or None)
    range_hi=18,                # Filing range high (or None)
)
```

Live prices and all return calculations happen automatically via Yahoo Finance.

## Features

- **Live prices** via yfinance (15-min cache)
- **4 performance windows**: Day-1, 1-Week, 1-Month, IPO-to-Date
- **Fade analysis**: scatter plot showing which pops hold vs. fade
- **Risk appetite gauge**: composite score from pricing, demand, and aftermarket signals
- **Filterable by year** with interactive Plotly charts
- **Sortable data table** with search
- **Dark theme** matching the original HTML dashboard
