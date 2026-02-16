"""
Biotech IPO Dashboard — 2020-2026 YTD
Real-time pricing via yfinance. Shareable via Streamlit Cloud.
Data loaded from deals.json with pending IPO tracking.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as p
from datetime import datetime, timedelta
import numpy as np
import json
import os

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Biotech IPO Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] {
        background: #181b23;
        border: 1px solid #2a2e3a;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMetric"] label { font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.8px; }
    .thesis-box {
        background: linear-gradient(135deg, rgba(79,140,255,0.08), rgba(124,92,252,0.08));
        border: 1px solid rgba(79,140,255,0.2);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        font-size: 14px;
    }
    .thesis-box strong { color: #4f8cff; }
    div[data-testid="stHorizontalBlock"] > div { min-width: 0 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #181b23;
        border: 1px solid #2a2e3a;
        border-radius: 6px;
        padding: 6px 16px;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] { background: #4f8cff !important; border-color: #4f8cff !important; }
    .pending-ipo-box {
        background: rgba(245,158,11,0.06);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────
def load_deals_from_json():
    """Load IPO deal data from deals.json file."""
    try:
        if os.path.exists("deals.json"):
            with open("deals.json", "r") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"Could not load deals.json: {e}. Using fallback data.")
    return []


# ──────────────────────────────────────────────
# LIVE PRICE FETCHING (cached 15 min)
# ──────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def fetch_live_prices(tickers: list[str]) -> dict:
    """Fetch current prices and historical data for return calculations."""
    results = {}
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist = stock.history(period="1y")
            if hist.empty:
                results[t] = {"current": None, "history": pd.DataFrame()}
                continue
            results[t] = {
                "current": round(float(hist["Close"].iloc[-1]), 2),
                "history": hist,
            }
        except Exception:
            results[t] = {"current": None, "history": pd.DataFrame()}
    return results


def calc_return_at_offset(history: pd.DataFrame, ipo_date: str, offer: float, trading_days: int) -> float | None:
    """Calculate return from offer price at N trading days after IPO."""
    try:
        ipo_dt = pd.Timestamp(ipo_date).tz_localize(history.index.tz) if history.index.tz else pd.Timestamp(ipo_date)
        post_ipo = history[history.index >= ipo_dt]
        if len(post_ipo) > trading_days:
            price = float(post_ipo["Close"].iloc[trading_days])
            return round((price / offer - 1) * 100, 1)
    except Exception:
        pass
    return None


def build_dataframe(deals: list[dict], prices: dict) -> pd.DataFrame:
    """Merge deal data with live prices and calculate return windows."""
    rows = []
    for d in deals:
        t = d.get("ticker")
        p = prices.get(t, {})
        current = p.get("current")
        hist = p.get("history", pd.DataFrame())

        # Only calculate returns for active IPOs
        if d.get("ipo_status") == "active" and d.get("offer"):
            day1 = calc_return_at_offset(hist, d["date"], d["offer"], 1)
            week1 = calc_return_at_offset(hist, d["date"], d["offer"], 5)
            month1 = calc_return_at_offset(hist, d["date"], d["offer"], 21)
            itd = round((current / d["offer"] - 1) * 100, 1) if current else None
        else:
            day1 = week1 = month1 = itd = None

        rows.append({
            **d,
            "current_price": current,
            "day1_ret": day1,
            "week1_ret": week1,
            "month1_ret": month1,
            "ipo_to_date": itd,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ──────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────
PURPLE = "#7c5cfc"
CYAN = "#06b6d4"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f59e0b"
TEAL = "#14b8a6"
BLUE = "#4f8cff"
DIM = "#8b8fa3"
SURFACE = "#181b23"
BG = "#0f1117"
BORDER = "#2a2e3a"

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(color=DIM, size=11),
    margin=dict(l=50, r=20, t=30, b=40),
    hoverlabel=dict(bgcolor=SURFACE, font_size=12),
)


def color_for_val(v):
    if v is None: return DIM
    return GREEN if v > 0 else RED if v < 0 else DIM


def fmt_ret(v):
    if v is None: return "—"
    return f"+{v:.1f}%" if v > 0 else f"{v:.1f}%"


# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():
    # Load deals
    deals_raw = load_deals_from_json()
    
    # Header
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.markdown("# 🧬 Biotech **IPO** Dashboard")
        st.caption(f"2020 — 2026 YTD  |  Prices refresh every 15 min  |  Last fetch: {datetime.now().strftime('%b %d, %Y %H:%M')}")
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh prices", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Thesis banner
    st.markdown("""
    <div class="thesis-box">
        <strong>Research Thesis — IPO Activity as Risk Appetite Indicator</strong><br>
        Monitoring pricing dynamics, demand signals (upsizing), and aftermarket trading as early indicators
        of risk sentiment and supply/demand. Mid-2026 is the next key proof point. Aftermarket performance
        windows reveal whether initial pops hold or fade — a critical distinction for gauging real vs. speculative demand.
    </div>
    """, unsafe_allow_html=True)

    # Fetch live data
    active_deals = [d for d in deals_raw if d.get("ipo_status") == "active"]
    tickers = [d["ticker"] for d in active_deals]
    with st.spinner("Fetching live prices..."):
        prices = fetch_live_prices(tickers)
    df = build_dataframe(active_deals, prices)

    # Year filter
    years_available = sorted(df["year"].unique())
    year_options = ["All"] + [str(y) for y in years_available]
    
    # Year filter
    year_filter = st.radio("Filter", year_options, horizontal=True, label_visibility="collapsed")
    if year_filter != "All":
        df = df[df["year"] == int(year_filter)]

    n = len(df)
    if n == 0:
        st.warning("No deals for this filter.")
        return

    # ── KPI ROW ──────────────────────────────
    total_proceeds = df["proceeds"].sum()
    avg_day1 = df["day1_ret"].mean() if df["day1_ret"].notna().any() else 0
    avg_itd = df["ipo_to_date"].mean() if df["ipo_to_date"].notna().any() else 0
    median_size = df["proceeds"].median()
    n_upsized = (df["status"] == "Upsized").sum()
    n_above = df["pricing"].isin(["Above", "At Top"]).sum()
    n_above_offer = (df["ipo_to_date"] > 0).sum() if df["ipo_to_date"].notna().any() else 0

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    k1.metric("Deals", n)
    k2.metric("Total Proceeds", f"${total_proceeds/1000:.1f}B")
    k3.metric("Avg Day-1", fmt_ret(avg_day1))
    k4.metric("Avg IPO-to-Date", fmt_ret(avg_itd))
    k5.metric("Median Size", f"${median_size:.0f}M")
    k6.metric("Upsized", f"{n_upsized}/{n}")
    k7.metric("Above Range", f"{n_above}/{n}")
    k8.metric("Above Offer", f"{n_above_offer}/{n}")

    st.markdown("---")

    # ── CHARTS ──────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Windows",
        "📈 Volume & Proceeds",
        "🎯 Fade Analysis",
        "💰 Deal Size",
        "🏷️ Pricing Signal",
    ])

    # ── TAB 1: Performance Windows ──
    with tab1:
        sorted_df = df.sort_values("date")
        fig = go.Figure()
        for col, name, color in [
            ("day1_ret", "Day-1", PURPLE),
            ("week1_ret", "1-Week", BLUE),
            ("month1_ret", "1-Month", CYAN),
            ("ipo_to_date", "IPO-to-Date", TEAL),
        ]:
            fig.add_trace(go.Bar(
                x=sorted_df["ticker"],
                y=sorted_df[col],
                name=name,
                marker_color=color,
                opacity=0.85,
                hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>",
            ))
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            barmode="group",
            height=400,
            yaxis_title="Return from Offer %",
            yaxis_ticksuffix="%",
            legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 2: Volume & Proceeds ──
    with tab2:
        df_vol = df.copy()
        df_vol["month"] = df_vol["date"].dt.to_period("M").astype(str)
        monthly = df_vol.groupby("month").agg(deals=("ticker", "count"), proceeds=("proceeds", "sum")).reset_index()

        # Fill missing months
        all_months = pd.period_range("2025-01", "2026-02", freq="M").astype(str)
        monthly = monthly.set_index("month").reindex(all_months, fill_value=0).reset_index().rename(columns={"index": "month"})
        monthly["label"] = monthly["month"].apply(lambda m: pd.Timestamp(m).strftime("%b '%y"))

        fig = go.Figure()
        colors = [CYAN if m.startswith("2026") else PURPLE for m in monthly["month"]]
        fig.add_trace(go.Bar(x=monthly["label"], y=monthly["deals"], name="Deals", marker_color=colors, yaxis="y", opacity=0.8))
        fig.add_trace(go.Scatter(x=monthly["label"], y=monthly["proceeds"], name="Proceeds ($M)", yaxis="y2",
                                 line=dict(color=ORANGE, width=2), fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                                 mode="lines+markers", marker=dict(size=5)))
        fig.update_layout(
            **LAYOUT_DEFAULTS,
            height=400,
            yaxis=dict(title="Deals", gridcolor=BORDER, dtick=1),
            yaxis2=dict(title="Proceeds ($M)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)",
                        tickprefix="$", ticksuffix="M", title_font=dict(color=ORANGE), tickfont=dict(color=ORANGE)),
            legend=dict(orientation="h", y=1.12, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3: Fade Analysis ──
    with tab3:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Day-1 Pop vs. IPO-to-Date** — above the diagonal = gains extended, below = faded")
            fig = go.Figure()
            for yr, color in [(2025, PURPLE), (2026, CYAN)]:
                mask = df["year"] == yr
                subset = df[mask]
                fig.add_trace(go.Scatter(
                    x=subset["day1_ret"], y=subset["ipo_to_date"],
                    mode="markers+text", text=subset["ticker"],
                    textposition="top center", textfont=dict(size=10, color=DIM),
                    marker=dict(size=12, color=color, opacity=0.8),
                    name=str(yr),
                    hovertemplate="%{text}: Day-1 %{x:.1f}%, ITD %{y:.1f}%<extra></extra>",
                ))
            # Diagonal reference
            min_v = min(df["day1_ret"].min(), df["ipo_to_date"].min(), -30)
            max_v = max(df["day1_ret"].max(), df["ipo_to_date"].max(), 30)
            fig.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines",
                                     line=dict(dash="dot", color=DIM, width=1), showlegend=False))
            # Zero lines
            fig.add_hline(y=0, line=dict(color=BORDER, width=1))
            fig.add_vline(x=0, line=dict(color=BORDER, width=1))
            fig.update_layout(
                **LAYOUT_DEFAULTS,
                height=420,
                xaxis_title="Day-1 Return %", yaxis_title="IPO-to-Date Return %",
                xaxis_ticksuffix="%", yaxis_ticksuffix="%",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Heatmap — Current status from offer**")
            for _, row in df.sort_values("ipo_to_date", ascending=False).iterrows():
                itd = row["ipo_to_date"]
                if itd is None or pd.isna(itd):
                    tag, bg, border = "No Data", "rgba(139,143,163,0.06)", BORDER
                elif itd >= 50:
                    tag, bg, border = "STRONG HOLD", "rgba(34,197,94,0.12)", GREEN
                elif itd >= 10:
                    tag, bg, border = "HOLDING", "rgba(34,197,94,0.06)", GREEN
                elif itd >= -5:
                    tag, bg, border = "FLAT", "rgba(139,143,163,0.06)", DIM
                elif itd >= -20:
                    tag, bg, border = "FADING", "rgba(239,68,68,0.06)", RED
                else:
                    tag, bg, border = "UNDERWATER", "rgba(239,68,68,0.12)", RED
                color = GREEN if (itd or 0) > 0 else RED if (itd or 0) < 0 else DIM
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border};border-radius:8px;padding:8px 12px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:700;font-size:13px;">{row['ticker']}</span>
                        <span style="color:{DIM};font-size:11px;margin-left:8px;">Day-1: {fmt_ret(row['day1_ret'])}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:{color};font-weight:700;font-size:15px;">{fmt_ret(itd)}</span>
                        <span style="color:{border};font-size:9px;text-transform:uppercase;letter-spacing:0.5px;margin-left:6px;">{tag}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: Deal Size ──
    with tab4:
        sorted_size = df.sort_values("proceeds", ascending=True)
        colors = [CYAN if y == 2026 else PURPLE for y in sorted_size["year"]]
        fig = go.Figure(go.Bar(
            y=sorted_size["ticker"],
            x=sorted_size["proceeds"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: $%{x:.1f}M<extra></extra>",
            text=[f"${p:.0f}M" for p in sorted_size["proceeds"]],
            textposition="outside",
            textfont=dict(size=11, color=DIM),
        ))
        fig.update_layout(**LAYOUT_DEFAULTS, height=450, xaxis_title="Gross Proceeds ($M)", xaxis_tickprefix="$", xaxis_ticksuffix="M")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: Pricing Signal ──
    with tab5:
        c1, c2 = st.columns(2)
        with c1:
            above = df["pricing"].isin(["Above", "At Top"]).sum()
            at = (df["pricing"] == "At").sum()
            below = (df["pricing"] == "Below").sum()
            fig = go.Figure(go.Pie(
                labels=["Above / Top", "At Range", "Below Range"],
                values=[above, at, below],
                marker=dict(colors=[GREEN, ORANGE, RED]),
                hole=0.55,
                textinfo="value+label",
                textfont=dict(size=13),
            ))
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=SURFACE, font=dict(color=DIM), height=300,
                              margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Risk appetite gauge — with data density factor
            pct_upsized = n_upsized / n if n else 0
            pct_above = n_above / n if n else 0
            pct_above_offer = n_above_offer / n if n else 0
            d1_score = min(25, max(0, (avg_day1 + 10) / 50 * 25))    # max 25 — Day-1 return signal
            ups_score = pct_upsized * 20                               # max 20 — % deals upsized
            abv_score = pct_above * 20                                 # max 20 — % priced above range
            itd_score = min(20, max(0, (avg_itd + 10) / 100 * 20))   # max 20 — IPO-to-date return
            hold_score = pct_above_offer * 15                          # max 15 — % still above offer
            raw_score = d1_score + ups_score + abv_score + itd_score + hold_score
            # Data density: penalize small samples (full confidence at N >= 15)
            density_factor = min(1.0, n / 15)
            score = min(100, max(0, round(raw_score * density_factor)))
            label = "Strong" if score >= 70 else "Constructive" if score >= 50 else "Cautious" if score >= 30 else "Risk-Off"
            if density_factor < 0.6:
                label += " (Low N)"
            gauge_color = GREEN if score >= 60 else ORANGE if score >= 40 else RED

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number=dict(suffix=f" — {label}", font=dict(size=22, color=gauge_color)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=DIM, dtick=25),
                    bar=dict(color=gauge_color),
                    bgcolor=SURFACE,
                    bordercolor=BORDER,
                    steps=[
                        dict(range=[0, 30], color="rgba(239,68,68,0.08)"),
                        dict(range=[30, 50], color="rgba(245,158,11,0.08)"),
                        dict(range=[50, 70], color="rgba(245,158,11,0.04)"),
                        dict(range=[70, 100], color="rgba(34,197,94,0.08)"),
                    ],
                ),
                title=dict(text="Risk Appetite Score", font=dict(size=14, color=DIM)),
            ))
            fig.update_layout(paper_bgcolor=BG, font=dict(color=DIM), height=300, margin=dict(l=30, r=30, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Score breakdown tooltip
            st.caption(
                f"**Score Breakdown** (N={n}, density={density_factor:.0%}):  \n"
                f"Day-1 Return: {d1_score:.1f}/25 · Upsized: {ups_score:.1f}/20 · "
                f"Above Range: {abv_score:.1f}/20 · ITD Return: {itd_score:.1f}/20 · "
                f"Above Offer: {hold_score:.1f}/15 → Raw {raw_score:.0f} × {density_factor:.0%} = **{score}**"
            )

    st.markdown("---")

    # ── FULL TABLE ──────────────────────────
    st.markdown("### All Biotech IPOs — Full Performance Detail")

    search = st.text_input("Search by company or ticker", "", label_visibility="collapsed", placeholder="Search company or ticker...")
    display_df = df.copy()
    if search:
        mask = display_df["company"].str.contains(search, case=False) | display_df["ticker"].str.contains(search, case=False)
        display_df = display_df[mask]

    # Format for display
    table_df = display_df[["year", "company", "ticker", "date", "area", "offer", "proceeds", "pricing", "status",
                           "day1_ret", "week1_ret", "month1_ret", "current_price", "ipo_to_date"]].copy()
    table_df["date"] = table_df["date"].dt.strftime("%b %d, '%y")
    table_df.columns = ["Year", "Company", "Ticker", "IPO Date", "Area", "Offer $", "Proceeds ($M)",
                         "Pricing", "Status", "Day-1 %", "1-Wk %", "1-Mo %", "Current $", "IPO-to-Date %"]

    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Offer $": st.column_config.NumberColumn(format="$%.2f"),
            "Proceeds ($M)": st.column_config.NumberColumn(format="$%.1fM"),
            "Current $": st.column_config.NumberColumn(format="$%.2f"),
            "Day-1 %": st.column_config.NumberColumn(format="%.1f%%"),
            "1-Wk %": st.column_config.NumberColumn(format="%.1f%%"),
            "1-Mo %": st.column_config.NumberColumn(format="%.1f%%"),
            "IPO-to-Date %": st.column_config.NumberColumn(format="%.1f%%"),
        },
        height=600,
    )

    # ── FOOTER ──────────────────────────────
    st.markdown("---")
    # ── PENDING IPOs SECTION ────────────────────────
    pending_deals = [d for d in deals_raw if d.get("ipo_status") == "pending"]
    if pending_deals:
        st.markdown("---")
        st.markdown("### Pending IPOs (Filed but not yet trading)")
        st.markdown("These companies have filed S-1s and are awaiting IPO launch.")
        for deal in sorted(pending_deals, key=lambda x: x.get("date", ""), reverse=True):
            st.markdown(f"""
            <div class="pending-ipo-box">
                <strong>{deal.get('company', 'Unknown')}</strong> \u2022 {deal.get('ticker', '?')}
                <br/>
                <span style="font-size:12px;color:#8b8fa3;">
                    {deal.get('area', 'Unknown')} | Filed: {deal.get('date', 'Unknown')}
                </span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
    "Sources: [BioPharma Dive](https://www.biopharmadive.com) · "
    "[BioBucks](https://www.biobucks.co) · "
    "[BioSpace](https://www.biospace.com) · "
    "[Fierce Biotech](https://www.fiercebiotech.com) · "
    "Live prices via [Yahoo Finance](https://finance.yahoo.com)  \n"
    "Not investment advice. Returns calculated from offer price."
    )


if __name__ == "__main__":
    main()
