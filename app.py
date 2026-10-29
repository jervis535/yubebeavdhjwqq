"""FX Forecaster - pick two currencies, see a forecast with confidence, and get a
convert-now-or-later read.  Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_sources import DataError, load_rates
from forecast import QUANTILES, advise, backtest, confidence, forecast

st.set_page_config(page_title="FX Forecaster", page_icon="💱", layout="wide")

CURRENCIES = {
    "USD": "US Dollar", "EUR": "Euro", "GBP": "British Pound", "JPY": "Japanese Yen",
    "CHF": "Swiss Franc", "AUD": "Australian Dollar", "CAD": "Canadian Dollar",
    "NZD": "New Zealand Dollar", "CNY": "Chinese Yuan", "HKD": "Hong Kong Dollar",
    "SGD": "Singapore Dollar", "THB": "Thai Baht", "IDR": "Indonesian Rupiah",
    "MYR": "Malaysian Ringgit", "PHP": "Philippine Peso", "VND": "Vietnamese Dong",
    "INR": "Indian Rupee", "KRW": "South Korean Won", "TWD": "Taiwan Dollar",
    "AED": "UAE Dirham", "SAR": "Saudi Riyal", "ZAR": "South African Rand",
    "TRY": "Turkish Lira", "MXN": "Mexican Peso", "BRL": "Brazilian Real",
    "SEK": "Swedish Krona", "NOK": "Norwegian Krone", "DKK": "Danish Krone",
    "PLN": "Polish Zloty", "CZK": "Czech Koruna",
}
HORIZONS = {"1 week": 5, "2 weeks": 10, "1 month": 21, "2 months": 42, "3 months": 63}
HISTORY = {"1 year": 365, "2 years": 730, "3 years": 1095, "5 years": 1825}
SOURCES = ["Google Sheets (GOOGLEFINANCE)", "Frankfurter (ECB)", "Synthetic demo (not real data)"]


def _plain(x):
    if hasattr(x, "items"):
        return {k: _plain(v) for k, v in x.items()}
    return x


def get_gsheets_secrets() -> dict:
    try:
        return _plain(st.secrets.get("gsheets", {}))
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def cached_rates(source, base, quote, days, _secrets):
    return load_rates(source, base, quote, days, _secrets)


def fmt_rate(x: float) -> str:
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}" if x >= 1 else f"{x:,.5f}"


# ----------------------------------------------------------------------------
# Header + inputs
# ----------------------------------------------------------------------------
st.title("💱 FX Forecaster")
st.caption("Forecast an exchange rate, see how much to trust it, and decide whether to convert now or wait.")

gs_secrets = get_gsheets_secrets()
with st.sidebar:
    st.header("Settings")
    default_src = 0 if gs_secrets else 1
    source = st.selectbox("Data source", SOURCES, index=default_src)
    if source == SOURCES[0] and not gs_secrets:
        st.warning("No `[gsheets]` secrets found. See README.md.")
    history = st.selectbox("History to use", list(HISTORY), index=1)
    with st.expander("Model settings"):
        trend = st.slider("Trend influence", 0, 100, 25, 5, format="%d%%",
                          help="How much of the recent trend is carried forward. 0% = pure random walk.") / 100
        lam = st.slider("Volatility memory", 0.80, 0.99, 0.94, 0.01,
                        help="Higher = volatility reacts more slowly to recent moves.")
        min_edge = st.slider("Minimum gain to justify waiting", 0.0, 2.0, 0.25, 0.05, format="%.2f%%",
                             help="Expected improvement needed before the app suggests waiting.") / 100

with st.form("inputs"):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
    codes = list(CURRENCIES)
    base = c1.selectbox("I have", codes, index=codes.index("USD"),
                        format_func=lambda c: f"{c} - {CURRENCIES[c]}")
    quote = c2.selectbox("I want", codes, index=codes.index("EUR"),
                         format_func=lambda c: f"{c} - {CURRENCIES[c]}")
    horizon_label = c3.selectbox("I could wait up to", list(HORIZONS), index=2)
    amount = c4.number_input("Amount (of what you have)", min_value=1.0, value=1000.0, step=100.0)
    go_btn = st.form_submit_button("Forecast", type="primary", width="stretch")

if not go_btn and "ran" not in st.session_state:
    st.info("Choose two currencies and press **Forecast**.")
    st.stop()
st.session_state["ran"] = True

if base == quote:
    st.error("Pick two different currencies.")
    st.stop()

horizon = HORIZONS[horizon_label]

# ----------------------------------------------------------------------------
# Load + model
# ----------------------------------------------------------------------------
try:
    with st.spinner("Fetching data..." + (" (Google Sheets can take ~10s)" if source == SOURCES[0] else "")):
        series = cached_rates(source, base, quote, HISTORY[history], gs_secrets)
except DataError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # network etc.
    st.error(f"Could not load data: {e}")
    st.stop()

if source == SOURCES[2]:
    st.warning("Synthetic demo data: these are NOT real exchange rates.")

fc = forecast(series, horizon, lam=lam, trend_shrink=trend)
bt = backtest(series, horizon, lam=lam, trend_shrink=trend)
p_up_h = float((fc.paths[:, -1] > fc.last_rate).mean())
conf = confidence(p_up_h, bt)
adv = advise(fc, conf, horizon, min_edge=min_edge)

# ----------------------------------------------------------------------------
# Verdict
# ----------------------------------------------------------------------------
color = {"WAIT": "#2e7d32", "CONVERT NOW": "#1565c0", "NO CLEAR EDGE": "#8d6e00"}[adv.verdict]
st.markdown(
    f"""<div style="border-left:6px solid {color};padding:12px 18px;border-radius:6px;
    background:rgba(128,128,128,0.08)">
    <div style="font-size:0.8rem;letter-spacing:.08em;color:{color};font-weight:700">{adv.verdict}</div>
    <div style="font-size:1.4rem;font-weight:600">{adv.headline}</div>
    <div style="opacity:.75">{base} → {quote}, deciding within {horizon_label}</div></div>""",
    unsafe_allow_html=True,
)
st.write("")

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Rate today ({base}/{quote})", fmt_rate(fc.last_rate))
m2.metric(f"Median in {horizon_label}", fmt_rate(float(fc.bands[0.5].iloc[-1])), f"{adv.median_change:+.2%}")
m3.metric("Chance it's higher then", f"{adv.p_up:.0%}")
m4.metric("Forecast confidence", f"{conf.label}", f"{conf.score:.0f}/100", delta_color="off")

# ----------------------------------------------------------------------------
# Fan chart
# ----------------------------------------------------------------------------
hist = series.iloc[-min(len(series), 250):]
b = fc.bands
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="History", line=dict(color="#555", width=2)))
for lo, hi, a, name in [(0.025, 0.975, 0.12, "95% range"), (0.10, 0.90, 0.22, "80% range"), (0.25, 0.75, 0.35, "50% range")]:
    x = list(b.index) + list(b.index[::-1])
    y = list(b[hi]) + list(b[lo][::-1])
    fig.add_trace(go.Scatter(x=x, y=y, fill="toself", name=name, line=dict(width=0),
                             fillcolor=f"rgba(21,101,192,{a})", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=b.index, y=b[0.5], name="Median forecast", line=dict(color="#1565c0", width=2, dash="dash")))
fig.add_hline(y=fc.last_rate, line=dict(color="#999", width=1, dash="dot"),
              annotation_text="today", annotation_position="bottom left")
fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                  title=f"{base}/{quote}: history and {horizon_label} forecast",
                  yaxis_title=f"1 {base} in {quote}", legend=dict(orientation="h", y=-0.15))
st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------------------
# Details
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1])
with left:
    st.subheader("What this means for your money")
    now_val = amount * fc.last_rate
    st.write(f"Converting **{amount:,.0f} {base}** today gives about **{now_val:,.2f} {quote}**.")
    tbl = pd.DataFrame({
        "Scenario": ["Bad case (5th pct)", "Median", "Expected", "Good case (95th pct)"],
        f"{quote} received": [now_val * (1 + adv.downside5), now_val * (1 + adv.median_change),
                              now_val * (1 + adv.expected_change), now_val * (1 + adv.upside95)],
        "vs. converting now": [f"{adv.downside5:+.2%}", f"{adv.median_change:+.2%}",
                               f"{adv.expected_change:+.2%}", f"{adv.upside95:+.2%}"],
    })
    tbl[f"{quote} received"] = tbl[f"{quote} received"].map(lambda v: f"{v:,.2f}")
    st.dataframe(tbl, hide_index=True, width="stretch")
    st.subheader("Why")
    for r in adv.reasons:
        st.markdown(f"- {r}")

with right:
    st.subheader("How confident is the forecast?")
    st.progress(int(conf.score), text=f"{conf.label} confidence - {conf.score:.0f}/100")
    cc = st.columns(3)
    cc[0].metric("Signal", f"{conf.signal:.0%}", help="How lopsided today's forecast is.")
    cc[1].metric("Track record", f"{conf.reliability:.0%}", help="Did past direction calls beat a coin flip?")
    cc[2].metric("Calibration", f"{conf.calibration:.0%}", help="Did past 80% ranges hold ~80% of outcomes?")
    for n in conf.notes:
        st.markdown(f"- {n}")

    pf = go.Figure(go.Scatter(x=fc.dates, y=fc.p_higher_by_day, mode="lines", line=dict(color="#1565c0")))
    pf.add_hline(y=0.5, line=dict(color="#999", dash="dot"))
    pf.update_layout(height=220, margin=dict(l=10, r=10, t=30, b=10),
                     title="Chance the rate is higher than today, by day",
                     yaxis=dict(range=[0, 1], tickformat=".0%"))
    st.plotly_chart(pf, width="stretch")

with st.expander("How the forecast works"):
    st.markdown(
        """
- **Model:** a Monte Carlo simulation of 5,000 possible paths. Daily moves use recent volatility (EWMA) that gradually
  reverts to the long-run level, shocks are resampled from this pair's own history, and a small, adjustable share of
  the recent trend is carried forward.
- **Confidence:** the same model is replayed over past dates. The score blends how lopsided the forecast is, whether
  past direction calls beat a coin flip, and whether past 80% ranges were honest.
- **Decision:** "wait" is suggested only if the chance of a better rate, the expected gain and the confidence score
  all clear their thresholds; otherwise waiting adds risk without enough expected reward.
- **Reality check:** exchange rates are very hard to predict. Expect Low or Moderate confidence most of the time, and
  treat the *range* as more informative than the median.
        """
    )
st.caption("Informational only, not financial advice. Real conversion rates include a spread/fees that differ by provider.")