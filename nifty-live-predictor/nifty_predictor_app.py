#!/usr/bin/env python3
"""
Nifty 15m Predictor v3 — calibrated min-move model + auto-refresh + live candles
Run:  streamlit run nifty_predictor_app.py
"""

from __future__ import annotations
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import yfinance as yf
    import joblib
except ImportError:
    st.error("Install: pip install yfinance scikit-learn joblib plotly streamlit")
    st.stop()

st.set_page_config(page_title="Nifty Live Predictor v3", page_icon="📈", layout="wide")

MODEL_DIRS = [
    Path(__file__).resolve().parent / "nifty15m_model_v3",
    Path(__file__).resolve().parent / "nifty15m_model_v2",
    Path(__file__).resolve().parent / "nifty15m_model",
    Path("/home/workdir/artifacts/nifty15m_model_v3"),
    Path("/home/workdir/artifacts/nifty15m_model_v2"),
    Path("/home/workdir/artifacts/nifty15m_model"),
]


class CalibratedModel:
    """Fallback class if unpickling needs it in namespace."""
    def __init__(self, base=None, iso=None, features=None):
        self.base = base
        self.iso = iso
        self.features = features or []
    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1]
        cal = self.iso.predict(raw)
        cal = np.clip(cal, 0.01, 0.99)
        return np.column_stack([1 - cal, cal])
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@st.cache_resource
def load_models():
    for p in MODEL_DIRS:
        meta_path = p / "meta.joblib"
        if not meta_path.exists():
            continue
        meta = joblib.load(meta_path)
        features = meta["features"]
        version = meta.get("version", 2)

        raw_path = p / "direction_model_raw.joblib"
        iso_path = p / "isotonic_calibrator.joblib"
        sess_path = p / "direction_models_by_session.joblib"
        legacy = p / "direction_model.joblib"

        models = {}
        # Session-specific boosters
        if sess_path.exists():
            try:
                models = joblib.load(sess_path)
            except Exception:
                models = {}

        # Global calibrated = raw + isotonic
        if raw_path.exists() and iso_path.exists():
            base = joblib.load(raw_path)
            iso = joblib.load(iso_path)
            models["ALL"] = CalibratedModel(base, iso, features)
        elif raw_path.exists():
            models["ALL"] = joblib.load(raw_path)
        elif legacy.exists():
            models["ALL"] = joblib.load(legacy)

        if not models:
            continue

        reg = None
        rp = p / "return_model.joblib"
        if rp.exists():
            try:
                reg = joblib.load(rp)
                if getattr(reg, "n_features_in_", None) != len(features):
                    reg = None
            except Exception:
                reg = None

        return models, reg, features, str(p), version, meta
    return None, None, None, None, None, None


def fetch_nifty_1m(n_bars: int = 90) -> pd.DataFrame:
    raw = yf.download("^NSEI", period="5d", interval="1m", progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("No 1m Nifty data (market closed or network).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    df = raw.reset_index()
    col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={col: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns and c.lower() in df.columns:
            df[c] = df[c.lower()]
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Datetime")
    return df.tail(n_bars).reset_index(drop=True)


def fetch_context() -> dict:
    ctx = {}
    nifty_d = yf.download("^NSEI", period="10d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(nifty_d.columns, pd.MultiIndex):
        nifty_d.columns = [c[0] for c in nifty_d.columns]
    nifty_d = nifty_d.dropna(subset=["Close"])
    if len(nifty_d) >= 2:
        prev_close = float(nifty_d["Close"].iloc[-2])
        today_open = float(nifty_d["Open"].iloc[-1])
        ctx["gap_pct"] = (today_open / prev_close - 1.0) * 100.0
    else:
        ctx["gap_pct"] = 0.0

    def last_chg(ticker, period="10d"):
        d = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d.dropna(subset=["Close"])
        if len(d) < 2:
            return None, 0.0, 0.5
        chg = float(d["Close"].iloc[-1] / d["Close"].iloc[-2] - 1) * 100
        pct = float(d["Close"].rank(pct=True).iloc[-1]) if len(d) > 5 else 0.5
        return float(d["Close"].iloc[-1]), chg, pct

    vix_c, vix_chg, vix_pct = last_chg("^INDIAVIX", "60d")
    ctx["vix_close"] = vix_c or 15.0
    ctx["vix_chg"] = vix_chg
    ctx["vix_pctile_60"] = vix_pct
    _, ctx["brent_chg"], _ = last_chg("BZ=F")
    _, ctx["banknifty_chg"], _ = last_chg("^NSEBANK")
    _, ctx["usdinr_chg"], _ = last_chg("INR=X")
    ctx["usd_strong"] = 1 if ctx.get("usdinr_chg", 0) > 0.3 else 0
    ctx["oil_spike"] = 1 if ctx.get("brent_chg", 0) > 2 else 0
    ctx["oil_drop"] = 1 if ctx.get("brent_chg", 0) < -1.5 else 0
    wd = datetime.now().weekday()
    ctx["days_to_expiry"] = float((1 - wd) % 7)
    ctx["is_expiry"] = 1 if wd == 1 else 0
    ctx["is_expiry_m1"] = 1 if wd == 0 else 0
    ctx["is_expiry_p1"] = 0
    ctx["in_event_window"] = 0
    ctx["is_event_day"] = 0
    ctx["nifty_bn_rs_chg"] = 0.0
    ctx["ATR_pctile"] = 0.5
    return ctx


def build_X(bars: pd.DataFrame, ctx: dict, feature_list: list):
    df = bars.copy().sort_values("Datetime").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Volume" not in df.columns:
        df["Volume"] = 1.0
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(1).clip(lower=1)

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_15"] = df["Close"].pct_change(15)
    df["ret_30"] = df["Close"].pct_change(30)
    df["ret_60"] = df["Close"].pct_change(60)
    for w in [10, 30, 60]:
        s = df["Close"].rolling(w).mean()
        df[f"dist_sma{w}"] = df["Close"] / s - 1
    df["high_15"] = df["High"].rolling(15).max()
    df["low_15"] = df["Low"].rolling(15).min()
    df["pos_in_15"] = (df["Close"] - df["low_15"]) / (df["high_15"] - df["low_15"] + 1e-9)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()

    # VWAP (session-style on the buffer)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["vwap"] = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["dist_vwap"] = df["Close"] / df["vwap"] - 1

    prev = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()
    df["rvol_15m"] = df["ret_1"].rolling(15).std() * np.sqrt(375)
    df["rvol_60m"] = df["ret_1"].rolling(60).std() * np.sqrt(375)
    df["range_pct"] = (df["High"] - df["Low"]) / df["Open"] * 100
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    df["CLV"] = (df["Close"] - df["Low"]) / hl
    body = (df["Close"] - df["Open"]).abs()
    df["body_pct"] = body / hl
    df["upper_wick_pct"] = (df["High"] - np.maximum(df["Open"], df["Close"])) / hl
    df["lower_wick_pct"] = (np.minimum(df["Open"], df["Close"]) - df["Low"]) / hl
    df["is_doji"] = (df["body_pct"] < 0.1).astype(int)
    df["is_pin_bar"] = (((df["lower_wick_pct"] > 0.6) & (df["body_pct"] < 0.3)) |
                        ((df["upper_wick_pct"] > 0.6) & (df["body_pct"] < 0.3))).astype(int)
    df["is_inside_bar"] = ((df["High"] <= df["High"].shift(1)) & (df["Low"] >= df["Low"].shift(1))).astype(int)
    df["is_outside_bar"] = ((df["High"] >= df["High"].shift(1)) & (df["Low"] <= df["Low"].shift(1))).astype(int)
    df["minute_of_day"] = df["Datetime"].dt.hour * 60 + df["Datetime"].dt.minute
    df["hour"] = df["Datetime"].dt.hour
    df["session_enc"] = np.where(df["minute_of_day"] <= 585, 2, np.where(df["minute_of_day"] >= 885, 0, 1))

    for k, v in ctx.items():
        df[k] = v
    df["gap_down_big"] = 1 if ctx.get("gap_pct", 0) <= -0.5 else 0
    df["gap_up_big"] = 1 if ctx.get("gap_pct", 0) >= 0.5 else 0

    row = df.iloc[[-1]]
    X = np.array([float(row[f].iloc[0]) if f in row.columns and pd.notna(row[f].iloc[0]) else 0.0
                  for f in feature_list], dtype=float).reshape(1, -1)
    return X, row, df


def session_name(mod: int) -> str:
    if mod <= 585:
        return "Open"
    if mod >= 885:
        return "Close"
    return "Mid"


def project_2h(last_close, proba_up, pred_ret_15m, atr_proxy, last_dt, steps=8):
    if abs(pred_ret_15m) < 1e-8:
        step = (proba_up - 0.5) * 2 * max(atr_proxy, last_close * 0.0003) / last_close
    else:
        step = pred_ret_15m
    decay = 0.85
    times, mid, upper, lower = [], [], [], []
    price = last_close
    cum_step = step
    for i in range(1, steps + 1):
        t = last_dt + timedelta(minutes=15 * i)
        price = price * (1 + cum_step)
        band = last_close * 0.0008 * np.sqrt(i)
        times.append(t)
        mid.append(price)
        upper.append(price + band)
        lower.append(price - band)
        cum_step *= decay
    return times, mid, upper, lower


def make_live_chart(hist, times, mid, upper, lower, signal, last_close):
    fig = go.Figure()
    h = hist.tail(75)
    fig.add_trace(go.Candlestick(
        x=h["Datetime"], open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"],
        name="Nifty 1m (live)",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    color = "#26a69a" if signal == "LONG" else ("#ef5350" if signal == "SHORT" else "#90a4ae")
    fig.add_trace(go.Scatter(
        x=times, y=mid, mode="lines+markers", name="2h projection",
        line=dict(color=color, width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=list(times) + list(times)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself", fillcolor="rgba(100,181,246,0.22)",
        line=dict(color="rgba(0,0,0,0)"), name="Uncertainty band", hoverinfo="skip",
    ))
    if len(h):
        fig.add_vline(x=h["Datetime"].iloc[-1], line_dash="dot", line_color="#b0bec5",
                      annotation_text="NOW", annotation_position="top")
    fig.add_hline(y=last_close, line_dash="dot", line_color="#78909c", opacity=0.5)
    fig.update_layout(
        title=f"Live Nifty 1m · {last_close:,.2f} · {signal}",
        xaxis_rangeslider_visible=False, template="plotly_dark", height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=50, r=20, t=70, b=40), uirevision="live_chart",
    )
    return fig


def run_prediction(n_bars, models, reg, features):
    bars = fetch_nifty_1m(n_bars)
    ctx = fetch_context()
    X, row, _ = build_X(bars, ctx, features)
    mod = int(row["minute_of_day"].iloc[0]) if "minute_of_day" in row.columns else 600
    sess = session_name(mod)
    model = models.get(sess) or models.get("ALL") or list(models.values())[0]
    proba_up = float(model.predict_proba(X)[0, 1])
    pred_ret = float(reg.predict(X)[0]) if reg is not None else 0.0
    last_close = float(row["Close"].iloc[0])
    last_dt = pd.to_datetime(row["Datetime"].iloc[0])
    atr_proxy = float(row["ATR_14"].iloc[0]) if "ATR_14" in row.columns and pd.notna(row["ATR_14"].iloc[0]) else last_close * 0.001

    if proba_up >= 0.58:
        signal, conf = "LONG", "HIGH"
    elif proba_up <= 0.42:
        signal, conf = "SHORT", "HIGH"
    elif proba_up >= 0.53:
        signal, conf = "LONG", "LOW"
    elif proba_up <= 0.47:
        signal, conf = "SHORT", "LOW"
    else:
        signal, conf = "NEUTRAL", "NONE"

    times, mid, upper, lower = project_2h(last_close, proba_up, pred_ret, atr_proxy, last_dt, steps=8)
    return {
        "bars": bars, "ctx": ctx, "proba_up": proba_up, "pred_ret": pred_ret,
        "last_close": last_close, "last_dt": last_dt, "sess": sess,
        "signal": signal, "conf": conf, "times": times, "mid": mid, "upper": upper, "lower": lower,
    }


# ========================= UI =========================
st.title("📈 Nifty Live Predictor v3")
st.caption("Min-move calibrated model · session-aware · auto-fetch 1m + gap/VIX/Brent/BankNifty · live chart · 2h projection")

models, reg, features, model_path, version, meta = load_models()
if models is None:
    st.error("Model not found. Place `nifty15m_model_v3` (or v2) next to this script.")
    st.stop()

min_move = meta.get("min_move", 0) if meta else 0
st.sidebar.markdown(f"**Model:** v{version}")
st.sidebar.markdown(f"**Path:** `{model_path}`")
if min_move:
    st.sidebar.markdown(f"**Min-move target:** {min_move*100:.2f}%")
st.sidebar.markdown(f"**Features:** {len(features)}")
st.sidebar.markdown("Prefer **HIGH** confidence · Close session historically strongest")

c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.5])
with c1:
    n_bars = st.slider("1m bars", 60, 120, 90, 5)
with c2:
    auto = st.toggle("🔄 Auto-refresh", value=False)
with c3:
    interval = st.selectbox("Refresh every", [30, 60, 90, 120], index=1, format_func=lambda x: f"{x} sec")
with c4:
    st.write("")
    manual = st.button("🚀 Predict now", type="primary", use_container_width=True)

if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_run_ts" not in st.session_state:
    st.session_state.last_run_ts = None

should_run = manual or auto
if should_run:
    do_fetch = True
    if auto and not manual and st.session_state.last_run_ts is not None:
        if time.time() - st.session_state.last_run_ts < interval:
            do_fetch = False
    if do_fetch:
        with st.spinner("Fetching live 1m Nifty + context..."):
            try:
                result = run_prediction(n_bars, models, reg, features)
                st.session_state.last_result = result
                st.session_state.last_run_ts = time.time()
            except Exception as e:
                st.error(f"Failed: {e}")
                st.exception(e)
    result = st.session_state.last_result
else:
    result = st.session_state.last_result

if result is None:
    st.info("Click **Predict now** or enable **Auto-refresh** for the live candle graph.")
else:
    r = result
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nifty", f"{r['last_close']:,.2f}")
    m2.metric("Signal", r["signal"])
    m3.metric("Confidence", r["conf"])
    m4.metric("P(UP) 15m", f"{r['proba_up']:.1%}")
    m5.metric("Session", r["sess"])

    ts = datetime.fromtimestamp(st.session_state.last_run_ts).strftime("%H:%M:%S") if st.session_state.last_run_ts else "—"
    if auto:
        st.success(f"🟢 LIVE · updated {ts} · next ~{interval}s · model v{version}")
    else:
        st.caption(f"Last update {ts} · model v{version}")

    if r["conf"] == "HIGH":
        if r["signal"] == "LONG":
            st.success("HIGH confidence LONG bias for next ~15 minutes (meaningful move filter).")
        elif r["signal"] == "SHORT":
            st.error("HIGH confidence SHORT bias for next ~15 minutes (meaningful move filter).")
    elif r["signal"] == "NEUTRAL":
        st.warning("NEUTRAL — no clear edge; stand aside.")

    st.plotly_chart(
        make_live_chart(r["bars"], r["times"], r["mid"], r["upper"], r["lower"], r["signal"], r["last_close"]),
        use_container_width=True,
    )

    with st.expander("Context (auto-fetched)", expanded=True):
        x1, x2, x3, x4, x5 = st.columns(5)
        x1.write(f"**Gap** {r['ctx'].get('gap_pct', 0):+.3f}%")
        x2.write(f"**VIX** {r['ctx'].get('vix_close', 0):.2f} ({r['ctx'].get('vix_chg', 0):+.2f}%)")
        x3.write(f"**Brent** {r['ctx'].get('brent_chg', 0):+.2f}%")
        x4.write(f"**BankNifty** {r['ctx'].get('banknifty_chg', 0):+.2f}%")
        x5.write(f"**USDINR** {r['ctx'].get('usdinr_chg', 0):+.2f}%")

    proj = pd.DataFrame({
        "Time": r["times"],
        "Projected": [round(x, 2) for x in r["mid"]],
        "Lower": [round(x, 2) for x in r["lower"]],
        "Upper": [round(x, 2) for x in r["upper"]],
    })
    st.subheader("Next 2 hours · 15-minute steps")
    st.dataframe(proj, use_container_width=True, hide_index=True)
    st.caption("v3 uses min-move labels (≥0.05%) + calibration. Prefer HIGH confidence. Not financial advice.")

if auto:
    time.sleep(interval)
    st.rerun()
