#!/usr/bin/env python3
"""
Nifty 15-minute live predictor with AUTO data fetch
====================================================
Automatically pulls:
  - Last 60–120 one-minute Nifty bars (Yahoo / NSE-compatible)
  - Context: gap_pct, vix_close, vix_chg, brent_chg, banknifty_chg

Usage:
  python live_predict_15m.py              # one-shot prediction now
  python live_predict_15m.py --loop 60    # refresh every 60 seconds
  python live_predict_15m.py --bars 90    # use last 90 minute bars

Optional broker (higher quality 1m during market hours):
  Set env ZERODHA_ENCTOKEN or pass --enctoken for Kite-style fetch (if pyzdata installed)
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("pip install yfinance")

try:
    import joblib
except ImportError:
    raise SystemExit("pip install joblib scikit-learn")

# ---------- model paths ----------
ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    ROOT / "nifty15m_model_v2",
    ROOT / "nifty15m_model",
    Path("/home/workdir/artifacts/nifty15m_model_v2"),
    Path("/home/workdir/artifacts/nifty15m_model"),
]

def find_model_dir() -> Path:
    for p in MODEL_CANDIDATES:
        if (p / "direction_model.joblib").exists() or (p / "direction_models_by_session.joblib").exists():
            return p
    raise FileNotFoundError("Model folder not found. Place nifty15m_model_v2 next to this script.")


# ---------- data fetch ----------
def fetch_nifty_1m(n_bars: int = 90) -> pd.DataFrame:
    """Last n_bars of Nifty 50 1-minute OHLC via Yahoo (works for recent sessions)."""
    # 5d of 1m is enough to cover n_bars even with gaps
    raw = yf.download("^NSEI", period="5d", interval="1m", progress=False, auto_adjust=False)
    if raw is None or len(raw) == 0:
        raise RuntimeError("No 1m Nifty data from Yahoo. Market closed or network issue.")

    # flatten multiindex columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.reset_index()
    # timezone-aware -> naive IST-like wall clock
    col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={col: "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            # try lowercase
            lc = c.lower()
            if lc in df.columns:
                df[c] = df[lc]
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Datetime")
    df = df.tail(n_bars).reset_index(drop=True)
    return df[["Datetime", "Open", "High", "Low", "Close"] + (["Volume"] if "Volume" in df.columns else [])]


def fetch_context() -> dict:
    """
    gap_pct, vix_close, vix_chg, brent_chg, banknifty_chg
    """
    ctx = {}

    # --- Nifty daily for gap ---
    nifty_d = yf.download("^NSEI", period="10d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(nifty_d.columns, pd.MultiIndex):
        nifty_d.columns = [c[0] for c in nifty_d.columns]
    nifty_d = nifty_d.dropna(subset=["Close"])
    if len(nifty_d) >= 2:
        prev_close = float(nifty_d["Close"].iloc[-2])
        # today's open if available else last close as proxy
        today_open = float(nifty_d["Open"].iloc[-1]) if "Open" in nifty_d.columns else float(nifty_d["Close"].iloc[-1])
        ctx["gap_pct"] = (today_open / prev_close - 1.0) * 100.0
        ctx["prev_day_close"] = prev_close
    else:
        ctx["gap_pct"] = 0.0

    # --- India VIX ---
    vix = yf.download("^INDIAVIX", period="60d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = [c[0] for c in vix.columns]
    vix = vix.dropna(subset=["Close"])
    if len(vix) >= 2:
        ctx["vix_close"] = float(vix["Close"].iloc[-1])
        ctx["vix_chg"] = float(vix["Close"].iloc[-1] / vix["Close"].iloc[-2] - 1.0) * 100.0
        # rough 60d percentile
        ctx["vix_pctile_60"] = float(vix["Close"].rank(pct=True).iloc[-1])
    else:
        ctx["vix_close"] = 15.0
        ctx["vix_chg"] = 0.0
        ctx["vix_pctile_60"] = 0.5

    # --- Brent ---
    brent = yf.download("BZ=F", period="10d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(brent.columns, pd.MultiIndex):
        brent.columns = [c[0] for c in brent.columns]
    brent = brent.dropna(subset=["Close"])
    if len(brent) >= 2:
        ctx["brent_chg"] = float(brent["Close"].iloc[-1] / brent["Close"].iloc[-2] - 1.0) * 100.0
        ctx["brent_close"] = float(brent["Close"].iloc[-1])
    else:
        ctx["brent_chg"] = 0.0

    # --- Bank Nifty ---
    bn = yf.download("^NSEBANK", period="10d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(bn.columns, pd.MultiIndex):
        bn.columns = [c[0] for c in bn.columns]
    bn = bn.dropna(subset=["Close"])
    if len(bn) >= 2:
        ctx["banknifty_chg"] = float(bn["Close"].iloc[-1] / bn["Close"].iloc[-2] - 1.0) * 100.0
        ctx["banknifty_close"] = float(bn["Close"].iloc[-1])
    else:
        ctx["banknifty_chg"] = 0.0

    # --- USDINR ---
    usd = yf.download("INR=X", period="10d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(usd.columns, pd.MultiIndex):
        usd.columns = [c[0] for c in usd.columns]
    usd = usd.dropna(subset=["Close"])
    if len(usd) >= 2:
        ctx["usdinr_chg"] = float(usd["Close"].iloc[-1] / usd["Close"].iloc[-2] - 1.0) * 100.0
    else:
        ctx["usdinr_chg"] = 0.0

    # flags used by model
    ctx["usd_strong"] = 1 if ctx.get("usdinr_chg", 0) > 0.3 else 0
    ctx["oil_spike"] = 1 if ctx.get("brent_chg", 0) > 2.0 else 0
    ctx["oil_drop"] = 1 if ctx.get("brent_chg", 0) < -1.5 else 0

    # expiry helpers (Tuesday-based; approximate)
    today = datetime.now().date()
    # days to next Tuesday
    wd = today.weekday()  # Mon=0
    days_to_tue = (1 - wd) % 7
    if days_to_tue == 0:
        days_to_tue = 0  # today is Tuesday
    ctx["days_to_expiry"] = float(days_to_tue)
    ctx["is_expiry"] = 1 if wd == 1 else 0
    ctx["is_expiry_m1"] = 1 if wd == 0 else 0  # Monday

    return ctx


# ---------- feature build + predict (aligned with training) ----------
def build_features_latest(bars: pd.DataFrame, ctx: dict, feature_list: list) -> np.ndarray:
    df = bars.copy().sort_values("Datetime").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_15"] = df["Close"].pct_change(15)
    df["ret_30"] = df["Close"].pct_change(30)
    df["ret_60"] = df["Close"].pct_change(60)
    for w in [10, 30, 60]:
        sma = df["Close"].rolling(w).mean()
        df[f"dist_sma{w}"] = df["Close"] / sma - 1
    df["high_15"] = df["High"].rolling(15).max()
    df["low_15"] = df["Low"].rolling(15).min()
    df["pos_in_15"] = (df["Close"] - df["low_15"]) / (df["high_15"] - df["low_15"] + 1e-9)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()

    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
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
    df["session_enc"] = np.where(df["minute_of_day"] <= 9 * 60 + 45, 2,
                                 np.where(df["minute_of_day"] >= 14 * 60 + 45, 0, 1))

    # inject context onto every row (daily fields)
    for k, v in ctx.items():
        df[k] = v
    df["gap_down_big"] = 1 if ctx.get("gap_pct", 0) <= -0.5 else 0
    df["gap_up_big"] = 1 if ctx.get("gap_pct", 0) >= 0.5 else 0
    if "ATR_pctile" not in df.columns:
        df["ATR_pctile"] = 0.5
    if "in_event_window" not in df.columns:
        df["in_event_window"] = 0
    if "is_event_day" not in df.columns:
        df["is_event_day"] = 0
    if "nifty_bn_rs_chg" not in df.columns:
        df["nifty_bn_rs_chg"] = 0.0
    if "is_expiry_p1" not in df.columns:
        df["is_expiry_p1"] = 0

    row = df.iloc[[-1]]
    # align to training feature order; missing -> 0
    X = []
    for f in feature_list:
        if f in row.columns and pd.notna(row[f].iloc[0]):
            X.append(float(row[f].iloc[0]))
        else:
            X.append(0.0)
    return np.array(X, dtype=float).reshape(1, -1), row


def predict_live(n_bars: int = 90) -> dict:
    model_dir = find_model_dir()
    meta = joblib.load(model_dir / "meta.joblib")
    features = meta["features"]

    # load session models if present else global
    sess_path = model_dir / "direction_models_by_session.joblib"
    if sess_path.exists():
        models = joblib.load(sess_path)
    else:
        models = {"ALL": joblib.load(model_dir / "direction_model.joblib")}

    reg_path = model_dir / "return_model.joblib"
    reg = None
    if reg_path.exists():
        try:
            reg = joblib.load(reg_path)
            # only use if n_features matches
            n_expected = getattr(reg, "n_features_in_", None)
            if n_expected is not None and n_expected != len(features):
                print(f"  (return model expects {n_expected} feats, have {len(features)} — skipping return pred)")
                reg = None
        except Exception:
            reg = None

    print("Fetching Nifty 1m bars...")
    bars = fetch_nifty_1m(n_bars)
    print(f"  got {len(bars)} bars | {bars['Datetime'].iloc[0]} → {bars['Datetime'].iloc[-1]}")

    print("Fetching context (gap, VIX, Brent, BankNifty)...")
    ctx = fetch_context()
    print(f"  gap_pct={ctx.get('gap_pct', 0):+.3f}%  VIX={ctx.get('vix_close', 0):.2f} ({ctx.get('vix_chg', 0):+.2f}%)  "
          f"Brent={ctx.get('brent_chg', 0):+.2f}%  BN={ctx.get('banknifty_chg', 0):+.2f}%")

    X, row = build_features_latest(bars, ctx, features)

    # pick session model
    mod = int(row["minute_of_day"].iloc[0]) if "minute_of_day" in row.columns else 600
    if mod <= 9 * 60 + 45:
        sess_name = "Open"
    elif mod >= 14 * 60 + 45:
        sess_name = "Close"
    else:
        sess_name = "Mid"
    model = models.get(sess_name) or models.get("ALL") or list(models.values())[0]

    proba_up = float(model.predict_proba(X)[0, 1])
    pred_ret = float(reg.predict(X)[0]) if reg is not None else 0.0

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

    out = {
        "as_of": str(row["Datetime"].iloc[0]),
        "close": float(row["Close"].iloc[0]),
        "session": sess_name,
        "proba_up_15m": round(proba_up, 4),
        "pred_direction": "UP" if proba_up >= 0.5 else "DOWN",
        "pred_return_15m_pct": round(pred_ret * 100, 4),
        "signal": signal,
        "confidence": conf,
        "context": {
            "gap_pct": round(ctx.get("gap_pct", 0), 3),
            "vix_close": round(ctx.get("vix_close", 0), 2),
            "vix_chg": round(ctx.get("vix_chg", 0), 2),
            "brent_chg": round(ctx.get("brent_chg", 0), 2),
            "banknifty_chg": round(ctx.get("banknifty_chg", 0), 2),
            "usdinr_chg": round(ctx.get("usdinr_chg", 0), 2),
            "days_to_expiry": ctx.get("days_to_expiry"),
            "is_expiry": ctx.get("is_expiry"),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser(description="Auto-fetch Nifty 1m + context → 15m prediction")
    ap.add_argument("--bars", type=int, default=90, help="Number of 1m bars to feed (60-120)")
    ap.add_argument("--loop", type=int, default=0, help="If >0, refresh every N seconds")
    args = ap.parse_args()
    n_bars = max(60, min(120, args.bars))

    def run_once():
        try:
            r = predict_live(n_bars)
            print("\n========== 15-MINUTE PREDICTION ==========")
            print(f"  As of:        {r['as_of']}  (session={r['session']})")
            print(f"  Nifty close:  {r['close']:.2f}")
            print(f"  Proba UP:     {r['proba_up_15m']:.1%}")
            print(f"  Direction:    {r['pred_direction']}")
            print(f"  Pred ret 15m: {r['pred_return_15m_pct']:+.3f}%")
            print(f"  SIGNAL:       {r['signal']}  [{r['confidence']}]")
            print("  Context:")
            for k, v in r["context"].items():
                print(f"    {k}: {v}")
            print("==========================================\n")
            return r
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return None

    if args.loop and args.loop > 0:
        print(f"Looping every {args.loop}s. Ctrl+C to stop.")
        while True:
            run_once()
            time.sleep(args.loop)
    else:
        run_once()


if __name__ == "__main__":
    main()
