# Nifty Live Predictor v3

15-minute direction model for Nifty 50 with live 1-minute candles, auto-fetched context (gap, India VIX, Brent, Bank Nifty), and a 2-hour projected path.

**Not financial advice.** For research and education only. Past accuracy does not guarantee future results.

---

## Features

- **One-click predict** — fetches last 60–120 Nifty 1-minute bars
- **Auto context** — gap %, India VIX, Brent crude, Bank Nifty, USDINR
- **v3 model** — min-move labels (≥0.05%), probability calibration, session-aware models
- **Live chart** — candlesticks + dashed 2-hour projection + uncertainty band
- **Auto-refresh** — optional 30–120 second updates during market hours

---

## Quick start (local)

```bash
pip install -r requirements.txt
streamlit run nifty_predictor_app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (public repo recommended for free tier).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **New app** → select this repository.
4. Set:
   - **Branch:** `main`
   - **Main file path:** `nifty_predictor_app.py`
5. Click **Deploy**.

App URL will look like: `https://<app-name>.streamlit.app`

### Required repo layout

```text
.
├── nifty_predictor_app.py
├── requirements.txt
├── README.md
└── nifty15m_model_v3/
    ├── meta.joblib
    ├── direction_model_raw.joblib
    ├── isotonic_calibrator.joblib
    ├── direction_models_by_session.joblib
    └── direction_model_3class.joblib
```

---

## How to use

| Control | Purpose |
|---------|---------|
| **Predict now** | One-shot fetch + prediction + chart |
| **Auto-refresh** | Live mode; re-runs on a timer |
| **Refresh every** | 30 / 60 / 90 / 120 seconds |
| **1m bars** | How many recent minutes to feed the model (60–120) |

### Signal guide

| Signal | Confidence | Suggestion |
|--------|------------|------------|
| LONG / SHORT | **HIGH** | Stronger bias for next ~15 minutes |
| LONG / SHORT | LOW | Weak; optional / skip |
| NEUTRAL | NONE | Stand aside |

Prefer **HIGH** confidence. Close session (14:45–15:30 IST) was historically the strongest in backtests.

---

## Model notes (v3)

- **Target:** next 15-minute direction only when move ≥ ~0.05% (noise filtered)
- **Calibration:** isotonic regression on probabilities
- **Sessions:** separate models for Open / Mid / Close when available
- **Features:** returns, VWAP distance, ATR/vol, gap, VIX, macro, expiry flags
- **Typical accuracy (held-out):** ~59% overall; ~60–64% on high-confidence / Close session

The 2-hour path is an **extended scenario** from the 15-minute bias (with decay), not a guaranteed forecast.

---

## Data sources

- Nifty 1-minute OHLC: Yahoo Finance (`^NSEI`)
- India VIX, Bank Nifty, Brent, USDINR: Yahoo Finance

No API keys required. After market close, the last available session bars are used.

---

## Disclaimer

This tool is for **educational and research** purposes only. It is not investment advice. Markets are risky; you can lose money. Always use your own judgment and risk management.
