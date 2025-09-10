import os
import json
import time
import math
import urllib.parse as urlparse
from datetime import datetime

import requests
import streamlit as st
import pandas as pd

# ===============================
# Streamlit config
# ===============================
st.set_page_config(page_title="Tastytrade: Put Credit Spread Scanner (Preview Only)", layout="wide")
st.title("🧃 Tastytrade Put Credit Spread Scanner — Preview Only")

# ===============================
# Secrets / Environment
# ===============================
TT_CLIENT_ID = st.secrets.get("TT_CLIENT_ID", os.getenv("TT_CLIENT_ID", ""))
TT_CLIENT_SECRET = st.secrets.get("TT_CLIENT_SECRET", os.getenv("TT_CLIENT_SECRET", ""))
TT_REDIRECT_URI = st.secrets.get("TT_REDIRECT_URI", os.getenv("TT_REDIRECT_URI", "http://localhost:8501/"))

env = st.sidebar.radio("Environment", ["Sandbox (cert)", "Production"], index=0)
if env.startswith("Sandbox"):
    BASE_URL = "https://api.cert.tastyworks.com"
    AUTH_URL = "https://cert-my.staging-tasty.works/auth.html"
else:
    BASE_URL = "https://api.tastyworks.com"
    AUTH_URL = "https://my.tastytrade.com/auth.html"
TOKEN_URL = f"{BASE_URL}/oauth/token"

# Session state for OAuth and preferences
if "oauth" not in st.session_state:
    st.session_state["oauth"] = {
        "access_token": None,
        "refresh_token": None,
        "expires_at": None,
        "token_type": "Bearer",
        "account_number": None
    }

def set_user_agent(headers: dict | None = None) -> dict:
    h = {} if headers is None else dict(headers)
    h["User-Agent"] = "tt-pcs-analyzer/1.1"
    return h

def auth_header() -> dict:
    tok = st.session_state["oauth"]["access_token"]
    if not tok:
        return {}
    return set_user_agent({"Authorization": f"Bearer {tok}", "Accept": "application/json", "Content-Type": "application/json"})

def save_tokens(token_resp: dict):
    st.session_state["oauth"]["access_token"] = token_resp.get("access_token")
    st.session_state["oauth"]["refresh_token"] = token_resp.get("refresh_token")
    st.session_state["oauth"]["token_type"] = token_resp.get("token_type", "Bearer")
    exp_in = int(token_resp.get("expires_in", 900))
    st.session_state["oauth"]["expires_at"] = int(time.time()) + exp_in - 15

def token_expired() -> bool:
    exp = st.session_state["oauth"]["expires_at"]
    return not exp or time.time() >= exp

def refresh_access_token() -> bool:
    rtok = st.session_state["oauth"]["refresh_token"]
    if not rtok:
        return False
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": rtok,
        "client_secret": TT_CLIENT_SECRET
    }
    try:
        r = requests.post(TOKEN_URL, json=payload, headers=set_user_agent({"Accept":"application/json","Content-Type":"application/json"}), timeout=20)
        if r.status_code == 200:
            save_tokens(r.json())
            return True
        else:
            st.warning(f"Token refresh failed ({r.status_code}): {r.text}")
            return False
    except Exception as e:
        st.error(f"Refresh exception: {e}")
        return False

def ensure_token() -> bool:
    if st.session_state["oauth"]["access_token"] and not token_expired():
        return True
    if st.session_state["oauth"]["refresh_token"]:
        return refresh_access_token()
    return False

# ===============================
# 1) OAuth2 – connect account (for data only)
# ===============================
st.subheader("1) Connect your Tastytrade account (for data only)")

missing = []
if not TT_CLIENT_ID: missing.append("TT_CLIENT_ID")
if not TT_CLIENT_SECRET: missing.append("TT_CLIENT_SECRET")
if not TT_REDIRECT_URI: missing.append("TT_REDIRECT_URI")
if missing:
    st.warning("Missing secrets: " + ", ".join(missing))

params = {
    "client_id": TT_CLIENT_ID,
    "redirect_uri": TT_REDIRECT_URI,
    "response_type": "code",
    "scope": "read openid",   # read-only
    "state": "pcs_preview_only"
}
auth_link = AUTH_URL + "?" + urlparse.urlencode(params, doseq=True)
st.markdown(f"[Authorize with Tastytrade]({auth_link})")

parsed_query = st.query_params
auth_code = parsed_query.get("code", [None])[0] if isinstance(parsed_query.get("code"), list) else parsed_query.get("code")
manual_redirect = st.text_input("Or paste the FULL redirected URL (must contain ?code=...)", "")

if manual_redirect and "code=" in manual_redirect:
    try:
        q = urlparse.urlparse(manual_redirect)
        qs = urlparse.parse_qs(q.query)
        auth_code = qs.get("code", [None])[0]
    except Exception as e:
        st.error(f"Could not parse URL: {e}")

if st.button("Exchange Code for Tokens", disabled=not auth_code or not (TT_CLIENT_ID and TT_CLIENT_SECRET and TT_REDIRECT_URI)):
    data = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "client_id": TT_CLIENT_ID,
        "client_secret": TT_CLIENT_SECRET,
        "redirect_uri": TT_REDIRECT_URI
    }
    r = requests.post(TOKEN_URL, json=data, headers=set_user_agent({"Accept":"application/json","Content-Type":"application/json"}), timeout=20)
    if r.status_code == 200:
        save_tokens(r.json())
        st.success("Access token obtained. You’re connected.")
    else:
        st.error(f"Token exchange failed ({r.status_code}): {r.text}")

# ===============================
# 2) Scanner controls
# ===============================
st.subheader("2) Scanner Settings")
symbols = st.text_input("Symbols (comma-separated)", "AAPL, MSFT, AMD").upper().replace(" ", "")

scan_mode = st.radio("Expiry window", ["Weekly (7–10 DTE)", "Bi-weekly (14–20 DTE)", "Custom"], index=0)
if scan_mode == "Weekly (7–10 DTE)":
    dte_min, dte_max = 7, 10
elif scan_mode == "Bi-weekly (14–20 DTE)":
    dte_min, dte_max = 14, 20
else:
    dte_min, dte_max = st.slider("Custom DTE range", 0, 90, (7, 20))

min_credit = st.number_input("Min net credit ($)", min_value=0.00, step=0.05, value=0.20, format="%.2f")
max_width = st.number_input("Max spread width ($)", min_value=0.5, step=0.5, value=5.0, format="%.2f")
limit_pairs_per_exp = st.slider("Max spreads per expiration (to limit combinations)", 5, 200, 50)
use_compact_chain = st.checkbox("Use compact chain endpoint (faster)", value=True)

st.markdown("**Delta targeting**")
target_delta = st.slider("Target short put |Δ|", 0.05, 0.50, 0.25, 0.01)
delta_band = st.slider("Acceptable |Δ| range around target", 0.00, 0.20, 0.05, 0.01)
min_pop = st.slider("Min POP (%)", 50, 90, 65)

st.caption("Note: This app **never** submits orders. It only analyzes and generates a downloadable idea list.")

# ===============================
# Utility math
# ===============================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pop_from_breakeven(S: float, be: float, iv_annual: float, dte: int) -> float | None:
    try:
        if S is None or iv_annual is None or be is None or dte <= 0 or iv_annual <= 0:
            return None
        T = dte / 365.0
        z = (math.log(be / S) + (-0.5 * (iv_annual ** 2) * T)) / (iv_annual * math.sqrt(T))
        prob = 1.0 - norm_cdf(z)
        return max(0.0, min(1.0, prob)) * 100.0
    except Exception:
        return None

# ===============================
# Data fetchers
# ===============================
def fetch_option_chain(symbol: str) -> dict | None:
    endpoint = f"{BASE_URL}/option-chains/compact/{symbol}" if use_compact_chain else f"{BASE_URL}/option-chains/{symbol}"
    try:
        r = requests.get(endpoint, headers=auth_header(), timeout=30)
        if r.status_code != 200:
            st.warning(f"{symbol}: chain error ({r.status_code}) {r.text[:200]}")
            return None
        return r.json().get("data", {})
    except Exception as e:
        st.warning(f"{symbol}: chain exception {e}")
        return None

def parse_puts_from_chain(chain: dict) -> pd.DataFrame:
    items = []
    underlying_price = chain.get("underlying-price") or chain.get("underlying_price")
    if "items" in chain:
        for it in chain["items"]:
            if (it.get("option-type") or it.get("option_type")) != "P":
                continue
            bid = it.get("bid"); ask = it.get("ask")
            mid = None
            if bid is not None and ask is not None:
                try: mid = (float(bid) + float(ask)) / 2.0
                except: pass
            items.append({
                "symbol": it.get("symbol"),
                "expiration": it.get("expiration-date") or it.get("expiration_date"),
                "strike": float(it.get("strike-price") or it.get("strike_price")),
                "bid": float(bid) if bid is not None else None,
                "ask": float(ask) if ask is not None else None,
                "mid": mid,
                "iv": (it.get("implied-volatility") or it.get("implied_volatility") or None),
                "delta": (it.get("delta") or None),
                "underlying_price": underlying_price
            })
    else:
        for exp in chain.get("expirations", []):
            exp_date = exp.get("expiration-date") or exp.get("expiration_date")
            for s in exp.get("strikes", []):
                if (s.get("option-type") or s.get("option_type")) != "P":
                    continue
                bid = s.get("bid"); ask = s.get("ask")
                mid = None
                if bid is not None and ask is not None:
                    try: mid = (float(bid) + float(ask)) / 2.0
                    except: pass
                items.append({
                    "symbol": s.get("symbol"),
                    "expiration": exp_date,
                    "strike": float(s.get("strike-price") or s.get("strike_price")),
                    "bid": float(bid) if bid is not None else None,
                    "ask": float(ask) if ask is not None else None,
                    "mid": mid,
                    "iv": (s.get("implied-volatility") or s.get("implied_volatility") or None),
                    "delta": (s.get("delta") or None),
                    "underlying_price": underlying_price
                })
    df = pd.DataFrame(items) if items else pd.DataFrame()
    if not df.empty:
        df["expiration"] = pd.to_datetime(df["expiration"])
    return df

def _extract_price_from_quote_payload(payload: dict) -> float | None:
    """
    Attempt to extract a reasonable underlying price from common quote payload shapes.
    Try last/mark/close/ask/bid, in that order.
    """
    if not payload:
        return None
    # common “data” -> “items”
    data = payload.get("data") or {}
    if isinstance(data, dict) and "items" in data:
        items = data.get("items") or []
        if items:
            q = items[0]
        else:
            q = {}
    else:
        q = data if isinstance(data, dict) else payload

    for k in ["last", "mark", "close", "ask", "bid"]:
        v = q.get(k) if isinstance(q, dict) else None
        try:
            if v is not None:
                return float(v)
        except:
            pass
    return None

def fetch_underlying_price(symbol: str) -> float | None:
    """
    Fetch underlying price via Tastytrade quotes. We try a few common endpoints
    to be robust across environments. We only read – no trading.
    """
    if not ensure_token():
        return None
    headers = auth_header()

    # 1) market-data/quotes (batch)
    try:
        url = f"{BASE_URL}/market-data/quotes"
        r = requests.get(url, params={"symbols": symbol}, headers=headers, timeout=15)
        if r.status_code == 200:
            px = _extract_price_from_quote_payload(r.json())
            if px is not None:
                return px
    except Exception:
        pass

    # 2) market-data/quotes/{symbol}
    try:
        url = f"{BASE_URL}/market-data/quotes/{symbol}"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            px = _extract_price_from_quote_payload(r.json())
            if px is not None:
                return px
    except Exception:
        pass

    # 3) fallback: market-metrics (some envs expose a last price)
    try:
        url = f"{BASE_URL}/market-metrics/{symbol}"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            px = _extract_price_from_quote_payload(r.json())
            if px is not None:
                return px
    except Exception:
        pass

    return None  # gracefully let UI handle manual entry if not found

# ===============================
# 3) Run scan (Preview only)
# ===============================
st.subheader("3) Run Scan")

run_scan = st.button("Run scan (data-only)")
results_rows = []

if run_scan:
    if not ensure_token():
        st.error("Please authorize first.")
    else:
        all_syms = [s for s in symbols.split(",") if s]
        for sym in all_syms:
            st.write(f"🔎 Scanning {sym} …")
            chain = fetch_option_chain(sym)
            if not chain:
                continue
            df_puts = parse_puts_from_chain(chain)
            if df_puts.empty:
                st.warning(f"{sym}: no puts parsed.")
                continue

            # Underlying price — from chain or quotes
            u_price = df_puts["underlying_price"].dropna().iloc[0] if df_puts["underlying_price"].notna().any() else None
            if u_price is None:
                u_price = fetch_underlying_price(sym)

            today = pd.Timestamp.utcnow().normalize()
            df_puts["DTE"] = (df_puts["expiration"] - today).dt.days
            df_puts = df_puts[(df_puts["DTE"] >= dte_min) & (df_puts["DTE"] <= dte_max)].copy()
            if df_puts.empty:
                st.info(f"{sym}: no expirations in DTE window {dte_min}-{dte_max}.")
                continue

            df_puts = df_puts.sort_values(["expiration", "strike"]).reset_index(drop=True)

            # For each expiration, build and rank delta-targeted candidates
            for exp_date, df_e in df_puts.groupby("expiration"):
                df_e = df_e.copy()

                # If delta present, filter around |delta| target; else use OTM strikes below price
                if df_e["delta"].notna().any():
                    df_e["abs_delta"] = df_e["delta"].abs()
                    lower, upper = max(0.0, target_delta - delta_band), min(1.0, target_delta + delta_band)
                    cand_short = df_e[(df_e["abs_delta"] >= lower) & (df_e["abs_delta"] <= upper)].copy()
                    if cand_short.empty:
                        # widen once if nothing found
                        widen = min(0.25, delta_band * 2 if delta_band > 0 else 0.1)
                        cand_short = df_e[(df_e["abs_delta"] >= target_delta - widen) & (df_e["abs_delta"] <= target_delta + widen)].copy()
                    cand_short["delta_distance"] = (cand_short["abs_delta"] - target_delta).abs()
                else:
                    # fallback heuristic if no Greeks
                    if u_price is None:
                        u_price = st.number_input(f"{sym} underlying price (not found via API)", min_value=0.01, step=0.01, value=100.00, key=f"px_{sym}")
                    cand_short = df_e[df_e["strike"] < u_price].copy()
                    cand_short["delta_distance"] = 0.5  # neutral placeholder when no deltas

                cand_short = cand_short.sort_values(["delta_distance", "strike"], ascending=[True, False]).head(limit_pairs_per_exp)

                for _, srow in cand_short.iterrows():
                    short_mid = srow["mid"] if pd.notna(srow["mid"]) else None
                    if short_mid is None:
                        continue
                    short_bid = srow["bid"]; short_ask = srow["ask"]
                    short_strike = srow["strike"]
                    iv_short = float(srow["iv"]) if srow["iv"] is not None else None
                    DTE = int(srow["DTE"])

                    # candidate long strikes below short, limited by width
                    df_long = df_e[df_e["strike"] < short_strike].copy()
                    df_long = df_long[df_long["strike"] >= short_strike - max_width]
                    df_long = df_long.sort_values("strike", ascending=False).head(8)

                    for _, lrow in df_long.iterrows():
                        long_mid = lrow["mid"] if pd.notna(lrow["mid"]) else None
                        if long_mid is None:
                            continue
                        long_ask = lrow["ask"]
                        long_strike = lrow["strike"]

                        credit = round(float(short_mid) - float(long_mid), 2)
                        if credit < float(min_credit):
                            continue

                        width = round(short_strike - long_strike, 2)
                        if width <= 0:
                            continue

                        max_loss = round(width - credit, 2)
                        if max_loss <= 0:
                            continue

                        roi = round(credit / max_loss * 100.0, 2)
                        breakeven = round(short_strike - credit, 2)

                        # IV fallback: median IV of expiration
                        iv = iv_short
                        if iv is None:
                            med_iv = pd.to_numeric(df_e["iv"], errors="coerce").median()
                            iv = float(med_iv) if pd.notna(med_iv) else None
                        if iv is not None and iv > 1.0:
                            iv = iv / 100.0

                        pop = pop_from_breakeven(u_price, breakeven, iv, DTE) if (u_price is not None and iv is not None) else None
                        if pop is not None and pop < float(min_pop):
                            continue
                        loss_prob = round(100.0 - pop, 2) if pop is not None else None

                        rr_txt = f"{credit:.2f} : {max_loss:.2f}"
                        link = f"https://my.tastytrade.com/symbols/{sym}" if AUTH_URL.endswith("auth.html") else "https://my.tastytrade.com"

                        results_rows.append({
                            "Symbol": sym,
                            "Price": round(float(u_price), 2) if u_price else None,
                            "Exp Date": exp_date.date().isoformat(),
                            "DTE": DTE,
                            "Short": f"{short_strike:.2f}P",
                            "Bid1": round(float(short_bid), 2) if short_bid is not None else None,
                            "Long": f"{long_strike:.2f}P",
                            "Ask2": round(float(long_ask), 2) if long_ask is not None else None,
                            "Credit": credit,
                            "Width": width,
                            "Max Profit": round(credit, 2),
                            "Max Loss": max_loss,
                            "Max Profit%": roi,
                            "BE": breakeven,
                            "BE%": round((breakeven - u_price) / u_price * 100.0, 2) if u_price else None,
                            "IV (short)": round(iv * 100.0, 2) if iv is not None else None,
                            "POP%": round(pop, 2) if pop is not None else None,
                            "Loss Prob%": loss_prob,
                            "Short |Δ|": round(abs(srow["delta"]), 3) if srow.get("delta") is not None else None,
                            "Δ distance": round(abs((abs(srow.get("delta")) if srow.get("delta") is not None else target_delta) - target_delta), 3),
                            "Risk/Reward": rr_txt,
                            "Links": link
                        })

# ===============================
# 4) Results table + downloads
# ===============================
st.subheader("4) Results (Preview Only)")

if results_rows:
    res_df = pd.DataFrame(results_rows)

    # Ranking: closest to target delta, higher POP, then ROI, then credit
    by = [c for c in ["Δ distance", "POP%", "Max Profit%", "Credit"] if c in res_df.columns]
    asc = [True, False, False, False][:len(by)]
    if by:
        res_df = res_df.sort_values(by=by, ascending=asc, na_position="last").reset_index(drop=True)

    top_n = st.slider("Show top N", 10, 500, min(100, len(res_df)))
    st.dataframe(res_df.head(top_n))

    csv_bytes = res_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name=f"pcs_scan_{datetime.utcnow().date().isoformat()}.csv", mime="text/csv")

    json_bytes = res_df.to_json(orient="records", indent=2).encode("utf-8")
    st.download_button("⬇️ Download JSON", data=json_bytes, file_name=f"pcs_scan_{datetime.utcnow().date().isoformat()}.json", mime="application/json")

    st.success("Preview-only analyzer. No orders are sent to Tastytrade.")
else:
    st.caption("Run a scan to see results here.")
