import os
import json
import time
import math
import urllib.parse as urlparse
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple, Union

import requests
import streamlit as st
import pandas as pd
import numpy as np
from requests.exceptions import RequestException, Timeout

# ===============================
# Streamlit config
# ===============================
st.set_page_config(
    page_title="Tastytrade: Put Credit Spread Scanner (Preview Only)", 
    layout="wide",
    page_icon="🧃"
)
st.title("🧃 Tastytrade Put Credit Spread Scanner — Preview Only")

# Add a disclaimer
st.warning("""
**Disclaimer**: This tool is for educational and informational purposes only. 
It does not constitute financial advice. Options trading involves substantial risk 
and is not suitable for all investors. Always do your own research and consider 
consulting with a qualified financial professional before making any investment decisions.
""")

# ===============================
# Constants and Configuration
# ===============================
DEFAULT_SYMBOLS = "SPY, QQQ, IWM, AAPL, MSFT, AMD, NVDA, TSLA"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 2
RETRY_DELAY = 1

# ===============================
# Secrets / Environment
# ===============================
TT_CLIENT_ID = st.secrets.get("TT_CLIENT_ID", os.getenv("TT_CLIENT_ID", ""))
TT_CLIENT_SECRET = st.secrets.get("TT_CLIENT_SECRET", os.getenv("TT_CLIENT_SECRET", ""))
TT_REDIRECT_URI = st.secrets.get("TT_REDIRECT_URI", os.getenv("TT_REDIRECT_URI", "http://localhost:8501/"))

env = st.sidebar.radio("Environment", ["Sandbox (cert)", "Production"], index=0)
if env.startswith("Sandbox"):
    BASE_URL = "https://api.cert.tastyworks.com"
    AUTH_URL = "https://cert.tastyworks.com/oauth/authorize"
    OPTION_CHAIN_BASE = "https://api.cert.tastyworks.com"
else:
    BASE_URL = "https://api.tastyworks.com"
    AUTH_URL = "https://api.tastyworks.com/oauth/authorize"
    OPTION_CHAIN_BASE = "https://api.tastyworks.com"
    
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

# Initialize scan results in session state
if "scan_results" not in st.session_state:
    st.session_state.scan_results = []
    
# Track authorization status
if "authorized" not in st.session_state:
    st.session_state.authorized = False

# ===============================
# Utility Functions
# ===============================
def set_user_agent(headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Set consistent User-Agent header for all requests."""
    h = {} if headers is None else dict(headers)
    h["User-Agent"] = "tt-pcs-analyzer/1.2"
    return h

def make_request_with_retry(
    method: str, 
    url: str, 
    headers: Optional[Dict[str, str]] = None, 
    json_data: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = REQUEST_TIMEOUT
) -> Optional[requests.Response]:
    """Make HTTP request with retry logic."""
    headers = set_user_agent(headers)
    
    for attempt in range(MAX_RETRIES):
        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=json_data, timeout=timeout)
            else:
                st.error(f"Unsupported HTTP method: {method}")
                return None
                
            if response.status_code == 200:
                return response
            elif response.status_code in [401, 403]:
                st.warning(f"Authentication error ({response.status_code}) on attempt {attempt + 1}")
                if attempt == MAX_RETRIES - 1:  # Last attempt
                    return response
            else:
                st.warning(f"Request failed with status {response.status_code} on attempt {attempt + 1}")
                
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
                
        except Timeout:
            st.warning(f"Request timed out on attempt {attempt + 1}")
            if attempt == MAX_RETRIES - 1:
                st.error("Request timed out after all retries")
                return None
        except RequestException as e:
            st.error(f"Request exception: {e}")
            return None
            
    return None

def auth_header() -> Dict[str, str]:
    """Generate authorization header with current access token."""
    tok = st.session_state["oauth"]["access_token"]
    if not tok:
        return {}
    return set_user_agent({
        "Authorization": f"Bearer {tok}", 
        "Accept": "application/json", 
        "Content-Type": "application/json"
    })

def save_tokens(token_resp: Dict[str, Any]):
    """Save OAuth tokens to session state."""
    st.session_state["oauth"]["access_token"] = token_resp.get("access_token")
    st.session_state["oauth"]["refresh_token"] = token_resp.get("refresh_token")
    st.session_state["oauth"]["token_type"] = token_resp.get("token_type", "Bearer")
    exp_in = int(token_resp.get("expires_in", 900))
    st.session_state["oauth"]["expires_at"] = int(time.time()) + exp_in - 15
    st.session_state.authorized = True

def token_expired() -> bool:
    """Check if the access token has expired."""
    exp = st.session_state["oauth"]["expires_at"]
    return not exp or time.time() >= exp

def refresh_access_token() -> bool:
    """Refresh the access token using the refresh token."""
    rtok = st.session_state["oauth"]["refresh_token"]
    if not rtok:
        return False
        
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": rtok,
        "client_id": TT_CLIENT_ID,
        "client_secret": TT_CLIENT_SECRET
    }
    
    response = make_request_with_retry("POST", TOKEN_URL, json_data=payload)
    if response and response.status_code == 200:
        save_tokens(response.json())
        return True
    else:
        st.warning("Token refresh failed")
        return False

def ensure_token() -> bool:
    """Ensure we have a valid access token, refreshing if necessary."""
    if st.session_state["oauth"]["access_token"] and not token_expired():
        return True
    if st.session_state["oauth"]["refresh_token"]:
        return refresh_access_token()
    return False

def norm_cdf(x: float) -> float:
    """Calculate the cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def pop_from_breakeven(S: float, be: float, iv_annual: float, dte: int) -> Optional[float]:
    """
    Calculate probability of profit (POP) from breakeven point.
    
    Args:
        S: Current underlying price
        be: Breakeven price
        iv_annual: Implied volatility (annualized)
        dte: Days to expiration
        
    Returns:
        Probability of profit as a percentage, or None if calculation fails
    """
    try:
        if S is None or iv_annual is None or be is None or dte <= 0 or iv_annual <= 0:
            return None
            
        T = dte / 365.0
        # Handle edge cases where be/S might be problematic
        if be <= 0 or S <= 0:
            return None
            
        log_term = math.log(be / S)
        denominator = iv_annual * math.sqrt(T)
        
        # Avoid division by zero
        if denominator == 0:
            return None
            
        z = (log_term + (-0.5 * (iv_annual ** 2) * T)) / denominator
        prob = 1.0 - norm_cdf(z)
        return max(0.0, min(1.0, prob)) * 100.0
    except (ValueError, ZeroDivisionError):
        return None

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
    st.info("Please set these as environment variables or in Streamlit secrets.")

params = {
    "client_id": TT_CLIENT_ID,
    "redirect_uri": TT_REDIRECT_URI,
    "response_type": "code",
    "scope": "read:trade",   # Updated scope
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
    with st.spinner("Exchanging authorization code for tokens..."):
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": TT_CLIENT_ID,
            "client_secret": TT_CLIENT_SECRET,
            "redirect_uri": TT_REDIRECT_URI
        }
        
        headers = set_user_agent({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
        
        response = make_request_with_retry("POST", TOKEN_URL, json_data=data, headers=headers)
        if response and response.status_code == 200:
            save_tokens(response.json())
            st.success("Access token obtained. You're connected.")
        else:
            error_msg = response.text if response else "No response received"
            st.error(f"Token exchange failed: {error_msg}")

# Display authorization status
if st.session_state.authorized:
    st.success("✅ You are authorized and connected to Tastytrade")
else:
    st.info("🔒 Please authorize with Tastytrade to use the scanner")

# ===============================
# 2) Scanner controls
# ===============================
st.subheader("2) Scanner Settings")

# Use columns for better layout
col1, col2 = st.columns(2)

with col1:
    symbols = st.text_input("Symbols (comma-separated)", DEFAULT_SYMBOLS).upper().replace(" ", "")
    scan_mode = st.radio("Expiry window", ["Weekly (7-10 DTE)", "Bi-weekly (14-20 DTE)", "Monthly (30-45 DTE)", "Custom"], index=0)
    
    if scan_mode == "Weekly (7-10 DTE)":
        dte_min, dte_max = 7, 10
    elif scan_mode == "Bi-weekly (14-20 DTE)":
        dte_min, dte_max = 14, 20
    elif scan_mode == "Monthly (30-45 DTE)":
        dte_min, dte_max = 30, 45
    else:
        dte_range = st.slider("Custom DTE range", 0, 90, (7, 20))
        dte_min, dte_max = dte_range
        
    min_credit = st.number_input("Min net credit ($)", min_value=0.00, step=0.05, value=0.20, format="%.2f")
    max_width = st.number_input("Max spread width ($)", min_value=0.5, step=0.5, value=5.0, format="%.2f")
    limit_pairs_per_exp = st.slider("Max spreads per expiration (to limit combinations)", 5, 200, 50)

with col2:
    st.markdown("**Delta targeting**")
    target_delta = st.slider("Target short put |Δ|", 0.05, 0.50, 0.25, 0.01)
    delta_band = st.slider("Acceptable |Δ| range around target", 0.00, 0.20, 0.05, 0.01)
    min_pop = st.slider("Min POP (%)", 50, 90, 65)
    
    st.markdown("**Advanced Options**")
    use_compact_chain = st.checkbox("Use compact chain endpoint (faster)", value=True)
    min_roi = st.number_input("Min ROI (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    max_bid_ask_spread = st.number_input("Max bid-ask spread (%)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)

st.caption("Note: This app **never** submits orders. It only analyzes and generates a downloadable idea list.")

# ===============================
# Data fetchers
# ===============================
def fetch_option_chain(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch option chain data for a symbol."""
    # Updated endpoint URL
    endpoint = f"{OPTION_CHAIN_BASE}/option-chains/{symbol}/nested"
    
    response = make_request_with_retry("GET", endpoint, headers=auth_header())
    if response and response.status_code == 200:
        return response.json().get("data", {})
    else:
        error_msg = response.text if response else "No response received"
        st.warning(f"{symbol}: chain error - {error_msg}")
        return None

def parse_puts_from_chain(chain: Dict[str, Any]) -> pd.DataFrame:
    """Parse put options from chain data."""
    items = []
    
    # Extract underlying price
    underlying_price = chain.get("underlying-price") or chain.get("underlying_price")
    if not underlying_price:
        # Try to get from quotes if not in chain
        underlying_price = fetch_underlying_price(chain.get("symbol", ""))
    
    if "option-chain" in chain and "expirations" in chain["option-chain"]:
        # New API format
        for exp in chain["option-chain"]["expirations"]:
            exp_date = exp.get("expiration-date") or exp.get("expiration_date")
            for strike_data in exp.get("strikes", []):
                for option in strike_data.get("options", []):
                    if option.get("option-type") == "put":
                        bid = option.get("bid")
                        ask = option.get("ask")
                        mid = None
                        if bid is not None and ask is not None:
                            try: 
                                mid = (float(bid) + float(ask)) / 2.0
                            except (ValueError, TypeError): 
                                pass
                                
                        items.append({
                            "symbol": option.get("symbol"),
                            "expiration": exp_date,
                            "strike": float(strike_data.get("strike-price") or strike_data.get("strike_price")),
                            "bid": float(bid) if bid is not None else None,
                            "ask": float(ask) if ask is not None else None,
                            "mid": mid,
                            "iv": option.get("implied-volatility") or option.get("implied_volatility"),
                            "delta": option.get("delta"),
                            "underlying_price": underlying_price
                        })
    else:
        # Fallback to old format parsing
        for exp in chain.get("expirations", []):
            exp_date = exp.get("expiration-date") or exp.get("expiration_date")
            for s in exp.get("strikes", []):
                if (s.get("option-type") or s.get("option_type")) != "put":
                    continue
                    
                bid = s.get("bid"); ask = s.get("ask")
                mid = None
                if bid is not None and ask is not None:
                    try: 
                        mid = (float(bid) + float(ask)) / 2.0
                    except (ValueError, TypeError): 
                        pass
                        
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
    if not df.empty and "expiration" in df.columns:
        df["expiration"] = pd.to_datetime(df["expiration"])
    return df

def _extract_price_from_quote_payload(payload: Dict[str, Any]) -> Optional[float]:
    """
    Attempt to extract a reasonable underlying price from common quote payload shapes.
    Try last/mark/close/ask/bid, in that order.
    """
    if not payload:
        return None
        
    # Common "data" -> "items" structure
    data = payload.get("data") or {}
    if isinstance(data, dict) and "items" in data:
        items = data.get("items") or []
        if items:
            q = items[0]
        else:
            q = {}
    else:
        q = data if isinstance(data, dict) else payload

    # Try different price fields in order of preference
    for k in ["last", "mark", "close", "ask", "bid"]:
        v = q.get(k) if isinstance(q, dict) else None
        try:
            if v is not None:
                price = float(v)
                if price > 0:
                    return price
        except (ValueError, TypeError):
            continue
            
    return None

def fetch_underlying_price(symbol: str) -> Optional[float]:
    """
    Fetch underlying price via Tastytrade quotes.
    """
    if not ensure_token():
        return None
        
    headers = auth_header()

    # Try the quotes endpoint
    try:
        url = f"{BASE_URL}/quotes/{symbol}"
        response = make_request_with_retry("GET", url, headers=headers)
        if response and response.status_code == 200:
            data = response.json()
            # Try to extract price from different possible response formats
            if "data" in data and "last" in data["data"]:
                return float(data["data"]["last"])
            elif "last" in data:
                return float(data["last"])
    except Exception:
        pass

    # Try market-metrics as fallback
    try:
        url = f"{BASE_URL}/market-metrics/{symbol}"
        response = make_request_with_retry("GET", url, headers=headers)
        if response and response.status_code == 200:
            data = response.json()
            if "data" in data and "last" in data["data"]:
                return float(data["data"]["last"])
    except Exception:
        pass

    return None

# ===============================
# 3) Run scan (Preview only)
# ===============================
st.subheader("3) Run Scan")

run_scan = st.button("Run scan (data-only)")

if run_scan:
    if not st.session_state.authorized:
        st.error("Please authorize first by completing step 1.")
    else:
        all_syms = [s.strip() for s in symbols.split(",") if s.strip()]
        if not all_syms:
            st.error("Please enter at least one symbol.")
            st.stop()
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_rows = []
        
        for i, sym in enumerate(all_syms):
            status_text.text(f"Scanning {sym} ({i+1}/{len(all_syms)})...")
            progress_bar.progress((i) / len(all_syms))
            
            chain = fetch_option_chain(sym)
            if not chain:
                continue
                
            df_puts = parse_puts_from_chain(chain)
            if df_puts.empty:
                st.warning(f"{sym}: no puts parsed.")
                continue

            # Get underlying price
            u_price = None
            if df_puts["underlying_price"].notna().any():
                u_price = df_puts["underlying_price"].dropna().iloc[0]
                
            if u_price is None:
                u_price = fetch_underlying_price(sym)
                
            if u_price is None:
                st.warning(f"{sym}: could not determine underlying price. Please enter manually.")
                u_price = st.number_input(f"{sym} underlying price", min_value=0.01, step=0.01, value=100.00, key=f"px_{sym}")
                if not u_price:
                    continue

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

                # Filter by delta if available, otherwise use OTM strikes
                if df_e["delta"].notna().any():
                    df_e["abs_delta"] = df_e["delta"].abs()
                    lower, upper = max(0.0, target_delta - delta_band), min(1.0, target_delta + delta_band)
                    cand_short = df_e[(df_e["abs_delta"] >= lower) & (df_e["abs_delta"] <= upper)].copy()
                    
                    if cand_short.empty:
                        # Widen delta range if no candidates found
                        widen = min(0.25, delta_band * 2 if delta_band > 0 else 0.1)
                        cand_short = df_e[
                            (df_e["abs_delta"] >= target_delta - widen) & 
                            (df_e["abs_delta"] <= target_delta + widen)
                        ].copy()
                        
                    cand_short["delta_distance"] = (cand_short["abs_delta"] - target_delta).abs()
                else:
                    # Fallback heuristic if no Greeks
                    cand_short = df_e[df_e["strike"] < u_price].copy()
                    cand_short["delta_distance"] = 0.5  # Neutral placeholder

                # Filter by bid-ask spread
                cand_short = cand_short[cand_short["bid"].notna() & cand_short["ask"].notna()].copy()
                if not cand_short.empty and "mid" in cand_short.columns:
                    cand_short["bid_ask_spread_pct"] = (
                        (cand_short["ask"] - cand_short["bid"]) / cand_short["mid"] * 100
                    )
                    cand_short = cand_short[cand_short["bid_ask_spread_pct"] <= max_bid_ask_spread].copy()
                
                if cand_short.empty:
                    continue
                    
                cand_short = cand_short.sort_values(["delta_distance", "strike"], ascending=[True, False])
                cand_short = cand_short.head(limit_pairs_per_exp)

                for _, srow in cand_short.iterrows():
                    short_mid = srow["mid"] if pd.notna(srow["mid"]) else None
                    if short_mid is None:
                        continue
                        
                    short_bid = srow["bid"]
                    short_ask = srow["ask"]
                    short_strike = srow["strike"]
                    iv_short = float(srow["iv"]) if srow["iv"] is not None else None
                    DTE = int(srow["DTE"])

                    # Find long put candidates
                    df_long = df_e[df_e["strike"] < short_strike].copy()
                    df_long = df_long[df_long["strike"] >= short_strike - max_width]
                    df_long = df_long.sort_values("strike", ascending=False).head(8)

                    for _, lrow in df_long.iterrows():
                        long_mid = lrow["mid"] if pd.notna(lrow["mid"]) else None
                        if long_mid is None:
                            continue
                            
                        long_bid = lrow["bid"]
                        long_ask = lrow["ask"]
                        long_strike = lrow["strike"]

                        # Calculate spread metrics
                        credit = round(float(short_bid) - float(long_ask), 2)  # Use worst case
                        if credit < float(min_credit):
                            continue

                        width = round(short_strike - long_strike, 2)
                        if width <= 0:
                            continue

                        max_loss = round(width - credit, 2)
                        if max_loss <= 0:
                            continue

                        roi = round(credit / max_loss * 100.0, 2)
                        if roi < min_roi:
                            continue
                            
                        breakeven = round(short_strike - credit, 2)

                        # Get IV for POP calculation
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
                        link = f"https://my.tastytrade.com/trade/{sym}/options"

                        results_rows.append({
                            "Symbol": sym,
                            "Price": round(float(u_price), 2),
                            "Exp Date": exp_date.date().isoformat(),
                            "DTE": DTE,
                            "Short": f"{short_strike:.2f}P",
                            "Short Bid": round(float(short_bid), 2) if short_bid is not None else None,
                            "Short Ask": round(float(short_ask), 2) if short_ask is not None else None,
                            "Long": f"{long_strike:.2f}P",
                            "Long Bid": round(float(long_bid), 2) if long_bid is not None else None,
                            "Long Ask": round(float(long_ask), 2) if long_ask is not None else None,
                            "Credit": credit,
                            "Width": width,
                            "Max Profit": round(credit, 2),
                            "Max Loss": max_loss,
                            "ROI%": roi,
                            "BE": breakeven,
                            "BE%": round((breakeven - u_price) / u_price * 100.0, 2),
                            "IV (short)": round(iv * 100.0, 2) if iv is not None else None,
                            "POP%": round(pop, 2) if pop is not None else None,
                            "Loss Prob%": loss_prob,
                            "Short |Δ|": round(abs(srow["delta"]), 3) if srow.get("delta") is not None else None,
                            "Δ distance": round(abs((abs(srow.get("delta")) if srow.get("delta") is not None else target_delta) - target_delta), 3),
                            "Risk/Reward": rr_txt,
                            "Links": link
                        })
        
        progress_bar.progress(1.0)
        status_text.text("Scan complete!")
        st.session_state.scan_results = results_rows

# ===============================
# 4) Results table + downloads
# ===============================
st.subheader("4) Results (Preview Only)")

if st.session_state.scan_results:
    res_df = pd.DataFrame(st.session_state.scan_results)
    
    if not res_df.empty:
        # Add a quality score for ranking
        if "POP%" in res_df.columns and "ROI%" in res_df.columns and "Δ distance" in res_df.columns:
            # Normalize values for scoring
            res_df["pop_norm"] = (res_df["POP%"] - res_df["POP%"].min()) / (res_df["POP%"].max() - res_df["POP%"].min() + 1e-10)
            res_df["roi_norm"] = (res_df["ROI%"] - res_df["ROI%"].min()) / (res_df["ROI%"].max() - res_df["ROI%"].min() + 1e-10)
            res_df["delta_norm"] = 1 - (res_df["Δ distance"] - res_df["Δ distance"].min()) / (res_df["Δ distance"].max() - res_df["Δ distance"].min() + 1e-10)
            
            # Weighted score (adjust weights as needed)
            res_df["quality_score"] = (
                0.4 * res_df["pop_norm"] + 
                0.3 * res_df["roi_norm"] + 
                0.3 * res_df["delta_norm"]
            )
            
            # Sort by quality score
            res_df = res_df.sort_values("quality_score", ascending=False)
        else:
            # Fallback sorting
            by = [c for c in ["POP%", "ROI%", "Credit"] if c in res_df.columns]
            asc = [False, False, False][:len(by)]
            if by:
                res_df = res_df.sort_values(by=by, ascending=asc, na_position="last")

        # Display results
        top_n = st.slider("Show top N results", 10, min(500, len(res_df)), min(100, len(res_df)))
        
        # Format display dataframe
        display_cols = [
            "Symbol", "Price", "Exp Date", "DTE", "Short", "Long", 
            "Credit", "Width", "Max Profit", "Max Loss", "ROI%", 
            "POP%", "Short |Δ|", "Risk/Reward", "Links"
        ]
        display_df = res_df[[c for c in display_cols if c in res_df.columns]].head(top_n)
        
        st.dataframe(display_df, use_container_width=True)
        
        # Add some summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Spreads Found", len(res_df))
            st.metric("Average Credit", f"${res_df['Credit'].mean():.2f}")
            
        with col2:
            st.metric("Average POP", f"{res_df['POP%'].mean():.1f}%")
            st.metric("Average ROI", f"{res_df['ROI%'].mean():.1f}%")
            
        with col3:
            st.metric("Average DTE", f"{res_df['DTE'].mean():.1f} days")
            st.metric("Best POP", f"{res_df['POP%'].max():.1f}%")

        # Download options
        st.subheader("Download Results")
        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV", 
            data=csv_bytes, 
            file_name=f"pcs_scan_{datetime.utcnow().date().isoformat()}.csv", 
            mime="text/csv"
        )

        json_bytes = res_df.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "⬇️ Download JSON", 
            data=json_bytes, 
            file_name=f"pcs_scan_{datetime.utcnow().date().isoformat()}.json", 
            mime="application/json"
        )

        st.success("Preview-only analyzer. No orders are sent to Tastytrade.")
    else:
        st.info("No spreads found matching your criteria. Try adjusting your parameters.")
else:
    st.caption("Run a scan to see results here.")
