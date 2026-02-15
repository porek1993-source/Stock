"""
Stock Picker Pro v3.0 (Final Mobile Fix + FCF TTM)
==================================================
Kompletní verze aplikace.
"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'google\.generativeai\..*')

import re
import json
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# --- Secrets / API keys ---
def _get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return str(os.getenv(name, default) or default)

GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")

# PDF Export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

# Constants
APP_NAME = "Stock Picker Pro"
APP_VERSION = "v3.0 Mobile"
GEMINI_MODEL = "gemini-2.0-flash-exp"

DATA_DIR = os.path.join(os.path.dirname(__file__), ".stock_picker_pro")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
MEMOS_PATH = os.path.join(DATA_DIR, "memos.json")

# Sector Peers
SECTOR_PEERS = {
    "Technology": {
        "AAPL": ["MSFT", "GOOGL", "META", "NVDA"],
        "MSFT": ["AAPL", "GOOGL", "META", "AMZN"],
        "GOOGL": ["AAPL", "MSFT", "META", "AMZN"],
        "META": ["AAPL", "GOOGL", "SNAP", "PINS"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "TSLA": ["RIVN", "LCID", "F", "GM"],
        "NFLX": ["DIS", "PARA", "WBD"],
    },
    "Consumer Cyclical": {
        "AMZN": ["WMT", "TGT", "EBAY", "BABA"],
        "TSLA": ["F", "GM", "RIVN", "LCID"],
    },
    "Healthcare": {
        "JNJ": ["PFE", "UNH", "ABT", "MRK"],
        "PFE": ["JNJ", "MRK", "ABBV", "LLY"],
    },
    "Financial Services": {
        "JPM": ["BAC", "WFC", "C", "GS"],
        "V": ["MA", "PYPL", "SQ"],
    },
    "Communication Services": {
        "T": ["VZ", "TMUS"],
    },
}

MACRO_CALENDAR = [
    {"date": "2026-02-20", "event": "FOMC Minutes Release", "importance": "High"},
    {"date": "2026-03-06", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-03-11", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-03-18", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-03-25", "event": "US GDP (Q4 2025 Final)", "importance": "Medium"},
]

# --- Utilities ---
def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, obj: Any) -> None:
    ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        if isinstance(x, (np.generic,)): x = x.item()
        if isinstance(x, (int, float)) and math.isfinite(float(x)): return float(x)
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "": return None
            v = float(x)
            if math.isfinite(v): return v
        return None
    except Exception:
        return None

def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None or b == 0: return None
    return a / b

def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None: return "—"
    return f"{v:,.{digits}f}"

def fmt_pct(x: Any, digits: int = 1) -> str:
    v = safe_float(x)
    if v is None: return "—"
    return f"{v*100:.{digits}f}%"

def fmt_money(x: Any, digits: int = 2, prefix: str = "$") -> str:
    v = safe_float(x)
    if v is None: return "—"
    return f"{prefix}{v:,.{digits}f}"

# --- FCF TTM LOGIC ---
def get_fcf_ttm_yfinance(t: "yf.Ticker", info: Dict[str, Any], *, debug: bool = True) -> Tuple[Optional[float], List[str]]:
    dbg: List[str] = []
    
    def _sum_last4(df: pd.DataFrame, row_candidates: List[str]) -> Optional[float]:
        try:
            cols = list(df.columns)
            if not cols: return None
            dated_cols = []
            for c in cols:
                try: d = pd.to_datetime(c, errors="coerce")
                except: d = pd.NaT
                dated_cols.append((d, c))
            dated_cols.sort(key=lambda x: (pd.isna(x[0]), x[0] if not pd.isna(x[0]) else pd.Timestamp.min), reverse=True)
            cols_last4 = [c for _, c in dated_cols[:4]]
            
            idx_lower = [str(i).lower() for i in df.index]
            def _get_vals(row_label):
                vals = []
                for c in cols_last4:
                    val = safe_float(df.loc[row_label, c])
                    if val is not None: vals.append(val)
                return vals

            for candidate in row_candidates:
                if candidate in df.index:
                    vals = _get_vals(candidate)
                    if len(vals) >= 1: return sum(vals) * (4/len(vals)) if len(vals) < 4 else sum(vals)
                cand_lower = candidate.lower()
                for i, idx_val in enumerate(idx_lower):
                    if cand_lower == idx_val or cand_lower in idx_val:
                        vals = _get_vals(df.index[i])
                        if len(vals) >= 1: return sum(vals) * (4/len(vals)) if len(vals) < 4 else sum(vals)
            return None
        except: return None

    try:
        qcf = t.quarterly_cashflow
        if qcf is not None and not qcf.empty:
            fcf_sum = _sum_last4(qcf, ["Free Cash Flow", "FreeCashFlow"])
            if fcf_sum: return fcf_sum, dbg
            cfo = _sum_last4(qcf, ["Total Cash From Operating Activities", "Operating Cash Flow"])
            capex = _sum_last4(qcf, ["Capital Expenditures", "CapitalExpenditures", "Capex"])
            if cfo is not None and capex is not None:
                return cfo - abs(capex), dbg
    except: pass

    fcf_fallback = safe_float(info.get("freeCashflow"))
    if fcf_fallback: return fcf_fallback, dbg
    return None, dbg

# --- Data Fetching ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    try: return yf.Ticker(ticker).info or {}
    except: return {}

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try: return yf.Ticker(ticker).history(period=period, auto_adjust=False)
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try: t = yf.Ticker(ticker); return (t.financials, t.balance_sheet, t.cashflow)
    except: return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

@st.cache_data(show_spinner=False, ttl=3600)
def get_all_time_high(ticker: str) -> Optional[float]:
    try:
        h = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty: return None
        col = "High" if "High" in h.columns else ("Close" if "Close" in h.columns else None)
        return float(pd.to_numeric(h[col], errors="coerce").max()) if col else None
    except: return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_insider_transactions(ticker: str) -> Optional[pd.DataFrame]:
    try: return getattr(yf.Ticker(ticker), "insider_transactions", None)
    except: return None

# --- Metrics ---
@dataclass
class Metric:
    name: str
    value: Optional[float]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    target_below: Optional[float] = None
    target_above: Optional[float] = None
    weight: float = 1.0

def extract_metrics(info: Dict[str, Any], ticker: str) -> Dict[str, Metric]:
    t = yf.Ticker(ticker)
    fcf, _ = get_fcf_ttm_yfinance(t, info)
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    market_cap = safe_float(info.get("marketCap"))
    fcf_yield = safe_div(fcf, market_cap) if fcf and market_cap else None
    def g(k): return safe_float(info.get(k))

    return {
        "price": Metric("Current Price", price),
        "pe": Metric("P/E Ratio", g("trailingPE"), target_below=25, weight=1.5),
        "pb": Metric("P/B Ratio", g("priceToBook"), target_below=3, weight=1.0),
        "ps": Metric("P/S Ratio", g("priceToSalesTrailing12Months"), target_below=2, weight=1.0),
        "peg": Metric("PEG Ratio", g("pegRatio"), target_below=1.5, weight=1.5),
        "ev_ebitda": Metric("EV/EBITDA", g("enterpriseToEbitda"), target_below=15, weight=1.0),
        "roe": Metric("ROE", g("returnOnEquity"), target_above=0.15, weight=2.0),
        "roa": Metric("ROA", g("returnOnAssets"), target_above=0.05, weight=1.0),
        "operating_margin": Metric("Op Margin", g("operatingMargins"), target_above=0.15, weight=1.5),
        "profit_margin": Metric("Net Margin", g("profitMargins"), target_above=0.10, weight=1.5),
        "gross_margin": Metric("Gross Margin", g("grossMargins"), target_above=0.30, weight=1.0),
        "revenue_growth": Metric("Rev Growth", g("revenueGrowth"), target_above=0.10, weight=2.0),
        "earnings_growth": Metric("EPS Growth", g("earningsGrowth"), target_above=0.10, weight=2.0),
        "current_ratio": Metric("Curr Ratio", g("currentRatio"), target_above=1.5, weight=1.0),
        "quick_ratio": Metric("Quick Ratio", g("quickRatio"), target_above=1.0, weight=0.8),
        "debt_to_equity": Metric("D/E", g("debtToEquity"), target_below=1.0
    # --- DCF, Insider, Peers ---
def calculate_dcf_fair_value(fcf: float, growth_rate: float, terminal_growth: float, wacc: float, years: int, shares: float) -> Optional[float]:
    if fcf <= 0 or not shares: return None
    try:
        pv_sum = 0.0
        current_fcf = fcf
        for year in range(1, years + 1):
            current_fcf *= (1 + growth_rate)
            pv_sum += current_fcf / ((1 + wacc) ** year)
        terminal_val = (current_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_terminal = terminal_val / ((1 + wacc) ** years)
        return (pv_sum + pv_terminal) / shares
    except: return None

def compute_buy_price_levels(fair_value: float | None) -> dict:
    if not fair_value: return {"buy": None, "strong_buy": None, "must_buy": None}
    fv = float(fair_value)
    return {"buy": fv * 0.95, "strong_buy": fv * 0.80, "must_buy": fv * 0.70}

def compute_insider_pro_signal(insider_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if insider_df is None or insider_df.empty:
        return {"signal": 0, "label": "Neutral", "recent_buys": 0, "recent_sells": 0}
    buy_count, sell_count = 0, 0
    cutoff = dt.datetime.now() - dt.timedelta(days=180)
    for _, row in insider_df.iterrows():
        try:
            d_str = row.get("Start Date", row.get("Date", ""))
            if pd.isna(d_str) or pd.to_datetime(d_str) < cutoff: continue
            trans = str(row.get("Transaction", "")).lower()
            if "buy" in trans or "purchase" in trans: buy_count += 1
            elif "sell" in trans and "tax" not in trans: sell_count += 1
        except: continue
    net_signal = buy_count - sell_count
    signal = min(100, max(-100, net_signal * 20))
    label = "Strong Buy" if signal >= 50 else "Buy" if signal >= 20 else "Sell" if signal <= -20 else "Neutral"
    return {"signal": signal, "label": label, "recent_buys": buy_count, "recent_sells": sell_count}

def get_auto_peers(ticker: str, sector: str, info: Dict[str, Any]) -> List[str]:
    for sect, map_ in SECTOR_PEERS.items():
        if ticker in map_: return map_[ticker][:5]
    return []

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_peer_comparison(ticker: str, peers: List[str]) -> pd.DataFrame:
    rows = []
    for t in [ticker] + peers:
        info = fetch_ticker_info(t)
        if info:
            rows.append({
                "Ticker": t,
                "P/E": safe_float(info.get("trailingPE")),
                "Op. Margin": safe_float(info.get("operatingMargins")),
                "Rev. Growth": safe_float(info.get("revenueGrowth")),
                "Market Cap": safe_float(info.get("marketCap"))
            })
    return pd.DataFrame(rows)

# --- PDF Export ---
def export_memo_pdf(ticker: str, company: str, memo: Dict[str, str], summary: Dict[str, str]) -> Optional[bytes]:
    if not _HAS_PDF: return None
    try:
        from io import BytesIO
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, 10*inch, f"Investment Memo: {company} ({ticker})")
        c.setFont("Helvetica", 10)
        y = 9.5*inch
        for k, v in summary.items():
            c.drawString(1*inch, y, f"{k}: {v}")
            y -= 0.2*inch
        y -= 0.3*inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, y, "Investment Thesis")
        y -= 0.2*inch
        c.setFont("Helvetica", 10)
        text = memo.get("thesis", "")
        for line in text.split('\n'):
            c.drawString(1*inch, y, line[:90])
            y -= 0.15*inch
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    except: return None

# --- AI Analyst ---
def generate_ai_analyst_report(ticker, company, metrics, info, dcf_fair_value, current_price, scorecard, insider_signal, macro_events, suggested_buy_price=None):
    if not GEMINI_API_KEY:
        return {"market_situation": "Chybí API klíč", "bull_case": [], "bear_case": [], "verdict": "N/A", "wait_for_price": None}
    if suggested_buy_price is None and dcf_fair_value:
        suggested_buy_price = float(dcf_fair_value) * 0.95
    context = f"""
    Jsi Senior Financial Analyst. Analyzuj akci {company} ({ticker}).
    DATA: Cena: {fmt_money(current_price)}, Férová hodnota (DCF): {fmt_money(dcf_fair_value)}, Scorecard: {scorecard:.0f}/100, Insider: {insider_signal.get('label')}, P/E: {metrics['pe'].value}
    INSTRUKCE: Vrať JSON s klíči: market_situation (str), bull_case (list[str]), bear_case (list[str]), verdict (BUY/HOLD/SELL), wait_for_price (number), reasoning (str)
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(context)
        text = re.sub(r'```json\s*|```', '', resp.text).strip()
        return json.loads(text)
    except Exception as e:
        return {"market_situation": f"Chyba: {e}", "verdict": "ERROR"}

# --- Watchlist/Memo ---
def get_watchlist() -> Dict[str, Any]: return load_json(WATCHLIST_PATH, {"items": {}})
def set_watchlist(data: Dict[str, Any]) -> None: save_json(WATCHLIST_PATH, data)
def get_memos() -> Dict[str, Any]: return load_json(MEMOS_PATH, {"memos": {}})
def set_memos(data: Dict[str, Any]) -> None: save_json(MEMOS_PATH, data)

# --- Verdict Logic ---
def get_advanced_verdict(scorecard: float, mos_dcf: Optional[float], mos_analyst: Optional[float], insider_signal: float, implied_growth: Optional[float]) -> Tuple[str, str, List[str]]:
    warnings = []
    if scorecard >= 75: base, color = "STRONG BUY", "#00ff88"
    elif scorecard >= 60: base, color = "BUY", "#88ff00"
    elif scorecard >= 45: base, color = "HOLD", "#ffaa00"
    else: base, color = "AVOID", "#ff4444"
    
    if mos_dcf is not None and mos_dcf >= 0.20 and base in ["HOLD"]: base, color = "BUY", "#88ff00"
    if insider_signal > 50: warnings.append(f"✅ Insider buying (+{insider_signal:.0f})")
    
    return base, color, warnings

def get_earnings_calendar_estimate(ticker: str, info: Dict[str, Any]) -> Optional[dt.date]:
    try:
        t = yf.Ticker(ticker)
        calendar = getattr(t, "calendar", None)
        if calendar is not None and not calendar.empty:
            if "Earnings Date" in calendar.index:
                next_earnings = calendar.loc["Earnings Date"].iloc[0]
                if pd.notna(next_earnings): return pd.to_datetime(next_earnings).date()
    except: pass
    today = dt.date.today()
    if today.month < 4: return dt.date(today.year, 4, 25)
    return dt.date(today.year, 7, 25)
    # ============================================================================
# HLAVNÍ APLIKACE
# ============================================================================

def main():
    """Main application entry point."""
    
    # --- 1. SESSION STATE ---
    if "sidebar_state" not in st.session_state: st.session_state["sidebar_state"] = "expanded"
    if "analysis_active" not in st.session_state: st.session_state["analysis_active"] = False

    def _open_sidebar(): st.session_state["sidebar_state"] = "expanded"

    # --- 2. CONFIG & CSS ---
    st.set_page_config(page_title="Stock Picker Pro v3.0", page_icon="📈", layout="wide", initial_sidebar_state=st.session_state["sidebar_state"])

    # AGRESIVNÍ CSS FIX
    sidebar_css = ""
    if st.session_state["sidebar_state"] == "collapsed":
        sidebar_css = """
            <style>
                section[data-testid="stSidebar"] { display: none !important; }
                button[kind="header"] { display: none !important; }
                div[data-testid="collapsedControl"] { display: none !important; }
            </style>
        """
    
    st.markdown(f"""
    <style>
        .stButton > button {{ width: 100%; margin: 5px 0; min-height: 44px; }}
        [data-testid="stMetricValue"] {{ font-size: clamp(1.2rem, 4vw, 2rem); }}
        .metric-card {{ padding: 15px; border-radius: 10px; background: rgba(255, 255, 255, 0.03); margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.1); }}
        .metric-label {{ font-size: 0.85rem; opacity: 0.7; }}
        .metric-value {{ font-size: clamp(1.5rem, 5vw, 2.5rem); font-weight: 700; }}
        section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, rgba(0,0,0,0.03) 0%, rgba(0,0,0,0.01) 100%); }}
        @media (max-width: 768px){{ section[data-testid="stSidebar"]{{ background: rgba(15,23,42,0.995)!important; }} }}
    </style>
    {sidebar_css}
    """, unsafe_allow_html=True)
    
    # --- 3. SIDEBAR ---
    if st.session_state["sidebar_state"] == "collapsed":
        st.button("☰ Změnit parametry", key="open_menu_btn", on_click=_open_sidebar)

    analyze_btn = st.session_state["analysis_active"]
    dcf_growth = st.session_state.get("dcf_growth", 0.10)
    dcf_wacc = st.session_state.get("dcf_wacc", 0.10)
    dcf_terminal = st.session_state.get("dcf_terminal", 0.03)
    dcf_years = st.session_state.get("dcf_years", 5)
    use_ai = st.session_state.get("use_ai", bool(GEMINI_API_KEY))

    with st.sidebar:
        st.title("📈 Stock Picker Pro")
        st.caption("v3.0 - Mobile Ready")
        st.markdown("---")
        ticker_input = st.text_input("Ticker Symbol", value=st.session_state.get("last_ticker", "AAPL"), max_chars=10).upper().strip()

        # HLAVNÍ TLAČÍTKO
        if st.button("🔍 Analyzovat", type="primary", use_container_width=True, key="trigger_analysis"):
            st.session_state["sidebar_state"] = "collapsed"
            st.session_state["analysis_active"] = True
            st.session_state["last_ticker"] = ticker_input
            st.rerun()

        st.markdown("---")
        with st.expander("⚙️ DCF Parametry", expanded=False):
            dcf_growth = st.slider("Růst FCF", 0.0, 0.50, dcf_growth, 0.01)
            dcf_wacc = st.slider("WACC", 0.05, 0.20, dcf_wacc, 0.01)
            dcf_terminal = st.slider("Terminální růst", 0.0, 0.10, dcf_terminal, 0.01)
            dcf_years = st.slider("Roky", 3, 10, dcf_years, 1)
            st.session_state.update({"dcf_growth": dcf_growth, "dcf_wacc": dcf_wacc, "dcf_terminal": dcf_terminal, "dcf_years": dcf_years})
        
        st.markdown("---")
        with st.expander("🤖 AI Nastavení", expanded=False):
            use_ai = st.checkbox("Povolit AI analýzu", value=use_ai, disabled=not GEMINI_API_KEY)
            st.session_state["use_ai"] = use_ai

    # --- 4. CONTENT ---
    if not st.session_state["analysis_active"]:
        st.title("Vítej v Stock Picker Pro! 🚀")
        st.info("👈 Zadej ticker v menu a klikni na Analyzovat")
        st.markdown("### Populární tickery")
        cols = st.columns(4)
        for i, tkr in enumerate(["AAPL", "MSFT", "NVDA", "TSLA"]):
            with cols[i]:
                if st.button(tkr, use_container_width=True):
                    st.session_state["last_ticker"] = tkr
                    st.rerun()
        st.stop()

    ticker = st.session_state.get("last_ticker", "AAPL")
    
    with st.spinner(f"Analyzuji {ticker}..."):
        info = fetch_ticker_info(ticker)
        if not info:
            st.error("Chyba načítání dat."); st.stop()
            
        metrics = extract_metrics(info, ticker)
        company = info.get("longName", ticker)
        current_price = metrics["price"].value
        
        t = yf.Ticker(ticker)
        fcf, _ = get_fcf_ttm_yfinance(t, info)
        shares = safe_float(info.get("sharesOutstanding"))
        
        fair_value_dcf = calculate_dcf_fair_value(fcf, dcf_growth, dcf_terminal, dcf_wacc, dcf_years, shares) if fcf and shares else None
        buy_levels = compute_buy_price_levels(fair_value_dcf)
        insider_df = fetch_insider_transactions(ticker)
        insider_signal = compute_insider_pro_signal(insider_df)
        scorecard, cat_scores, _ = build_scorecard_advanced(metrics, info)
        
        verdict, v_color, warnings = get_advanced_verdict(scorecard, (fair_value_dcf/current_price - 1) if fair_value_dcf and current_price else None, None, insider_signal["signal"], None)
        auto_peers = get_auto_peers(ticker, info.get("sector", ""), info)

        # HEADER UI
        st.title(f"{company} ({ticker})")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.markdown(f'<div class="metric-card"><div class="metric-label">Cena</div><div class="metric-value">{fmt_money(current_price)}</div></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-label">Férovka (DCF)</div><div class="metric-value">{fmt_money(fair_value_dcf)}</div></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-label">Score</div><div class="metric-value">{scorecard:.0f}/100</div></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><div class="metric-label">Verdikt</div><div class="metric-value" style="color:{v_color}">{verdict}</div></div>', unsafe_allow_html=True)
        col5.markdown(f'<div class="metric-card" style="border:1px solid {v_color}"><div class="metric-label">Buy Price</div><div class="metric-value">{fmt_money(buy_levels["buy"])}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        tabs = st.tabs(["📊 Přehled", "💰 Valuace", "🤖 AI Analytik", "🏢 Konkurence", "📝 Watchlist"])
        
        with tabs[0]:
            c1, c2 = st.columns(2)
            c1.metric("P/E", fmt_num(metrics["pe"].value))
            c2.metric("Růst Tržeb", fmt_pct(metrics["revenue_growth"].value))
            st.line_chart(fetch_price_history(ticker)["Close"])
            if warnings:
                st.error("Upozornění:")
                for w in warnings: st.write(f"- {w}")

        with tabs[1]:
            st.markdown(f"### DCF Model (FCF: {fmt_money(fcf/1e9 if fcf else 0)} mld)")
            st.write("Férová cena vypočtena na základě 5-leté projekce Free Cash Flow.")
            st.metric("Strong Buy Price", fmt_money(buy_levels["strong_buy"]))
            st.metric("Must Buy Price", fmt_money(buy_levels["must_buy"]))

        with tabs[2]:
            if GEMINI_API_KEY:
                if st.button("Vygenerovat Report"):
                    with st.spinner("Generuji..."):
                        report = generate_ai_analyst_report(ticker, company, metrics, info, fair_value_dcf, current_price, scorecard, insider_signal, MACRO_CALENDAR)
                        st.write(report.get("market_situation"))
                        st.success(f"Verdikt AI: {report.get('verdict')}")
                        c1, c2 = st.columns(2)
                        with c1: st.write("**Bull Case:**", report.get("bull_case"))
                        with c2: st.write("**Bear Case:**", report.get("bear_case"))
            else:
                st.warning("Chybí API klíč")

        with tabs[3]:
            st.dataframe(fetch_peer_comparison(ticker, auto_peers), use_container_width=True)

        with tabs[4]:
            watch = get_watchlist()
            if st.button("⭐ Přidat do Watchlistu"):
                watch.setdefault("items", {})[ticker] = {"added": dt.datetime.now().isoformat()}
                set_watchlist(watch)
                st.success("Přidáno!")
            st.write(watch.get("items", {}))

if __name__ == "__main__":
    main()
