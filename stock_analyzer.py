"""
Stock Picker Pro v4.0
================================
Robustní, dvojjazyčná (CZ/EN) aplikace optimalizovaná pro Gemini 2.5 Flash Lite (Free Tier)
s pokročilou finanční analýzou, sektorovou inteligencí a perfektním UX.

Author: Enhanced by Claude
Verze: 4.0
"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'google\.generativeai\..*')
import requests
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
import streamlit.components.v1 as components


# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Stock Picker Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Layout fix (full width on desktop) ---
st.markdown(
    """
    <style>
      .block-container { max-width: 100% !important; padding-left: 1.2rem; padding-right: 1.2rem; }
      @media (max-width: 768px) { .block-container { padding-left: 0.8rem; padding-right: 0.8rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)
# Use full width on desktop (avoid centered/narrow container)
st.markdown(
    """
    <style>
      .block-container { max-width: 100% !important; padding-left: 1.2rem; padding-right: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def js_close_sidebar():
    """Return HTML+JS that attempts to close Streamlit sidebar/drawer (mobile + desktop)."""
    return """
    <script>
      (function () {
        function getDoc() {
          try { return (window.parent && window.parent.document) ? window.parent.document : document; }
          catch (e) { return document; }
        }

        function isSidebarOpen(doc) {
          var sb = doc.querySelector('section[data-testid="stSidebar"], [data-testid="stSidebar"]');
          if (!sb) return false;
          try {
            var r = sb.getBoundingClientRect();
            // On desktop sidebar has width; on mobile drawer may overlay with width as well
            return (r.width && r.width > 40) || (r.right && r.right > 40);
          } catch (e) {
            return true;
          }
        }

        function findCloseButton(doc) {
          var selectors = [
            'button[aria-label="Close sidebar"]',
            'button[aria-label="Collapse sidebar"]',
            'button[title="Close sidebar"]',
            '[data-testid="stSidebarCollapseButton"]',
            '[data-testid="stSidebarToggleButton"]',
            'header button[aria-label="Close sidebar"]',
            'header button[aria-label="Collapse sidebar"]',
            'header [data-testid="stSidebarCollapseButton"]',
            'header [data-testid="stSidebarToggleButton"]'
          ];
          for (var i = 0; i < selectors.length; i++) {
            var el = doc.querySelector(selectors[i]);
            if (el) return el;
          }
          // Fallback: first button inside sidebar section
          var sb = doc.querySelector('section[data-testid="stSidebar"], [data-testid="stSidebar"]');
          if (sb) {
            var b = sb.querySelector('button');
            if (b) return b;
          }
          return null;
        }

        function attemptClose() {
          var doc = getDoc();
          if (!isSidebarOpen(doc)) return true; // nothing to do
          var btn = findCloseButton(doc);
          if (btn) {
            btn.click();
            // second click helps on some mobile browsers
            setTimeout(function(){ try { btn.click(); } catch(e){} }, 120);
            return true;
          }
          return false;
        }

        var tries = 0;
        var maxTries = 25;
        var timer = setInterval(function () {
          tries++;
          var ok = false;
          try { ok = attemptClose(); } catch (e) { ok = false; }
          if (ok || tries >= maxTries) {
            clearInterval(timer);
          }
        }, 120);

        // Also try shortly after start
        setTimeout(function(){ try { attemptClose(); } catch(e){} }, 60);
      })();
    </script>
    """

def js_open_tab(tab_label: str) -> str:
    """Return HTML+JS that tries to re-select a Streamlit tab by its label (robust against emoji)."""
    # Use JSON encoding to avoid quote escaping issues
    target = json.dumps(tab_label)
    return f"""
<script>
(function() {{
  const target = {target};
  function norm(s) {{
    return (s || "")
      .toLowerCase()
      .replace(/[^a-z0-9 ]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }}
  const want = norm(target);
  function tryClick() {{
    const doc = window.parent.document;
    const tabs = doc.querySelectorAll('[role="tab"], button[role="tab"]');
    for (const t of tabs) {{
      const txt = norm(t.innerText || t.textContent);
      if (txt && (txt === want || txt.includes(want) || want.includes(txt))) {{
        t.click();
        return true;
      }}
    }}
    return false;
  }}
  let tries = 0;
  const timer = setInterval(() => {{
    tries += 1;
    if (tryClick() || tries > 25) clearInterval(timer);
  }}, 200);
}})();
</script>
"""


def _get_secret(name: str, default: str = "") -> str:
    try:
        # Streamlit Cloud secrets
        return str(st.secrets.get(name, default) or default)
    except Exception:
        # local env fallback
        return str(os.getenv(name, default) or default)

# Read from Streamlit secrets (preferred) or env.
# In Streamlit Cloud > App settings > Secrets:
# GEMINI_API_KEY="..."
# FMP_API_KEY="..."
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY", "")
FMP_API_KEY = _get_secret("FMP_API_KEY", "")
# PDF Export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

# Constants
APP_NAME = "Stock Picker Pro"
APP_VERSION = "v4.0"

GEMINI_MODEL = "gemini-2.5-flash-lite"  # Optimized for Free Tier
MAX_AI_RETRIES = 3  # Retry logic for rate limits
RETRY_DELAY = 2  # seconds


# ============================================================================


# -----------------------------------------------------------------------------
# Social & Guru (X/Twitter) handles
# -----------------------------------------------------------------------------
GURUS = {
    "CZ/SK Scéna": {
        "Jaroslav Brychta": "JaroslavBrychta",
        "Dominik Stroukal": "stroukal",
        "Jaroslav Šura": "jarsura",
        "Tomáš Plecháč": "TPlechac",
        "Akciový Guru": "akciovyguru",
        "Nicnevim": "Nicnevim11",
        "Bulios": "Bulios_cz",
        "Michal Semotan": "MichalSemotan",
    },
    "Global & News": {
        "Walter Bloomberg (News)": "DeItaone",
        "Brian Feroldi (Education)": "BrianFeroldi",
        "App Economy Insights": "AppEconomyInsights",
    },
}

DATA_DIR = os.path.join(os.path.dirname(__file__), ".stock_picker_pro")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
MEMOS_PATH = os.path.join(DATA_DIR, "memos.json")

# Sector to peers mapping (expand as needed)
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
        "KOMB.PR": ["MONETA.PR", "JPM", "BAC"],  # Czech: Komerční banka
        "MONETA.PR": ["KOMB.PR", "JPM", "BAC"],  # Czech: Moneta Money Bank
    },
    "Communication Services": {
        "T": ["VZ", "TMUS"],
    },
    "Utilities": {
        "CEZ.PR": ["NEE", "DUK", "SO", "D"],  # Czech: ČEZ
        "NEE": ["DUK", "SO", "D", "AEP"],
    },
}

# Macro Calendar Events (Feb-Mar 2026)
MACRO_CALENDAR = [
    {"date": "2026-02-20", "event": "FOMC Minutes Release", "importance": "High"},
    {"date": "2026-03-06", "event": "US Employment Report (NFP)", "importance": "High"},
    {"date": "2026-03-11", "event": "US CPI (Inflation Data)", "importance": "High"},
    {"date": "2026-03-18", "event": "FOMC Meeting (Interest Rate Decision)", "importance": "Critical"},
    {"date": "2026-03-25", "event": "US GDP (Q4 2025 Final)", "importance": "Medium"},
]



# ============================================================================
# UTILITIES & QUANT LOGIC
# ============================================================================

def calculate_roic(info: Dict[str, Any]) -> Optional[float]:
    """Aproximace ROIC: NOPAT / (Debt + Equity)."""
    try:
        ebit = safe_float(info.get("ebitda")) # EBITDA jako proxy
        nopat = ebit * 0.79 if ebit else None # 21% US Tax proxy
        invested_capital = (safe_float(info.get("totalDebt")) or 0) + (safe_float(info.get("totalStockholderEquity")) or 0)
        return safe_div(nopat, invested_capital)
    except: return None

def detect_market_regime(price_history: pd.DataFrame) -> str:
    """Detekce režimu na základě volatility a trendu za 6 měsíců."""
    if price_history.empty or len(price_history) < 20: return "Stable / Neutral"
    returns = price_history['Close'].pct_change().dropna()
    vol = returns.std() * math.sqrt(252)
    avg_ret = returns.mean() * 252
    
    if vol > 0.28 and avg_ret < -0.10: return "High Volatility / Bear"
    if vol < 0.18 and avg_ret > 0.05: return "Low Volatility / Bull"
    return "Stable / Transition"
    
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
        if x is None:
            return None
        if isinstance(x, (np.generic,)):
            x = x.item()
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
        if isinstance(x, str):
            x = x.strip().replace(",", "")
            if x == "":
                return None
            v = float(x)
            if math.isfinite(v):
                return v
        return None
    except Exception:
        return None


def safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    a = safe_float(a)
    b = safe_float(b)
    if a is None or b is None or b == 0:
        return None
    return a / b


def fmt_num(x: Any, digits: int = 2) -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{v:,.{digits}f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{v*100:.{digits}f}%"


def fmt_money(x: Any, digits: int = 2, prefix: str = "$") -> str:
    v = safe_float(x)
    if v is None:
        return "—"
    return f"{prefix}{v:,.{digits}f}"


def clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


# ============================================================================
# DATA FETCHING (CACHED)
# ============================================================================

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """
    Získá info o firmě. Primárně z Yahoo, při neúspěchu zkusí FMP API (Backup).
    """
    # 1. Pokus: Yahoo Finance (yfinance)
    try:
        t = yf.Ticker(ticker)
        info = t.info
        # Yahoo občas vrátí prázdný dict nebo dict bez klíčových dat, i když nespadne
        if info and 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
            return info
    except Exception:
        pass # Ignorujeme chybu a jdeme na backup

    # 2. Pokus: Financial Modeling Prep (FMP) - Backup
    if FMP_API_KEY:
        try:
            # Endpoint v3/profile je zdarma a velmi spolehlivý
            url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={FMP_API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if data and isinstance(data, list) and len(data) > 0:
                fmp_data = data[0]
                
                # Převedeme FMP formát na formát, který očekává tvůj zbytek kódu (Yahoo style)
                # Tím zajistíme, že zbytek aplikace (grafy, metriky) bude fungovat
                return {
                    'longName': fmp_data.get('companyName'),
                    'symbol': fmp_data.get('symbol'),
                    'sector': fmp_data.get('sector'),
                    'industry': fmp_data.get('industry'),
                    'longBusinessSummary': fmp_data.get('description'),
                    'currentPrice': fmp_data.get('price'),
                    'regularMarketPrice': fmp_data.get('price'),
                    'marketCap': fmp_data.get('mktCap'),
                    'beta': fmp_data.get('beta'),
                    'currency': fmp_data.get('currency'),
                    'website': fmp_data.get('website'),
                    # FMP nemá přímo P/E v profilu, ale má cenu. Ostatní metriky (P/E) se dají dopočítat nebo nechat N/A
                    'trailingPE': None, # FMP má P/E v jiném endpointu (ratios), pro profil stačí základ
                    'dividendYield': None,
                    'returnOnEquity': None, 
                    'freeCashflow': None, # To si skript bere z cashflow statementu
                    'country': fmp_data.get('country')
                }
        except Exception as e:
            print(f"FMP Profile Error: {e}")

    # Pokud vše selže
    return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_price_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical price data."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=False)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_financials(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch income statement, balance sheet, and cash flow."""
    try:
        t = yf.Ticker(ticker)
        income = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow
        return (
            income if income is not None else pd.DataFrame(),
            balance if balance is not None else pd.DataFrame(),
            cashflow if cashflow is not None else pd.DataFrame()
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=3600)
def get_fcf_ttm_yfinance(ticker: str, market_cap: Optional[float] = None) -> Tuple[Optional[float], List[str]]:
    """Robustně spočítá roční Free Cash Flow (TTM) z yfinance quarterly_cashflow.

    Pravidla:
    - Primárně sečte poslední 4 dostupné kvartály (TTM).
    - Když chybí řádek 'Free Cash Flow', spočítá FCF jako Operating Cash Flow - |CapEx|.
    - Pokud jsou dostupná jen 1-3 kvartální čísla, annualizuje průměrem ×4.
    - Sanity check: pro obří firmy (MarketCap > $1T) a podezřele nízké FCF (< $30B)
      aplikuje pojistku násobení 4× (typicky když provider vrátí jen 1 kvartál).
    - Vrací (fcf_ttm, dbg) kde dbg je list informativních zpráv.
    """
    dbg: List[str] = []
    try:
        t = yf.Ticker(ticker)
        qcf = getattr(t, "quarterly_cashflow", None)
        if qcf is None or not isinstance(qcf, pd.DataFrame) or qcf.empty:
            dbg.append("FCF: quarterly_cashflow není k dispozici (prázdné). Zkouším fallback.")
            qcf = pd.DataFrame()

        def _pick_row(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            if df is None or df.empty:
                return None
            idx = set(map(str, df.index))
            for c in candidates:
                if c in idx:
                    return c
            # zkus case-insensitive match
            low_map = {str(i).strip().lower(): str(i) for i in df.index}
            for c in candidates:
                key = c.strip().lower()
                if key in low_map:
                    return low_map[key]
            return None

        def _sorted_quarter_cols(df: pd.DataFrame) -> List[Any]:
            cols = list(df.columns)
            if not cols:
                return []
            dts = pd.to_datetime(cols, errors="coerce")
            if dts.notna().any():
                order = sorted(range(len(cols)), key=lambda i: dts[i], reverse=True)
                return [cols[i] for i in order]
            return cols  # fallback: keep original order

        # 1) vyber poslední dostupné kvartály
        cols_sorted = _sorted_quarter_cols(qcf)
        cols_sel = cols_sorted[:4] if cols_sorted else []
        if cols_sel:
            dbg.append(f"FCF: Načítám kvartály: {', '.join([str(c) for c in cols_sel])}")
        else:
            dbg.append("FCF: Nenalezeny žádné kvartální sloupce v quarterly_cashflow.")

        # 2) primárně: přímý řádek Free Cash Flow
        fcf_row = _pick_row(qcf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
        used_method = None

        fcf_quarters = None
        non_null = 0

        if fcf_row and cols_sel:
            s = pd.to_numeric(qcf.loc[fcf_row, cols_sel], errors="coerce")
            non_null = int(s.notna().sum())
            if non_null > 0:
                fcf_quarters = s
                used_method = f"quarterly row '{fcf_row}'"
        # 3) fallback: OCF - |CapEx|
        if fcf_quarters is None and cols_sel:
            ocf_row = _pick_row(qcf, [
                "Operating Cash Flow",
                "Total Cash From Operating Activities",
                "Total Cash From Operating Activities (Continuing Operations)",
                "Cash Flow From Continuing Operating Activities",
                "Net Cash Provided By Operating Activities",
            ])
            capex_row = _pick_row(qcf, [
                "Capital Expenditures",
                "Capital Expenditure",
                "CapitalExpenditures",
                "Purchase Of PPE",
                "Purchase of Property Plant Equipment",
            ])
            if ocf_row and capex_row:
                ocf = pd.to_numeric(qcf.loc[ocf_row, cols_sel], errors="coerce")
                capex = pd.to_numeric(qcf.loc[capex_row, cols_sel], errors="coerce")
                non_null = int((ocf.notna() & capex.notna()).sum())
                if non_null > 0:
                    # CapEx bývá záporný; chceme: FCF = OCF - |CapEx|
                    fcf_quarters = ocf - capex.abs()
                    used_method = f"computed: '{ocf_row}' - |'{capex_row}'|"

        # 4) pokud pořád nic, fallback na annual cashflow / info
        if fcf_quarters is None:
            # annual cashflow
            acf = getattr(t, "cashflow", None)
            if isinstance(acf, pd.DataFrame) and not acf.empty:
                acf_cols = _sorted_quarter_cols(acf)[:1]  # nejnovější rok
                fcf_row_a = _pick_row(acf, ["Free Cash Flow", "FreeCashFlow", "Free cash flow"])
                if fcf_row_a and acf_cols:
                    v = safe_float(acf.loc[fcf_row_a, acf_cols[0]])
                    if v is not None:
                        dbg.append("FCF: Používám annual cashflow (nejnovější rok) – řádek Free Cash Flow.")
                        used_method = "annual row 'Free Cash Flow'"
                        fcf_ttm = float(v)
                        msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard ({used_method})"
                        dbg.append(msg)
                        print(msg)
                        return fcf_ttm, dbg

            # last resort: info['freeCashflow']
            try:
                info = getattr(t, "info", None) or {}
            except Exception:
                info = {}
            v = safe_float(info.get("freeCashflow"))
            if v is not None:
                used_method = "info['freeCashflow'] (fallback)"
                fcf_ttm = float(v)
                msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard ({used_method})"
                dbg.append(msg)
                print(msg)
                return fcf_ttm, dbg

            dbg.append("FCF: Nepodařilo se získat FCF ani z quarterly ani z annual ani z info.")
            return None, dbg

        # 5) TTM / extrapolace
        fcf_vals = pd.to_numeric(fcf_quarters, errors="coerce").dropna()
        n = int(fcf_vals.shape[0])
        applied_extrap = False
        used_sum4 = False

        if n >= 4:
            fcf_ttm = float(fcf_vals.iloc[:4].sum())
            used_sum4 = True
        elif n > 0:
            # annualizace průměrem ×4
            fcf_ttm = float(fcf_vals.mean() * 4.0)
            applied_extrap = True
        else:
            dbg.append("FCF: kvartální hodnoty jsou všechny NaN.")
            return None, dbg

        # 6) Sanity check (market cap > 1T & FCF < 30B) -> 4×
        mc = safe_float(market_cap)
        if (not applied_extrap) and used_sum4 and mc and mc > 1e12 and fcf_ttm < 30e9:
            fcf_ttm *= 4.0
            dbg.append("FCF: Sanity check aktivován (MarketCap > $1T a FCF < $30B) -> násobím 4× (podezření na 1 kvartál).")

        # 7) Debug zprávy
        if used_method:
            dbg.append(f"FCF metoda: {used_method}. Kvartály použity: {n}.")
        if applied_extrap:
            dbg.append(f"FCF: Extrapolace do roční báze (k dispozici {n} kvartály) -> průměr ×4.")
        if used_sum4:
            dbg.append("FCF: TTM = součet posledních 4 kvartálů.")

        msg = f"Použité roční FCF (TTM): ${fcf_ttm/1e9:.1f} miliard"
        dbg.append(msg)
        print(msg)

        return fcf_ttm, dbg
    except Exception as e:
        dbg.append(f"FCF: chyba při výpočtu TTM: {e}")
        return None, dbg
@st.cache_data(show_spinner=False, ttl=3600)
def get_all_time_high(ticker: str) -> Optional[float]:
    """Get all-time high price."""
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        col = "High" if "High" in h.columns else ("Close" if "Close" in h.columns else None)
        if not col:
            return None
        return float(pd.to_numeric(h[col], errors="coerce").max())
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_insider_transactions_fmp(ticker: str) -> Optional[pd.DataFrame]:
    """
    FALLBACK: Načítá insider obchody scrapováním Finvizu (protože API jsou placená).
    """
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    # Finviz vyžaduje User-Agent, aby si myslel, že jsme prohlížeč
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None
        
        # Pandas umí vycucnout všechny tabulky z HTML
        # Tabulka insiderů je obvykle ta poslední na stránce Finvizu
        dfs = pd.read_html(response.text)
        
        # Hledáme tabulku, která obsahuje slova 'Transaction' a 'SEC Form 4'
        insider_df = None
        for df in dfs:
            if 'Transaction' in df.columns and 'SEC Form 4' in df.columns:
                insider_df = df
                break
                
        # Pokud jsme tabulku nenašli, zkusíme vzít poslední (často to tak je)
        if insider_df is None and len(dfs) > 5:
            insider_df = dfs[-1]
            
        if insider_df is None or insider_df.empty:
            return pd.DataFrame()

        # --- ČIŠTĚNÍ DAT Z FINVIZU ---
        # Finviz má sloupce: [Owner, Relationship, Date, Transaction, Cost, #Shares, Value ($), #Shares Total, SEC Form 4]
        
        # Přejmenování pro kompatibilitu s tvým skriptem
        rename_map = {
            'Date': 'Date',
            'Transaction': 'Transaction',
            'Relationship': 'Position',
            'Value ($)': 'Value',
            '#Shares': 'Shares',
            'Cost': 'Price'
        }
        
        # Flexibilní přejmenování (ignoruje, co tam není)
        insider_df = insider_df.rename(columns=rename_map)
        
        # Filtrujeme jen sloupce, které potřebujeme
        needed_cols = ['Date', 'Transaction', 'Position', 'Value']
        
        # Pokud chybí Value, zkusíme ji dopočítat
        if 'Value' not in insider_df.columns and 'Shares' in insider_df.columns and 'Price' in insider_df.columns:
             # Čištění numerických dat (odstranění čárek)
             def clean_num(x):
                 if isinstance(x, str): return pd.to_numeric(x.replace(',', ''), errors='coerce')
                 return x
             
             insider_df['Value'] = clean_num(insider_df['Shares']) * clean_num(insider_df['Price'])

        # Finální formátování
        # Finviz datum je např. "Feb 13", musíme přidat rok (odhadem aktuální rok)
        def parse_finviz_date(d_str):
            try:
                # Přidáme aktuální rok, protože Finviz rok neuvádí
                current_year = dt.datetime.now().year
                return pd.to_datetime(f"{d_str} {current_year}", format="%b %d %Y")
            except:
                return pd.NaT

        insider_df['Date'] = insider_df['Date'].apply(parse_finviz_date)
        
        # Čištění sloupce Value (odstranění čárek)
        if 'Value' in insider_df.columns:
             insider_df['Value'] = insider_df['Value'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')

        return insider_df

    except Exception as e:
        print(f"Finviz Scraper Error: {e}")
        return pd.DataFrame() # Vrať prázdnou tabulku místo None, aby aplikace nepadala
        
        # --- FLEXIBILNÍ PŘEJMENOVÁNÍ ---
        # FMP API vrací různé názvy sloupců v různých verzích
        rename_map = {
            'transactionDate': 'Date', 'transaction_date': 'Date',
            'transactionType': 'Transaction', 'type': 'Transaction', 'acquistionOrDisposition': 'Transaction',
            'officerTitle': 'Position', 'reportingName': 'Position',
            'securitiesTransacted': 'Shares', 'securities_transacted': 'Shares',
            'price': 'Price', 'priceAtTransaction': 'Price'
        }
        
        df = df.rename(columns=rename_map)
        
        # Ověříme, zda máme klíčové sloupce, jinak je vytvoříme
        required_cols = ['Date', 'Transaction', 'Position', 'Shares', 'Price']
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0 if col in ['Shares', 'Price'] else "N/A"

        # Výpočet hodnoty (Value)
        df['Value'] = pd.to_numeric(df['Shares'], errors='coerce') * pd.to_numeric(df['Price'], errors='coerce')
        
        # Filtrujeme jen relevantní sloupce
        final_df = df[['Date', 'Transaction', 'Position', 'Value', 'Shares', 'Price']].dropna(subset=['Date'])
        
        return final_df

    except Exception as e:
        print(f"FMP API Error: {e}")
        return None

# ============================================================================
# METRICS & SCORING
# ============================================================================

@dataclass
class Metric:
    name: str
    value: Optional[float]
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    target_below: Optional[float] = None
    target_above: Optional[float] = None
    weight: float = 1.0
    source: str = "yfinance"


def extract_metrics(info: Dict[str, Any], ticker: str) -> Dict[str, Metric]:
    """Extract comprehensive metrics from Yahoo Finance info."""
    
    # Price metrics
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    
    # Valuation
    pe = safe_float(info.get("trailingPE"))
    pb = safe_float(info.get("priceToBook"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    peg = safe_float(info.get("pegRatio"))
    ev_ebitda = safe_float(info.get("enterpriseToEbitda"))
    
    # Profitability
    roe = safe_float(info.get("returnOnEquity"))
    roa = safe_float(info.get("returnOnAssets"))
    operating_margin = safe_float(info.get("operatingMargins"))
    profit_margin = safe_float(info.get("profitMargins"))
    gross_margin = safe_float(info.get("grossMargins"))
    
    # Growth
    revenue_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))
    earnings_quarterly_growth = safe_float(info.get("earningsQuarterlyGrowth"))
    
    # Financial health
    current_ratio = safe_float(info.get("currentRatio"))
    quick_ratio = safe_float(info.get("quickRatio"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    total_cash = safe_float(info.get("totalCash"))
    total_debt = safe_float(info.get("totalDebt"))
    
    # Cash flow
    operating_cashflow = safe_float(info.get("operatingCashflow"))
    market_cap = safe_float(info.get('marketCap'))
    fcf, _fcf_dbg = get_fcf_ttm_yfinance(ticker, market_cap)
    fcf_yield = safe_div(fcf, market_cap) if fcf and market_cap else None
    
    # Analyst targets
    target_mean = safe_float(info.get("targetMeanPrice"))
    target_median = safe_float(info.get("targetMedianPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    target_low = safe_float(info.get("targetLowPrice"))
    recommendation = info.get("recommendationKey", "")
    
    # Dividend
    dividend_yield = safe_float(info.get("dividendYield"))
    payout_ratio = safe_float(info.get("payoutRatio"))
    
    metrics = {
        "price": Metric("Current Price", price),
        "pe": Metric("P/E Ratio", pe, target_below=25, weight=1.5),
        "pb": Metric("P/B Ratio", pb, target_below=3, weight=1.0),
        "ps": Metric("P/S Ratio", ps, target_below=2, weight=1.0),
        "peg": Metric("PEG Ratio", peg, target_below=1.5, weight=1.5),
        "ev_ebitda": Metric("EV/EBITDA", ev_ebitda, target_below=15, weight=1.0),
        "roe": Metric("ROE", roe, target_above=0.15, weight=2.0),
        "roa": Metric("ROA", roa, target_above=0.05, weight=1.0),
        "operating_margin": Metric("Operating Margin", operating_margin, target_above=0.15, weight=1.5),
        "profit_margin": Metric("Profit Margin", profit_margin, target_above=0.10, weight=1.5),
        "gross_margin": Metric("Gross Margin", gross_margin, target_above=0.30, weight=1.0),
        "revenue_growth": Metric("Revenue Growth", revenue_growth, target_above=0.10, weight=2.0),
        "earnings_growth": Metric("Earnings Growth", earnings_growth, target_above=0.10, weight=2.0),
        "current_ratio": Metric("Current Ratio", current_ratio, target_above=1.5, weight=1.0),
        "quick_ratio": Metric("Quick Ratio", quick_ratio, target_above=1.0, weight=0.8),
        "debt_to_equity": Metric("Debt/Equity", debt_to_equity, target_below=1.0, weight=1.5),
        "fcf_yield": Metric("FCF Yield", fcf_yield, target_above=0.05, weight=2.0),
        "dividend_yield": Metric("Dividend Yield", dividend_yield, target_above=0.02, weight=0.5),
        "payout_ratio": Metric("Payout Ratio", payout_ratio, target_below=0.70, weight=0.5),
        "target_mean": Metric("Analyst Target (Mean)", target_mean),
        "target_median": Metric("Analyst Target (Median)", target_median),
        "target_high": Metric("Analyst Target (High)", target_high),
        "target_low": Metric("Analyst Target (Low)", target_low),
    }
    
    return metrics


def calculate_metric_score(metric: Metric) -> float:
    """Calculate 0-10 score for a single metric."""
    if metric.value is None:
        return 5.0
    
    val = metric.value
    
    # Target below (lower is better)
    if metric.target_below is not None:
        if val <= metric.target_below * 0.7:
            return 10.0
        elif val <= metric.target_below:
            return 8.0
        elif val <= metric.target_below * 1.5:
            return 5.0
        else:
            return 2.0
    
    # Target above (higher is better)
    if metric.target_above is not None:
        if val >= metric.target_above * 1.5:
            return 10.0
        elif val >= metric.target_above:
            return 8.0
        elif val >= metric.target_above * 0.5:
            return 5.0
        else:
            return 2.0
    
    return 5.0


def build_scorecard_advanced(metrics: Dict[str, Metric], info: Dict[str, Any]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Build advanced scorecard (0-100) with category breakdown.
    Returns: (total_score, category_scores, individual_metric_scores)
    """
    
    # Category definitions
    categories = {
        "Valuace": ["pe", "pb", "ps", "peg", "ev_ebitda"],
        "Kvalita": ["roe", "roa", "operating_margin", "profit_margin", "gross_margin"],
        "Růst": ["revenue_growth", "earnings_growth"],
        "Fin. zdraví": ["current_ratio", "quick_ratio", "debt_to_equity", "fcf_yield"],
    }
    
    category_scores = {}
    individual_scores = {}
    
    for cat_name, metric_keys in categories.items():
        cat_scores = []
        for key in metric_keys:
            metric = metrics.get(key)
            if metric and metric.weight > 0:
                score = calculate_metric_score(metric)
                individual_scores[metric.name] = score
                cat_scores.append((score, metric.weight))
        
        if cat_scores:
            weighted_sum = sum(s * w for s, w in cat_scores)
            total_weight = sum(w for _, w in cat_scores)
            category_scores[cat_name] = (weighted_sum / total_weight) * 10  # Scale to 0-100
        else:
            category_scores[cat_name] = 50.0
    
    # Overall score (equal weight per category)
    total_score = sum(category_scores.values()) / len(category_scores)
    
    return total_score, category_scores, individual_scores


# ============================================================================
# DCF VALUATION
# ============================================================================

def calculate_dcf_fair_value(
    fcf: float,
    growth_rate: float = 0.10,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: Optional[float] = None
) -> Optional[float]:
    """DCF calculation."""
    if fcf <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None
    
    try:
        pv_sum = 0.0
        current_fcf = fcf
        
        for year in range(1, years + 1):
            current_fcf *= (1 + growth_rate)
            pv_sum += current_fcf / ((1 + wacc) ** year)
        
        terminal_fcf = current_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** years)
        
        enterprise_value = pv_sum + pv_terminal
        fair_value_per_share = enterprise_value / shares_outstanding
        
        return fair_value_per_share
    except Exception:
        return None


def reverse_dcf_implied_growth(
    current_price: float,
    fcf: float,
    terminal_growth: float = 0.03,
    wacc: float = 0.10,
    years: int = 5,
    shares_outstanding: Optional[float] = None
) -> Optional[float]:
    """Calculate implied growth rate from current price."""
    if fcf <= 0 or shares_outstanding is None or shares_outstanding <= 0:
        return None
    
    try:
        def dcf_at_growth(g: float) -> float:
            fv = calculate_dcf_fair_value(fcf, g, terminal_growth, wacc, years, shares_outstanding)
            return fv if fv else 0.0
        
        low, high = -0.5, 1.0
        for _ in range(50):
            mid = (low + high) / 2.0
            fv = dcf_at_growth(mid)
            if abs(fv - current_price) < 0.01:
                return mid
            if fv < current_price:
                low = mid
            else:
                high = mid
        
        return (low + high) / 2.0
    except Exception:
        return None


# ============================================================================
# INSIDER TRADING ANALYSIS
# ============================================================================

def compute_insider_pro_signal(insider_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Advanced insider trading signal with role weighting and cluster detection.
    
    Returns:
    {
        "signal": float (-100 to +100),
        "label": str (Strong Buy / Buy / Neutral / Sell / Strong Sell),
        "confidence": float (0-1),
        "insights": List[str],
        "recent_buys": int,
        "recent_sells": int,
        "cluster_detected": bool
    }
    """
    
    if insider_df is None or insider_df.empty:
        return {
            "signal": 0,
            "label": "Neutral",
            "confidence": 0.0,
            "insights": ["Žádné insider transakce k dispozici"],
            "recent_buys": 0,
            "recent_sells": 0,
            "cluster_detected": False
        }
    
    # Role weights (CEO/CFO matter more)
    role_weights = {
        "ceo": 3.0,
        "chief executive officer": 3.0,
        "cfo": 2.5,
        "chief financial officer": 2.5,
        "president": 2.0,
        "director": 1.5,
        "coo": 2.0,
        "vice president": 1.2,
        "officer": 1.0,
    }
    
    # Analyze recent transactions (last 6 months)
    cutoff_date = dt.datetime.now() - dt.timedelta(days=180)
    
    buy_signal = 0.0
    sell_signal = 0.0
    buy_count = 0
    sell_count = 0
    buy_dates = []
    sell_dates = []
    
    for _, row in insider_df.iterrows():
        try:
            # Parse date
            date_str = row.get("Start Date", row.get("Date", ""))
            if pd.isna(date_str):
                continue
            
            trans_date = pd.to_datetime(date_str)
            if trans_date < cutoff_date:
                continue
            
            # Get transaction type
            transaction = str(row.get("Transaction", "")).lower()
            value = safe_float(row.get("Value", 0))
            if value is None:
                value = 0
            
            # Get role weight
            position = str(row.get("Position", "")).lower()
            weight = 1.0
            for role, w in role_weights.items():
                if role in position:
                    weight = max(weight, w)
            
            # Classify transaction
            if "buy" in transaction or "purchase" in transaction:
                buy_signal += value * weight
                buy_count += 1
                buy_dates.append(trans_date)
            elif "sell" in transaction or "sale" in transaction:
                # Filter out tax-related sells
                if "tax" not in transaction and "10b5-1" not in transaction:
                    sell_signal += value * weight
                    sell_count += 1
                    sell_dates.append(trans_date)
        
        except Exception:
            continue
    
    # Detect cluster buying (multiple insiders buying within 30 days)
    cluster_detected = False
    if len(buy_dates) >= 3:
        buy_dates_sorted = sorted(buy_dates)
        for i in range(len(buy_dates_sorted) - 2):
            if (buy_dates_sorted[i+2] - buy_dates_sorted[i]).days <= 30:
                cluster_detected = True
                break
    
    # Calculate signal (-100 to +100)
    net_signal = buy_signal - sell_signal
    max_signal = max(buy_signal + sell_signal, 1.0)
    signal = (net_signal / max_signal) * 100
    
    # Boost for cluster buying
    if cluster_detected:
        signal = min(100, signal * 1.3)
    
    # Determine label
    if signal >= 50:
        label = "Strong Buy"
    elif signal >= 20:
        label = "Buy"
    elif signal >= -20:
        label = "Neutral"
    elif signal >= -50:
        label = "Sell"
    else:
        label = "Strong Sell"
    
    # Confidence based on transaction count and recency
    confidence = min(1.0, (buy_count + sell_count) / 10.0)
    
    # Generate insights
    insights = []
    if buy_count > 0:
        insights.append(f"✅ {buy_count} insider nákupů v posledních 6 měsících")
    if sell_count > 0:
        insights.append(f"⚠️ {sell_count} insider prodejů v posledních 6 měsících")
    if cluster_detected:
        insights.append(f"🔥 Cluster buying detekován - více insiderů nakupuje současně!")
    if signal > 30:
        insights.append(f"💪 Silný bullish signál od insiderů ({signal:.0f}/100)")
    elif signal < -30:
        insights.append(f"📉 Silný bearish signál od insiderů ({signal:.0f}/100)")
    
    return {
        "signal": signal,
        "label": label,
        "confidence": confidence,
        "insights": insights if insights else ["Žádné významné insider aktivity"],
        "recent_buys": buy_count,
        "recent_sells": sell_count,
        "cluster_detected": cluster_detected
    }


# ============================================================================
# PEER COMPARISON
# ============================================================================

def get_auto_peers(ticker: str, sector: str, info: Dict[str, Any]) -> List[str]:
    """
    Automaticky najde 3-5 konkurentů na základě tickeru a sektoru.
    """
    
    # 1) Check manual mapping first
    for sect, tickers_map in SECTOR_PEERS.items():
        if ticker in tickers_map:
            return tickers_map[ticker][:5]
    
    # 2) Try to find similar companies in the same sector
    # (In production, you'd use API like FMP or screen by market cap + industry)
    # For now, return placeholder
    
    return []


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_peer_comparison(ticker: str, peers: List[str]) -> pd.DataFrame:
    """
    Fetch comparison metrics for ticker and its peers.
    Returns DataFrame with columns: Ticker, P/E, Op. Margin, Rev. Growth, FCF Yield
    """
    
    all_tickers = [ticker] + peers
    rows = []
    
    for t in all_tickers:
        try:
            info = fetch_ticker_info(t)
            if not info:
                continue
            
            mc = safe_float(info.get('marketCap'))
            fcf_ttm_peer, _ = get_fcf_ttm_yfinance(t, mc)
            fcf_yield_peer = safe_div(fcf_ttm_peer, mc) if fcf_ttm_peer and mc else None

            rows.append({
                "Ticker": t,
                "P/E": safe_float(info.get("trailingPE")),
                "Op. Margin": safe_float(info.get("operatingMargins")),
                "Rev. Growth": safe_float(info.get("revenueGrowth")),
                "FCF Yield": fcf_yield_peer,
                "Market Cap": mc,
            })
        except Exception:
            continue
    
    return pd.DataFrame(rows)


# ============================================================================
# AI ANALYST (GEMINI)
# ============================================================================

def generate_ai_analyst_report_with_retry(ticker: str, company: str, info: Dict, metrics: Dict, 
                             dcf_fair_value: float, current_price: float, 
                             scorecard: float, macro_events: List[Dict], insider_signal: Any = None) -> Dict:
    """
    Wrapper s retry logikou pro Free Tier Gemini 2.5 Flash Lite.
    Zkusí max MAX_AI_RETRIES pokusů s RETRY_DELAY sekundami mezi pokusy.
    """
    for attempt in range(MAX_AI_RETRIES):
        try:
            result = generate_ai_analyst_report(ticker, company, info, metrics, 
                                              dcf_fair_value, current_price, 
                                              scorecard, macro_events, insider_signal)
            
            # Check if result indicates an error that should trigger retry
            if "Chyba AI analýzy" in result.get("market_situation", ""):
                error_msg = result["market_situation"]
                # Check for rate limit errors
                if any(keyword in error_msg.lower() for keyword in ["429", "quota", "rate limit", "too many"]):
                    if attempt < MAX_AI_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        result["market_situation"] = "⚠️ AI je přetížená (Rate Limit). Zkuste to za chvíli."
                        return result
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            # Check for rate limit errors
            if any(keyword in error_msg.lower() for keyword in ["429", "quota", "rate limit", "too many"]):
                if attempt < MAX_AI_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return {
                        "market_situation": "⚠️ AI je přetížená (Rate Limit). Zkuste to za chvíli.",
                        "bull_case": [],
                        "bear_case": [],
                        "verdict": "HOLD",
                        "wait_for_price": current_price,
                        "reasoning": "Rate limit překročen i po několika pokusech.",
                        "confidence": "LOW"
                    }
            else:
                # Non-rate-limit error - don't retry
                return {
                    "market_situation": f"Chyba AI analýzy: {error_msg}",
                    "bull_case": [],
                    "bear_case": [],
                    "verdict": "HOLD",
                    "wait_for_price": current_price,
                    "reasoning": "Selhalo spojení s Gemini API.",
                    "confidence": "LOW"
                }
    
    # Fallback (shouldn't reach here)
    return {
        "market_situation": "⚠️ AI selhala po všech pokusech.",
        "bull_case": [],
        "bear_case": [],
        "verdict": "HOLD",
        "wait_for_price": current_price,
        "reasoning": "Maximální počet pokusů vyčerpán.",
        "confidence": "LOW"
    }


def generate_ai_analyst_report(ticker: str, company: str, info: Dict, metrics: Dict, 
                               dcf_fair_value: float, current_price: float, 
                               scorecard: float, macro_events: List[Dict], insider_signal: Any = None) -> Dict:
    """
    Generuje hloubkovou asymetrickou analýzu pomocí Gemini.
    """
    if not GEMINI_API_KEY:
        return {"market_situation": "Chybí API klíč.", "verdict": "N/A"}

    # 1. URČENÍ JAZYKA (Pojistka proti vietnamštině)
    target_lang = "ČEŠTINĚ" if st.session_state.get("language") == "cz" else "ANGLIČTINĚ"

    # 2. PŘÍPRAVA DAT
    roic_val = calculate_roic(info) 
    regime = detect_market_regime(fetch_price_history(ticker, "6mo"))
    debt_ebitda = safe_div(info.get("totalDebt"), info.get("ebitda"))
    fcf_yield_val = metrics.get("fcf_yield").value if metrics.get("fcf_yield") else 0

    # 3. SESTAVENÍ PROMPTU (Tady byla ta chyba v odsazení)
    context = f"""
Jsi Seniorní Portfolio Manažer a Contrarian Analyst se specializací na ASYMETRICKÝ RISK/REWARD.
DŮLEŽITÉ: Celou analýzu a všechny texty v JSON výstupu napiš v {target_lang}.

VSTUPNÍ DATA:
- Aktiva: {company} ({ticker}) | Sektor: {info.get('sector')} / {info.get('industry')}
- Tržní cena: {fmt_money(current_price)} | Kalkulovaná Férovka (DCF): {fmt_money(dcf_fair_value)}
- Metriky: P/E: {info.get('trailingPE')}, ROIC: {fmt_pct(roic_val)}, Net Debt/EBITDA: {fmt_num(debt_ebitda)}, FCF Yield: {fmt_pct(fcf_yield_val)}
- Tržní Režim: {regime}
- Makro události: {macro_events[:2]}

TVŮJ ANALYTICKÝ RÁMEC (Chain-of-Thought):
1. FUNDAMENTÁLNÍ PODLAHA: Je cena blízko hodnotě aktiv? Jak bezpečný je dluh?
2. EMBEDDED OPTIONALITY: Má firma aktiva (data, patenty), která trh oceňuje nulou?
3. RED TEAMING: Hraj roli Short Sellera. Proč tato firma za 2 roky ztratí 50 % hodnoty?
4. ASYMETRIE: Je poměr mezi Downside a Upside alespoň 1:3?

VÝSTUP POUZE JSON:
{{
  "asymmetry_score": (číslo 0-100),
  "fundamental_floor": "Analýza bezpečnosti investice jednou větou.",
  "red_team_warning": "BRUTÁLNĚ upřímná analýza největšího rizika - proč to nekoupit.",
  "bull_case": ["Argument 1", "Argument 2"],
  "bear_case": ["Riziko 1", "Riziko 2"],
  "verdict": "STRONGBUY/BUY/HOLD/SELL/AVOID",
  "wait_for_price": {current_price * 0.85 if current_price else 0},
  "risk_reward_ratio": "Např. 1:4",
  "reasoning_synthesis": "Konečný verdikt pro investiční komisi. Proč právě teď?",
  "confidence": "HIGH/MEDIUM/LOW"
}}
"""

    # 4. PARSOVÁNÍ JSONU
    def _extract_json(text: str) -> Dict[str, Any]:
        if not text: raise ValueError("Empty AI response")
        cleaned = re.sub(r"```json\n?|```", "", str(text)).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if not m: raise
            return json.loads(m.group(0))

    # 5. VOLÁNÍ API
    try:
        raw_text = ""
        try:
            from google import genai as genai_new
            client = genai_new.Client(api_key=GEMINI_API_KEY)
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=context)
            raw_text = getattr(resp, "text", None) or str(resp)
        except Exception:
            import google.generativeai as genai_legacy
            genai_legacy.configure(api_key=GEMINI_API_KEY)
            model = genai_legacy.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(context)
            raw_text = getattr(resp, "text", None) or str(resp)

        return _extract_json(raw_text)

    except Exception as e:
        return {
            "market_situation": f"Chyba AI: {str(e)}", 
            "bull_case": [], "bear_case": [], 
            "verdict": "HOLD", "wait_for_price": current_price
        }
        


    def _extract_json(text: str) -> Dict[str, Any]:
        """Try hard to parse JSON from model output."""
        if not text:
            raise ValueError("Empty AI response")
        cleaned = re.sub(r"```json\n?|```", "", str(text)).strip()
        try:
            return json.loads(cleaned)
        except Exception:
            # Fallback: find first JSON object in the text
            m = re.search(r"\{[\s\S]*\}", cleaned)
            if not m:
                raise
            return json.loads(m.group(0))

    try:
        raw_text = ""

        # Prefer new Google GenAI SDK (google-genai)
        try:
            from google import genai as genai_new  # type: ignore
            client = genai_new.Client(api_key=GEMINI_API_KEY)
            try:
                from google.genai import types as genai_types  # type: ignore
                cfg = genai_types.GenerateContentConfig(response_mime_type="application/json")
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=context, config=cfg)
            except Exception:
                # Older/newer variants of the SDK
                resp = client.models.generate_content(model=GEMINI_MODEL, contents=context)

            raw_text = getattr(resp, "text", None) or str(resp)

        except Exception:
            # Fallback to legacy SDK (google-generativeai)
            import google.generativeai as genai_legacy  # type: ignore
            genai_legacy.configure(api_key=GEMINI_API_KEY)
            model = genai_legacy.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(context)
            raw_text = getattr(resp, "text", None) or str(resp)

        data = _extract_json(raw_text)

        # Normalize/validate keys expected by the UI
        required = ["market_situation", "bull_case", "bear_case", "verdict", "wait_for_price"]
        for k in required:
            if k not in data:
                data[k] = "N/A" if k != "wait_for_price" else current_price

        # Optional extras (keep UI stable even if missing)
        if "reasoning" not in data:
            data["reasoning"] = ""
        if "confidence" not in data:
            data["confidence"] = "MEDIUM"

        return data

    except Exception as e:
        return {
            "market_situation": f"Chyba AI analýzy: {str(e)}",
            "bull_case": [],
            "bear_case": [],
            "verdict": "HOLD",
            "wait_for_price": current_price,
            "reasoning": "Selhalo spojení s Gemini API nebo parsování JSON.",
            "confidence": "LOW"
        }

def get_earnings_calendar_estimate(ticker: str, info: Dict[str, Any]) -> Optional[dt.date]:
    """
    Estimate next earnings date based on historical pattern.
    Most companies report quarterly, roughly same time each quarter.
    """
    try:
        t = yf.Ticker(ticker)
        calendar = getattr(t, "calendar", None)
        if calendar is not None and not calendar.empty:
            # Look for "Earnings Date" row
            if "Earnings Date" in calendar.index:
                next_earnings = calendar.loc["Earnings Date"].iloc[0]
                if pd.notna(next_earnings):
                    return pd.to_datetime(next_earnings).date()
    except Exception:
        pass
    
    # Fallback: Estimate based on common patterns (most tech companies: late Jan, late Apr, late Jul, late Oct)
    today = dt.date.today()
    # Simple heuristic: next month-end
    if today.month < 4:
        return dt.date(today.year, 4, 25)
    elif today.month < 7:
        return dt.date(today.year, 7, 25)
    elif today.month < 10:
        return dt.date(today.year, 10, 25)
    else:
        return dt.date(today.year + 1, 1, 25)


# ============================================================================
# WATCHLIST & MEMOS
# ============================================================================

def get_watchlist() -> Dict[str, Any]:
    return load_json(WATCHLIST_PATH, {"items": {}})


def set_watchlist(data: Dict[str, Any]) -> None:
    save_json(WATCHLIST_PATH, data)


def get_memos() -> Dict[str, Any]:
    return load_json(MEMOS_PATH, {"memos": {}})


def set_memos(data: Dict[str, Any]) -> None:
    save_json(MEMOS_PATH, data)


# ============================================================================
# PDF EXPORT
# ============================================================================

def export_memo_pdf(ticker: str, company: str, memo: Dict[str, str], summary: Dict[str, str]) -> Optional[bytes]:
    """Export memo to PDF."""
    if not _HAS_PDF:
        return None
    
    try:
        from io import BytesIO
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        c.setFont("Helvetica-Bold", 16)
        c.drawString(1*inch, 10*inch, f"Investment Memo: {company} ({ticker})")
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(1*inch, 9.5*inch, "Summary")
        c.setFont("Helvetica", 10)
        y = 9.2*inch
        for key, val in summary.items():
            c.drawString(1*inch, y, f"{key}: {val}")
            y -= 0.2*inch
        
        y -= 0.3*inch
        sections = [
            ("Thesis", memo.get("thesis", "")),
            ("Key Drivers", memo.get("drivers", "")),
            ("Risks", memo.get("risks", "")),
            ("Catalysts", memo.get("catalysts", "")),
            ("Buy Conditions", memo.get("buy_conditions", "")),
            ("Notes", memo.get("notes", ""))
        ]
        
        for title, content in sections:
            if y < 2*inch:
                c.showPage()
                y = 10*inch
            
            c.setFont("Helvetica-Bold", 11)
            c.drawString(1*inch, y, title)
            y -= 0.2*inch
            
            c.setFont("Helvetica", 9)
            lines = content.split('\n')
            for line in lines[:10]:
                if y < 1*inch:
                    break
                c.drawString(1.2*inch, y, line[:80])
                y -= 0.15*inch
            y -= 0.2*inch
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    except Exception:
        return None


# ============================================================================
# VERDICT LOGIC
# ============================================================================



def detect_value_trap(info: Dict[str, Any], metrics: Dict[str, Metric]) -> Tuple[bool, str]:
    """
    Detekce potenciální "pasti na hodnotu".
    
    Returns:
        (is_trap, warning_message)
    """
    pe = metrics.get("marketCap").value if metrics.get("marketCap") else None
    revenue_growth = metrics.get("marketCap").value if metrics.get("marketCap") else None
    debt_to_equity = metrics.get("marketCap").value if metrics.get("marketCap") else None
    eps = safe_float(info.get("trailingEps"))
    
    is_trap = False
    warnings = []
    
    # Podmínka 1: Nízké P/E (< 10)
    if pe and pe < 10:
        # Podmínka 2: Klesající tržby
        if revenue_growth and revenue_growth < -0.05:
            is_trap = True
            warnings.append("Klesající tržby (YoY)")
        
        # Podmínka 3: Vysoký dluh
        if debt_to_equity and debt_to_equity > 200:
            is_trap = True
            warnings.append("Vysoká zadluženost (D/E > 2)")
        
        # Podmínka 4: Negativní EPS
        if eps and eps <= 0:
            is_trap = True
            warnings.append("Negativní/nulové EPS")
    
    if is_trap:
        warning_msg = f"⚠️ **Potenciální Value Trap**: {', '.join(warnings)}. Nízká valuace může být oprávněná kvůli úpadku byznysu."
        return True, warning_msg
    
    return False, ""


def get_advanced_verdict(
    scorecard: float,
    mos_dcf: Optional[float],
    mos_analyst: Optional[float],
    insider_signal: float,
    implied_growth: Optional[float]
) -> Tuple[str, str, List[str]]:
    """
    Advanced verdict with multiple signals.
    
    Returns: (verdict, color, warnings)
    """
    
    warnings = []
    
    # Base verdict from scorecard
    if scorecard >= 85:
        base = "STRONG BUY"
        color = "#00ff88"
    elif scorecard >= 60:
        base = "BUY"
        color = "#88ff00"
    elif scorecard >= 45:
        base = "HOLD"
        color = "#ffaa00"
    elif scorecard >= 30:
        base = "CAUTION"
        color = "#ff8800"
    else:
        base = "AVOID"
        color = "#ff4444"
    
    # Adjust for MOS
    if mos_dcf is not None:
        if mos_dcf >= 0.20:
            if base in ["HOLD", "CAUTION"]:
                base = "BUY"
                color = "#88ff00"
        elif mos_dcf < -0.15:
            if base in ["STRONG BUY", "BUY"]:
                base = "HOLD"
                color = "#ffaa00"
                warnings.append("⚠️ DCF model ukazuje přeceněnost (-15% MOS)")
    
    # Check for mismatch: Analysts bullish but DCF says overvalued
    if mos_analyst is not None and mos_dcf is not None:
        if mos_analyst > 0.15 and mos_dcf < -0.10:
            warnings.append("🚨 MISMATCH WARNING: Analytici vidí upside +15%, ale DCF model ukazuje overvalued -10%!")
            warnings.append("   → Trh možná implikuje vyšší růst než je ve tvém DCF modelu konzervativní")
    
    # Insider signal adjustment
    if insider_signal > 50:
        warnings.append(f"✅ Silný insider buying signal (+{insider_signal:.0f}) podporuje BUY tezi")
    elif insider_signal < -30:
        warnings.append(f"⚠️ Negativní insider selling signal ({insider_signal:.0f})")
    
    # Implied growth check
    if implied_growth is not None:
        if implied_growth > 0.25:
            warnings.append(f"⚠️ Trh implikuje velmi agresivní růst FCF ({implied_growth*100:.0f}% ročně) - vysoká očekávání!")
        elif implied_growth < 0:
            warnings.append(f"📉 Trh implikuje pokles FCF ({implied_growth*100:.0f}%) - možná undervalued opportunity")
    
    return base, color, warnings


# End of Part 1
# ============================================================================
# MAIN APPLICATION
# ============================================================================


def render_twitter_timeline(handle: str, height: int = 600) -> None:
    """Render X/Twitter content without embeds (widgets are often blocked)."""
    handle = (handle or "").lstrip("@").strip()
    if not handle:
        st.info("Vyber guru účet.")
        return
    st.warning("⚠️ X (Twitter) často blokuje náhledy v cizích aplikacích. Použij přímý odkaz níže.")
    st.markdown(f"👉 Otevřít profil **@{handle}**: https://twitter.com/{handle}")

def analyze_social_text_with_gemini(text: str) -> str:
    """Analyze manually pasted tweet/comment using Gemini."""
    text = (text or "").strip()
    if not text:
        return "Chybí text k analýze."

    if not GEMINI_API_KEY:
        return "AI analýza není dostupná (chybí GEMINI_API_KEY)."

    prompt = f"""Jako seniorní investor analyzuj tento text z sociálních sítí týkající se financí.

1) Jaký je sentiment (Bullish/Bearish/Neutral)?
2) Jsou tam nějaká fakta nebo jen šum?
3) Verdikt pro investora.

TEXT:
{text}
"""

    try:
        # Try new google-genai SDK first
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
            return (response.text or "").strip()
        except ImportError:
            # Fallback to old SDK
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()
    except Exception as e:
        return f"Chyba při volání Gemini: {e}"


# -----------------------------------------------------------------------------
# Smart parameter estimation (Quality Premium)
# -----------------------------------------------------------------------------
def estimate_smart_params(info: Dict[str, Any], metrics: Dict[str, "Metric"]) -> Dict[str, Any]:
    """
    Konzervativní odhad DCF parametrů.
    Cíl: Zabránit "úletům" u Mega Caps (MSFT, AAPL) a opravit Amazon.
    """
    market_cap = safe_float(info.get('marketCap')) or 0.0
    sector = str(info.get('sector') or "").strip()
    
    # 1. DEFINICE VELIKOSTI
    is_mega_cap = market_cap > 200e9  # > 200 mld USD
    is_large_cap = market_cap > 50e9   # > 50 mld USD

    # 2. WACC (Diskontní sazba)
    # Zvedáme "podlahu" na 9.0% pro větší bezpečnost
    beta = safe_float(info.get("beta"))
    if beta is None or beta <= 0:
        base_wacc = 0.10
    else:
        # RiskFree (4.2%) + Beta * ERP (5.0%)
        base_wacc = 0.042 + (beta * 0.05)
    
    # Omezení WACC: Min 9%, Max 15%
    wacc = max(0.09, min(0.15, base_wacc))
    
    # Size Premium: Malé firmy jsou rizikovější -> přidáme 1.5%
    if market_cap < 10e9 and market_cap > 0:
        wacc += 0.015

    # 3. RŮST (Weighted Growth)
    # Vážíme tržby (70%) a zisky (30%), protože tržby jsou stabilnější
    rev_g = None
    earn_g = None
    try:
        if metrics.get("revenue_growth") and metrics["revenue_growth"].value is not None:
            rev_g = float(metrics["revenue_growth"].value)
    except:
        pass
    
    try:
        if metrics.get("earnings_growth") and metrics["earnings_growth"].value is not None:
            earn_g = float(metrics["earnings_growth"].value)
    except:
        pass

    # Výpočet váženého růstu
    if rev_g is not None and earn_g is not None:
        raw_growth = (0.7 * rev_g) + (0.3 * earn_g)
    elif rev_g is not None:
        raw_growth = rev_g
    elif earn_g is not None:
        raw_growth = earn_g
    else:
        raw_growth = 0.10  # Fallback

    # 4. STROP RŮSTU (Growth Cap) - Tady se krotí ty "brutální" čísla
    if is_mega_cap:
        # Giganti nemohou růst o 20% věčně -> Cap 12%
        growth_cap = 0.08
    elif is_large_cap:
        growth_cap = 0.12
    else:
        # Malé dravé firmy mohou růst rychleji
        growth_cap = 0.20
        
    growth = max(0.03, min(growth_cap, raw_growth))

    # 5. EXIT MULTIPLE (Konzervativní)
    # Základ podle sektoru
    sector_l = sector.lower()
    
    if "technology" in sector_l:
        base_multiple = 20.0
    elif "communication" in sector_l:  # Google, Meta
        base_multiple = 18.0
    elif "consumer cyclical" in sector_l:  # Amazon, Tesla
        base_multiple = 20.0
    elif "financial" in sector_l or "energy" in sector_l:
        base_multiple = 12.0
    elif "healthcare" in sector_l:
        base_multiple = 18.0
    else:
        base_multiple = 15.0
        
    # === PLYNULÝ QUALITY PREMIUM - BODOVÝ SYSTÉM ===
    quality_score = 0
    
    # ROE > 15% → +2 body, > 10% → +1 bod
    roe = safe_float(metrics.get("roe").value) if metrics.get("roe") else 0
    if roe > 0.15:
        quality_score += 2
    elif roe > 0.10:
        quality_score += 1
    
    # Net Margin > 20% → +2 body, > 10% → +1 bod
    pm = safe_float(metrics.get("profit_margin").value) if metrics.get("profit_margin") else 0
    if pm > 0.20:
        quality_score += 2
    elif pm > 0.10:
        quality_score += 1
    
    # ROIC (aproximace pomocí ROA) > 15% → +2 body, > 10% → +1 bod
    roa = safe_float(metrics.get("roa").value) if metrics.get("roa") else 0
    if roa > 0.15:
        quality_score += 2
    elif roa > 0.10:
        quality_score += 1
    
    # Debt/Equity < 0.5 (50) → +1 bod
    debt_eq = safe_float(metrics.get("debt_to_equity").value) if metrics.get("debt_to_equity") else 100
    if debt_eq < 50:
        quality_score += 1
    
    # Konverze bodů na Exit Multiple: Base + score, max 25x
    exit_multiple = base_multiple + quality_score
    exit_multiple = min(25.0, exit_multiple)

    return {
        "wacc": float(wacc),
        "growth": float(growth),
        "exit_multiple": float(exit_multiple),
        "is_mega_cap": bool(is_mega_cap),
        "market_cap": float(market_cap),
        "sector": sector
    }


# -------------------------------------------------------------------
# i18n helper (minimal)
# -------------------------------------------------------------------
TRANSLATIONS = {
    "cs": {
        "app_name": "Stock Picker Pro",
        "language": "Jazyk",
    },
    "en": {
        "app_name": "Stock Picker Pro",
        "language": "Language",
    },
}

def t(key: str, lang: str = "cs") -> str:
    """Tiny translation helper. Returns key if translation missing."""
    try:
        lang = (lang or "cs").lower()
    except Exception:
        lang = "cs"
    return TRANSLATIONS.get(lang, TRANSLATIONS["cs"]).get(key, str(key))

def main():
    # Session state initialization
    if "force_tab_label" not in st.session_state:
        st.session_state.force_tab_label = None
    if "language" not in st.session_state:
        st.session_state.language = "cz"  # Default language
    if "ai_report_data" not in st.session_state:
        st.session_state.ai_report_data = None
    if "ai_report_ticker" not in st.session_state:
        st.session_state.ai_report_ticker = None
    if "active_tab_index" not in st.session_state:
        st.session_state.active_tab_index = 0
    if "ai_report_cache" not in st.session_state:
        st.session_state.ai_report_cache = {}
    
    """Main application entry point."""

    # --- UI mode state (picker vs results) ---
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "PICKER"
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = ""

    if "close_sidebar_js" not in st.session_state:
        st.session_state.close_sidebar_js = False

    # Optional: hide sidebar overlay on mobile after analyze (keeps results visible)
    if st.session_state.get("marketCap"):
        st.markdown("""
        <style>
        @media (max-width: 900px) {
          section[data-testid="stSidebar"], [data-testid="stSidebar"] {
            transform: translateX(-120%) !important;
            opacity: 0 !important;
            pointer-events: none !important;
          }
        }
        </style>
        """, unsafe_allow_html=True)

    if "sidebar_hidden" not in st.session_state:
        st.session_state.sidebar_hidden = False


    # If requested (e.g., after clicking Analyze), inject JS in MAIN area to force-close the sidebar on mobile.
    if st.session_state.get("marketCap"):
        components.html(js_close_sidebar(), height=0, width=0)
        st.session_state.close_sidebar_js = False



    # Page configuration is set at module import (must be first Streamlit command)
    
    # Custom CSS
    st.markdown("""
    <style>
        /* Mobile-friendly spacing */
        .stButton > button {
            width: 100%;
            margin: 5px 0;
            min-height: 44px;
        }
        
        /* Responsive metrics */
        [data-testid="stMetricValue"] {
            font-size: clamp(1.2rem, 4vw, 2rem);
        }
        
        /* Smart header cards */
        .metric-card {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.03);
            margin-bottom: 10px;
        }
        
        .metric-label {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: clamp(1.5rem, 5vw, 2.5rem);
            font-weight: 700;
        }
        
        .metric-delta {
            font-size: 0.9rem;
            margin-top: 3px;
        }
        
        /* Responsive tables */
        .dataframe {
            font-size: clamp(0.75rem, 2vw, 0.95rem);
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(0,0,0,0.03) 0%, rgba(0,0,0,0.01) 100%);
        }
        
        /* Warning boxes */
        .warning-box {
            padding: 15px;
            border-left: 4px solid #ff8800;
            background: rgba(255, 136, 0, 0.1);
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Success boxes */
        .success-box {
            padding: 15px;
            border-left: 4px solid #00ff88;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 5px;
            margin: 10px 0;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            margin: 20px 0 10px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        }
    
@media (max-width: 768px){
  section[data-testid="stSidebar"]{
    background: rgba(15,23,42,0.995)!important;
    backdrop-filter: none!important;
    -webkit-backdrop-filter: none!important;
  }
  /* Ensure sidebar content readable on mobile */
  section[data-testid="stSidebar"] *{
    color: #e5e7eb;
  }
}
</style>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - Settings & Controls
    # ========================================================================

    with st.sidebar:
        st.title(f"📈 {t('app_name', st.session_state.language)}")
        st.caption(f"{APP_VERSION} - Advanced Quant Analysis")
        st.markdown("---")
        
        # Language selector
        lang_options = {"🇨🇿 Čeština": "cz", "🇺🇸 English": "en"}
        selected_lang_label = st.selectbox(
            t("language", st.session_state.language),
            options=list(lang_options.keys()),
            index=0 if st.session_state.language == "cz" else 1,
            key="lang_select"
        )
        st.session_state.language = lang_options[selected_lang_label]
        lang = st.session_state.language
        
        st.markdown("---")
        
        # Ticker input (Form -> Enter submits)

        
        with st.form("analyze_form", clear_on_submit=False):

        
            default_ticker = st.session_state.get("marketCap") or st.session_state.get("marketCap") or "AAPL"

        
            _raw_ticker = st.text_input(

        
                "Ticker Symbol",

        
                value=str(default_ticker),

        
                help="Zadej ticker (např. AAPL, MSFT, GOOGL) a potvrď Enterem",

        
                max_chars=10,

        
                key="ticker_input",

        
            )

        
            ticker_input = (_raw_ticker or "").upper().strip()

        
            analyze_btn = st.form_submit_button("🔍 Analyzovat", type="primary", use_container_width=True)

        
        

        
        if analyze_btn:

        
            # Request sidebar close (mobile drawer) and rerun into RESULTS mode.

        
            st.session_state.close_sidebar_js = True

        
            st.session_state.sidebar_hidden = True
            st.session_state.ui_mode = "RESULTS"

        
            st.session_state.selected_ticker = ticker_input

        
            st.session_state["last_ticker"] = ticker_input

        
            st.rerun()
        st.markdown("---")
        
        # DCF Settings
        with st.expander("⚙️ DCF Parametry", expanded=False):
            smart_dcf = st.checkbox("⚡ Smart DCF (Automaticky)", value=True, key="smart_dcf")
            dcf_growth = st.slider(
                "Růst FCF (roční)",
                0.0, 0.50, 0.10, 0.01,
                help="Očekávaný roční růst Free Cash Flow",
                disabled=smart_dcf
            )
            dcf_terminal = st.slider(
                "Terminální růst",
                0.0, 0.10, 0.03, 0.01,
                help="Dlouhodobý růst po projektovaném období"
            )
            dcf_wacc = st.slider(
                "WACC (diskont)",
                0.05, 0.20, 0.10, 0.01,
                help="Vážené průměrné náklady kapitálu",
                disabled=smart_dcf
            )
            dcf_years = st.slider(
                "Projektované roky",
                3, 10, 5, 1,
                help="Počet let pro projekci FCF"
            )
            dcf_exit_multiple = st.slider(
                "Exit Multiple (FCF)",
                10.0, 50.0, 25.0, 1.0,
                help="Násobek FCF v posledním projektovaném roce pro terminal value (Exit Multiple metoda)",
                disabled=smart_dcf
            )
        
        st.markdown("---")
        
        # AI Settings
        with st.expander("🤖 AI Nastavení", expanded=False):
            use_ai = st.checkbox(
                "Povolit AI analýzu",
                value=bool(GEMINI_API_KEY),
                help="Vyžaduje Gemini API klíč",
                disabled=not GEMINI_API_KEY
            )
            if not GEMINI_API_KEY:
                st.warning("⚠️ Nastav GEMINI_API_KEY v kódu")
        
        st.markdown("---")
        
        # Quick links
        st.markdown("### 🔗 Odkazy")
        if ticker_input:
            st.markdown(f"- [Yahoo Finance](https://finance.yahoo.com/quote/{ticker_input})")
            st.markdown(f"- [SEC Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=&type=&dateb=&owner=exclude&count=40&search_text={ticker_input})")
            st.markdown(f"- [Finviz](https://finviz.com/quote.ashx?t={ticker_input})")
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Welcome screen if no analysis yet
    if st.session_state.get("marketCap") == "PICKER" and (not analyze_btn) and ("last_ticker" not in st.session_state):
        display_welcome_screen()
        st.stop()
    
        # Pokud jsme ve výsledcích, nabídni rychlý návrat na výběr (hlavně pro mobil)
    if st.session_state.get("marketCap") == "RESULTS":
        colA, colB = st.columns([1, 2])
        with colA:
            if st.button("☰ Menu", use_container_width=True):
                st.session_state.sidebar_hidden = False
                st.rerun()
        with colB:
            st.empty()
        if st.button("⬅️ Zpět na výběr", use_container_width=True):
            st.session_state.ui_mode = "PICKER"
            st.session_state.sidebar_hidden = False
            st.session_state.selected_ticker = ""
            st.session_state.pop("last_ticker", None)
            st.rerun()

    # Process ticker
    ticker = (st.session_state.get("marketCap") or ticker_input) if analyze_btn else st.session_state.get("last_ticker", "AAPL")
    st.session_state["last_ticker"] = ticker
    
    # Fetch data
    with st.spinner(f"📊 Načítám data pro {ticker}..."):
        info = fetch_ticker_info(ticker)
        
        if not info:
            st.error(f"❌ Nepodařilo se načíst data pro {ticker}. Zkontroluj ticker.")
            st.stop()
        
        company = info.get("longName") or info.get("shortName") or ticker
        metrics = extract_metrics(info, ticker)
        price_history = fetch_price_history(ticker, period="1y")
        income, balance, cashflow = fetch_financials(ticker)
        
        # Advanced data
        ath = get_all_time_high(ticker)
        insider_df = fetch_insider_transactions_fmp(ticker)
        insider_signal = compute_insider_pro_signal(insider_df)
        
        # DCF calculations
        market_cap_for_fcf = safe_float(info.get('marketCap'))
        fcf, fcf_dbg = get_fcf_ttm_yfinance(ticker, market_cap_for_fcf)
        for _m in (fcf_dbg or []):
            print(_m)
        shares = safe_float(info.get("sharesOutstanding"))
        current_price = metrics.get("price").value if metrics.get("price") else None

        # Decide DCF inputs (Smart vs Manual)
        used_dcf_growth = float(dcf_growth)
        used_dcf_wacc = float(dcf_wacc)
        used_exit_multiple = float(dcf_exit_multiple)
        used_mode_label = "Manual"

        if st.session_state.get("smart_dcf", True):
            smart = estimate_smart_params(info, metrics)
            used_dcf_growth = float(smart["growth"])
            used_dcf_wacc = float(smart["wacc"])
            used_exit_multiple = float(smart["exit_multiple"])
            used_mode_label = "Smart"

        
        # --- Amazon-style reinvestment heavy adjustment (Adjusted FCF) ---
        # If FCF is unusually low relative to Operating Cash Flow, treat it as heavy reinvestment and
        # use an adjusted cash-flow proxy for DCF (maintenance earnings proxy).
        dcf_fcf_used = fcf
        try:
            operating_cashflow = safe_float(info.get("operatingCashflow"))
        except Exception:
            operating_cashflow = None

        if operating_cashflow and dcf_fcf_used and dcf_fcf_used > 0 and operating_cashflow > 0:
            if dcf_fcf_used < (0.3 * operating_cashflow):
                dcf_fcf_used = operating_cashflow * 0.6
                st.warning("⚠️ Detekováno vysoké reinvestování (Amazon style). Použito upravené OCF místo FCF.")
                print(f"Adjusted FCF used (reinvestment-heavy): {dcf_fcf_used/1e9:.2f}B (OCF {operating_cashflow/1e9:.2f}B)")
        fair_value_dcf = None
        mos_dcf = None
        implied_growth = None
        
        if dcf_fcf_used and shares and dcf_fcf_used > 0:
            # --- NOVÝ VÝPOČET DCF (Exit Multiple Metoda) ---
            # 1. Spočítáme budoucí FCF pro každý rok
            future_fcf = []
            current_fcf = dcf_fcf_used
            
            # Diskontní faktor
            discount_factors = [(1 + used_dcf_wacc) ** i for i in range(1, dcf_years + 1)]
            
            for i in range(dcf_years):
                current_fcf = current_fcf * (1 + used_dcf_growth)
                future_fcf.append(current_fcf)
            
            # 2. Terminal Value (Hodnota na konci 5. roku)
            # Použijeme Exit Multiple (pro Big Tech standardně 25x, ne konzervativní Gordon)
            exit_multiple = float(used_exit_multiple)
            terminal_value = future_fcf[-1] * exit_multiple
            
            # 3. Diskontování na dnešní hodnotu (PV)
            pv_cash_flows = sum([f / d for f, d in zip(future_fcf, discount_factors)])
            pv_terminal_value = terminal_value / ((1 + used_dcf_wacc) ** dcf_years)
            
            enterprise_value = pv_cash_flows + pv_terminal_value
            
            # 4. Equity Value (EV + Cash - Debt)
            total_cash = safe_float(info.get("totalCash")) or 0
            total_debt = safe_float(info.get("totalDebt")) or 0
            equity_value = enterprise_value + total_cash - total_debt
            
            fair_value_dcf = equity_value / shares
            
            # Přepočet MOS a Implied Growth
            if current_price:
                mos_dcf = (fair_value_dcf / current_price) - 1.0
                implied_growth = reverse_dcf_implied_growth(
                    current_price, fcf, dcf_terminal, dcf_wacc, dcf_years, shares
                )
        
        # Analyst fair value
        analyst_target = metrics.get("target_mean").value if metrics.get("target_mean") else None
        mos_analyst = None
        if analyst_target and current_price:
            mos_analyst = (analyst_target / current_price) - 1.0
        
        # Scorecard
        scorecard, category_scores, individual_scores = build_scorecard_advanced(metrics, info)
        
        # Verdict
        verdict, verdict_color, verdict_warnings = get_advanced_verdict(
            scorecard, mos_dcf, mos_analyst, insider_signal.get("signal", 0), implied_growth
        )
        
        # Peers
        sector = info.get("sector", "")
        auto_peers = get_auto_peers(ticker, sector, info)
    
    # ========================================================================
    # SMART HEADER (5 cards)
    # ========================================================================
    
    st.title(f"{company} ({ticker})")
    st.caption(f"📊 {sector} | Market Cap: {fmt_money(info.get('marketCap'), 0) if info.get('marketCap') else '—'}")
    
    # Header cards row
    h1, h2, h3, h4, h5 = st.columns(5)
    
    with h1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Aktuální cena</div>
            <div class="metric-value">{fmt_money(current_price)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h2:
        analyst_price = analyst_target if analyst_target else None
        analyst_delta = f"+{((analyst_price/current_price - 1)*100):.1f}%" if analyst_price and current_price else "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Férovka (Analytici)</div>
            <div class="metric-value">{fmt_money(analyst_price)}</div>
            <div class="metric-delta" style="color: #00ff88;">{analyst_delta if analyst_price else ""}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h3:
        dcf_mos_str = f"{mos_dcf*100:+.1f}% MOS" if mos_dcf is not None else "—"
        dcf_color = "#00ff88" if mos_dcf and mos_dcf > 0 else "#ff4444"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Férovka (DCF)</div>
            <div class="metric-value">{fmt_money(fair_value_dcf)}</div>
            <div class="metric-delta" style="color: {dcf_color};">{dcf_mos_str}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h4:
        if ath and current_price:
            pct_from_ath = ((current_price / ath) - 1) * 100
            ath_str = f"{pct_from_ath:+.1f}%"
        else:
            ath_str = "—"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">ATH</div>
            <div class="metric-value">{fmt_money(ath)}</div>
            <div class="metric-delta">{ath_str} od vrcholu</div>
        </div>
        """, unsafe_allow_html=True)
    
    with h5:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid {verdict_color};">
            <div class="metric-label">Sektor</div>
            <div class="metric-value" style="font-size: 1.2rem;">{sector[:20]}</div>
            <div class="metric-delta" style="color: {verdict_color}; font-weight: 700;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tabs = st.tabs([
        "📊 Overview",
        "🗓️ Market Watch",
        "🤖 AI Analyst",
        "🏢 Peer Comparison",
        "📋 Scorecard Pro",
        "💰 Valuace (DCF)",
        "📝 Memo & Watchlist",
        "🐦 Social & Guru"
    ])

    # Keep user on the tab they clicked (Streamlit rerun otherwise jumps to first tab)
    if "force_tab_label" in st.session_state and st.session_state.force_tab_label:
        components.html(js_open_tab(st.session_state.force_tab_label), height=0, width=0)
        st.session_state.force_tab_label = None

    
    # ------------------------------------------------------------------------
    # TAB 1: Overview
    # ------------------------------------------------------------------------
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Rychlý přehled</div>', unsafe_allow_html=True)
        
        # Two columns
        left, right = st.columns([1, 1])
        
        with left:
            st.markdown("#### 📌 Základní info")
            st.write(f"**Společnost:** {company}")
            st.write(f"**Ticker:** {ticker}")
            st.write(f"**Sektor:** {sector}")
            st.write(f"**Odvětví:** {info.get('industry', '—')}")
            st.write(f"**Země:** {info.get('country', '—')}")
            
            summary = info.get("longBusinessSummary", "")
            if summary:
                st.markdown("#### 📝 O společnosti")
                with st.expander("Zobrazit popis", expanded=False):
                    st.write(summary)
        
        with right:
            st.markdown("#### 💎 Klíčové metriky")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("P/E", fmt_num(metrics["pe"].value if metrics.get("marketCap") else None))
                st.metric("ROE", fmt_pct(metrics["roe"].value if metrics.get("marketCap") else None))
                st.metric("Op. Margin", fmt_pct(metrics["operating_margin"].value if metrics.get("marketCap") else None))
            
            with m2:
                st.metric("FCF Yield", fmt_pct(metrics["fcf_yield"].value if metrics.get("marketCap") else None))
                st.metric("Debt/Equity", fmt_num(metrics["debt_to_equity"].value if metrics.get("marketCap") else None))
                st.metric("Rev. Growth", fmt_pct(metrics["revenue_growth"].value if metrics.get("marketCap") else None))
        
        # Price chart
        st.markdown("---")
        st.markdown("#### 📈 Cenový vývoj (1 rok)")
        if not price_history.empty:
            chart_data = price_history[["Close"]].copy()
            chart_data.columns = ["Cena"]
            st.line_chart(chart_data, use_container_width=True, height=400)
        else:
            st.info("Graf není k dispozici")
        
        # Insider signal
        st.markdown("---")
        st.markdown("#### 🔐 Insider Trading Signal")
        
        ins1, ins2, ins3 = st.columns(3)
        with ins1:
            st.metric(
                "Signal",
                f"{insider_signal.get('signal', 0):.0f}/100",
                delta=insider_signal.get('label', 'N/A')
            )
        with ins2:
            st.metric("Nákupy (6M)", insider_signal.get('recent_buys', 0))
        with ins3:
            st.metric("Prodeje (6M)", insider_signal.get('recent_sells', 0))
        
        if insider_signal.get("marketCap"):
            st.markdown(
                '<div class="success-box">🔥 <b>Cluster Buying Detected!</b> Více insiderů nakupuje současně - silný bullish signál.</div>',
                unsafe_allow_html=True
            )
        
        for insight in insider_signal.get('insights', []):
            st.write(f"• {insight}")
    
    # ------------------------------------------------------------------------
    # TAB 2: Market Watch (Makro & Earnings Calendar)
    # ------------------------------------------------------------------------
    with tabs[1]:
        st.markdown('<div class="section-header">🗓️ Market Watch - Upcoming Events</div>', unsafe_allow_html=True)
        
        st.markdown("### 🌍 Makroekonomické události (příští 2 měsíce)")
        
        macro_df = pd.DataFrame(MACRO_CALENDAR)
        macro_df['date'] = pd.to_datetime(macro_df['date'])
        macro_df = macro_df[macro_df['date'] >= dt.datetime.now()]
        macro_df = macro_df.sort_values('date')
        
        if not macro_df.empty:
            # Color code by importance
            def color_importance(val):
                if val == "Critical":
                    return 'background-color: #ff4444; color: white; font-weight: bold;'
                elif val == "High":
                    return 'background-color: #ff8800; color: white;'
                else:
                    return 'background-color: #ffaa00;'
            
            styled_df = macro_df.style.applymap(color_importance, subset=['importance'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 📊 Earnings Calendar")
        
        # Estimate earnings date
        next_earnings = get_earnings_calendar_estimate(ticker, info)
        
        if next_earnings:
            st.success(f"📅 **{ticker} očekávané earnings:** {next_earnings.strftime('%d.%m.%Y')}")
        else:
            st.info(f"📅 Earnings datum pro {ticker} není k dispozici")
        
        # Show peer earnings too
        if auto_peers:
            st.markdown("#### Earnings konkurence")
            peer_earnings = []
            for peer in auto_peers[:3]:
                peer_info = fetch_ticker_info(peer)
                peer_date = get_earnings_calendar_estimate(peer, peer_info)
                if peer_date:
                    peer_earnings.append({
                        "Ticker": peer,
                        "Earnings Date": peer_date.strftime('%d.%m.%Y')
                    })
            
            if peer_earnings:
                st.dataframe(pd.DataFrame(peer_earnings), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.info("💡 **Tip:** Sleduj tyto události pro včasné rozhodnutí o entry/exit pointech!")
    
    # ------------------------------------------------------------------------
    # TAB 3: AI Analyst Report
    # ------------------------------------------------------------------------
# ------------------------------------------------------------------------
    # TAB 3: AI Analyst Report (ASIMETRICKÁ VERZE 4.0)
    # ------------------------------------------------------------------------
   # ------------------------------------------------------------------------
    # TAB 3: AI Analyst Report (ASIMETRICKÁ VERZE 4.0)
    # ------------------------------------------------------------------------
    with tabs[2]:
        st.markdown('<div class="section-header">🤖 AI Analytik & Asymetrie</div>', unsafe_allow_html=True)
        
        # --- EDUKATIVNÍ LEGENDA ---
        with st.expander("ℹ️ Co znamenají tyto metriky?", expanded=False):
            st.markdown("""
            ### ⚖️ Asymmetry Score
            Měří tzv. **konvexitu** investice. Cílem je najít situace, kde je distribuce pravděpodobnosti "nakloněna" ve váš prospěch.
            * **Vysoké skóre (70+):** Downside je omezen (např. vysokou hotovostí, aktivy), zatímco upside je otevřený.
            * **Nízké skóre (0-30):** Riskujete 50 %, abyste vydělali 10 %. To je asymetrie, které se chceme vyhnout.

            ### 🥊 Red Team Attack
            Technika eliminace **konfirmačního zkreslení** (tendence hledat jen důkazy pro svůj názor). 
            AI v tomto modulu simuluje roli *Short Sellera* nebo agresivního oponenta. Pokud vaše investiční teze přežije 
            tento "útok" a rizika jsou akceptovatelná, je vaše rozhodnutí mnohem robustnější.
            """)
            
        if not GEMINI_API_KEY:
            st.warning("⚠️ **AI analýza není dostupná**")
            st.info("Nastav GEMINI_API_KEY v secrets pro aktivaci AI analytika.")
        else:
            # OPRAVENÉ TLAČÍTKO: Teď už skutečně volá funkci
            if st.button("🚀 Vygenerovat Asymetrický Report", use_container_width=True, type="primary"):
                st.session_state.force_tab_label = "🤖 AI Analyst"
                st.session_state.ai_report_ticker = None
                
                with st.spinner("🧠 Seniorní manažer analyzuje asymetrii trhu..."):
                    # Volání tvé retry funkce
                    ai_report = generate_ai_analyst_report_with_retry(
                        ticker=ticker,
                        company=company,
                        metrics=metrics,
                        info=info,
                        dcf_fair_value=fair_value_dcf,
                        current_price=current_price,
                        scorecard=scorecard,
                        macro_events=MACRO_CALENDAR,
                        insider_signal=insider_signal
                    )
                    
                    # Uložení výsledku do session_state
                    st.session_state['ai_report'] = ai_report
                    st.session_state.ai_report_ticker = ticker
                    st.rerun() # Refresh pro zobrazení výsledků

            # --- ZOBRAZENÍ VÝSLEDKŮ ---
            if 'ai_report' in st.session_state and st.session_state.ai_report_ticker == ticker:
                report = st.session_state['ai_report']
                
                # 1. Gauge Chart (Ukazatel asymetrie)
                import plotly.graph_objects as go
                score = report.get("asymmetry_score", 50)
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score,
                    title = {'text': "Asymmetry Score", 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#00ff88" if score > 70 else "#ffaa00"},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(255, 68, 68, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(255, 170, 0, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(0, 255, 136, 0.2)"}
                        ],
                        'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': score}
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

                # 2. RED TEAM WARNING BOX
                st.markdown(f"""
                    <div style="background-color: rgba(255, 68, 68, 0.1); border: 2px solid #ff4444; padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                        <h3 style="color: #ff4444; margin-top: 0; font-size: 1.2rem;">🚨 RED TEAM ATTACK</h3>
                        <p style="font-style: italic; color: #ffcccc; margin-bottom: 0;">{report.get('red_team_warning', 'N/A')}</p>
                    </div>
                """, unsafe_allow_html=True)

                # 3. Bull & Bear Case Sloupce
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🐂 Bull Case (Upside)")
                    for item in report.get('bull_case', []):
                        st.write(f"✅ {item}")
                
                with col2:
                    st.markdown("### 🐻 Bear Case (Downside)")
                    for item in report.get('bear_case', []):
                        st.write(f"⚠️ {item}")

                # 4. Syntéza a detaily
                st.markdown("---")
                st.markdown(f"**🛡️ Fundamentální podlaha:** {report.get('fundamental_floor', 'N/A')}")
                st.info(f"**🎯 Strategická syntéza:** {report.get('reasoning_synthesis', 'N/A')}")
                
                # Spodní řada metrik
                v_col1, v_col2, v_col3 = st.columns(3)
                with v_col1:
                    verdict = report.get('verdict', 'N/A')
                    st.metric("Finální verdikt", verdict)
                with v_col2:
                    st.metric("Risk/Reward Ratio", report.get('risk_reward_ratio', 'N/A'))
                with v_col3:
                    st.metric("Confidence", report.get('confidence', 'N/A'))
    # ------------------------------------------------------------------------
    # TAB 4: Peer Comparison
    # ------------------------------------------------------------------------
    with tabs[3]:
        st.markdown('<div class="section-header">🏢 Srovnání s konkurencí</div>', unsafe_allow_html=True)
        
        if not auto_peers:
            st.info(f"📊 **{ticker}** - Aktuálně bez přímé srovnatelné konkurence v databázi.")
            st.caption("Přidej manuálně do SECTOR_PEERS slovníku v kódu pro zobrazení peer analýzy.")
        else:
            st.success(f"🔍 Nalezeno {len(auto_peers)} konkurentů: {', '.join(auto_peers)}")
            
            with st.spinner("Načítám data konkurence..."):
                peer_df = fetch_peer_comparison(ticker, auto_peers)
            
            if not peer_df.empty:
                # Format for display
                display_df = peer_df.copy()
                display_df['P/E'] = display_df['P/E'].apply(lambda x: fmt_num(x))
                display_df['Op. Margin'] = display_df['Op. Margin'].apply(lambda x: fmt_pct(x))
                display_df['Rev. Growth'] = display_df['Rev. Growth'].apply(lambda x: fmt_pct(x))
                display_df['FCF Yield'] = display_df['FCF Yield'].apply(lambda x: fmt_pct(x))
                display_df['Market Cap'] = display_df['Market Cap'].apply(lambda x: fmt_money(x, 0, "$") if x else "—")
                
                # Highlight main ticker
                def highlight_ticker(row):
                    if row['Ticker'] == ticker:
                        return ['background-color: #00ff8820'] * len(row)
                    return [''] * len(row)
                
                styled = display_df.style.apply(highlight_ticker, axis=1)
                st.dataframe(styled, use_container_width=True, hide_index=True)
                
                # Insights
                st.markdown("#### 📊 Relativní pozice")
                
                # Calculate percentiles
                if len(peer_df) > 1:
                    main_row = peer_df[peer_df['Ticker'] == ticker].iloc[0] if ticker in peer_df['Ticker'].values else None
                    
                    if main_row is not None:
                        insights = []
                        
                        # P/E comparison
                        pe_val = main_row['P/E']
                        if pd.notna(pe_val):
                            pe_rank = (peer_df['P/E'] < pe_val).sum() + 1
                            total = peer_df['P/E'].notna().sum()
                            if pe_rank <= total * 0.33:
                                insights.append(f"✅ P/E je v dolní třetině (levnější valuace než většina konkurence)")
                            elif pe_rank >= total * 0.67:
                                insights.append(f"⚠️ P/E je v horní třetině (dražší valuace)")
                        
                        # Revenue growth
                        rg_val = main_row['Rev. Growth']
                        if pd.notna(rg_val):
                            rg_rank = (peer_df['Rev. Growth'] > rg_val).sum() + 1
                            total = peer_df['Rev. Growth'].notna().sum()
                            if rg_rank <= total * 0.33:
                                insights.append(f"🚀 Revenue growth v TOP třetině (roste rychleji než konkurence)")
                        
                        for insight in insights:
                            st.write(f"• {insight}")
            else:
                st.warning("Nepodařilo se načíst data konkurence")
    
    # ------------------------------------------------------------------------
    # TAB 5: Scorecard Pro
    # ------------------------------------------------------------------------
    with tabs[4]:
        st.markdown('<div class="section-header">📋 Investiční Scorecard Pro</div>', unsafe_allow_html=True)
        
        # Overall score
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; border: 3px solid {verdict_color}; border-radius: 15px; background: rgba(255,255,255,0.03);">
            <div style="font-size: 1rem; opacity: 0.8;">CELKOVÉ SKÓRE</div>
            <div style="font-size: 4rem; font-weight: 900; color: {verdict_color};">{scorecard:.0f}<span style="font-size: 2rem; opacity: 0.6;">/100</span></div>
            <div style="font-size: 1.5rem; font-weight: 700; margin-top: 10px;">{verdict}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Category breakdown
        st.markdown("### 📊 Rozpad podle kategorií")
        
        cat_cols = st.columns(len(category_scores))
        for idx, (cat_name, cat_score) in enumerate(category_scores.items()):
            with cat_cols[idx]:
                cat_color = "#00ff88" if cat_score >= 70 else ("#ffaa00" if cat_score >= 50 else "#ff4444")
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border: 2px solid {cat_color}; border-radius: 10px;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">{cat_name}</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {cat_color};">{cat_score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Warnings from verdict
        if verdict_warnings:
            st.markdown("### ⚠️ Důležitá upozornění")
            for warning in verdict_warnings:
                st.markdown(f'<div class="warning-box">{warning}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Individual metrics
        st.markdown("### 🔍 Detailní metriky")
        
        if individual_scores:
            metric_df = pd.DataFrame([
                {
                    "Metrika": name,
                    "Hodnota": fmt_num(metrics.get(key).value) if key in ["pe", "pb", "ps", "peg", "current_ratio", "quick_ratio", "debt_to_equity"] 
                               else fmt_pct(metrics.get(key).value) if key in ["roe", "roa", "operating_margin", "profit_margin", "gross_margin", "revenue_growth", "earnings_growth", "fcf_yield"]
                               else fmt_num(metrics.get(key).value),
                    "Skóre": f"{score:.1f}/10"
                }
                for key, metric in metrics.items()
                for name, score in individual_scores.items()
                if metric.name == name
            ])
            
            st.dataframe(metric_df, use_container_width=True, hide_index=True)
    
    # ------------------------------------------------------------------------
    # TAB 6: DCF Valuation
    # ------------------------------------------------------------------------
    with tabs[5]:
        st.markdown('<div class="section-header">💰 DCF Valuace & Reverse DCF</div>', unsafe_allow_html=True)
        
        st.info(f"Použitý Růst: {used_dcf_growth*100:.1f} % ({used_mode_label}) | Použitý WACC: {used_dcf_wacc*100:.1f} % ({used_mode_label}) | Exit Multiple: {used_exit_multiple:.1f}× ({used_mode_label})")
        
        if fcf and shares and fcf > 0:
            # Main DCF results
            dcf_col1, dcf_col2, dcf_col3, dcf_col4 = st.columns(4)
            
            with dcf_col1:
                st.metric("Férová hodnota (DCF)", fmt_money(fair_value_dcf))
            with dcf_col2:
                st.metric("Aktuální cena", fmt_money(current_price))
            with dcf_col3:
                mos_str = f"{mos_dcf*100:+.1f}%" if mos_dcf is not None else "—"
                mos_color_delta = mos_str if mos_dcf else None
                st.metric("Margin of Safety", mos_str, delta=mos_color_delta)
            with dcf_col4:
                if implied_growth is not None:
                    st.metric("Implied Growth (Reverse DCF)", f"{implied_growth*100:.1f}%")
                else:
                    st.metric("Implied Growth", "—")
            
            st.markdown("---")
            
            # Sensitivity analysis
            st.markdown("### 📊 Sensitivity Analysis")
            
            sens_col1, sens_col2 = st.columns(2)
            
            with sens_col1:
                st.markdown("**🔼 Růst FCF Impact**")
                growth_rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
                sens_data = []
                for g in growth_rates:
                    fv = calculate_dcf_fair_value(fcf, g, dcf_terminal, dcf_wacc, dcf_years, shares)
                    upside = ((fv / current_price) - 1) * 100 if fv and current_price else None
                    sens_data.append({
                        "Růst": f"{g*100:.0f}%",
                        "Fair Value": fmt_money(fv),
                        "Upside": f"{upside:+.1f}%" if upside else "—"
                    })
                st.dataframe(pd.DataFrame(sens_data), use_container_width=True, hide_index=True)
            
            with sens_col2:
                st.markdown("**💹 WACC Impact**")
                wacc_rates = [0.08, 0.09, 0.10, 0.11, 0.12, 0.15]
                wacc_data = []
                for w in wacc_rates:
                    fv = calculate_dcf_fair_value(fcf, dcf_growth, dcf_terminal, w, dcf_years, shares)
                    upside = ((fv / current_price) - 1) * 100 if fv and current_price else None
                    wacc_data.append({
                        "WACC": f"{w*100:.0f}%",
                        "Fair Value": fmt_money(fv),
                        "Upside": f"{upside:+.1f}%" if upside else "—"
                    })
                st.dataframe(pd.DataFrame(wacc_data), use_container_width=True, hide_index=True)
            
            # Interpretation
            st.markdown("---")
            st.markdown("### 🧠 Interpretace")
            
            if implied_growth is not None:
                if implied_growth < 0:
                    st.warning(f"📉 **Trh implikuje pokles FCF ({implied_growth*100:.1f}%)** - možná příležitost nebo reálné problémy")
                elif implied_growth < 0.05:
                    st.info(f"📊 Trh očekává nízký růst ({implied_growth*100:.1f}%) - konzervativní valuace")
                elif implied_growth < 0.15:
                    st.success(f"✅ Trh očekává zdravý růst ({implied_growth*100:.1f}%) - v souladu s tvým modelem")
                else:
                    st.warning(f"🚀 Trh očekává agresivní růst ({implied_growth*100:.1f}%) - vysoká očekávání, riziko zklamání")
        
        else:
            st.warning("⚠️ Nedostatek dat pro DCF (chybí FCF nebo počet akcií)")
    
    # ------------------------------------------------------------------------
    # TAB 7: Memo & Watchlist
    # ------------------------------------------------------------------------
    with tabs[6]:
        st.markdown('<div class="section-header">📝 Investment Memo & Watchlist</div>', unsafe_allow_html=True)
        
        # Load existing
        memos = get_memos()
        watch = get_watchlist()
        
        memo = memos.get("memos", {}).get(ticker, {})
        wl = watch.get("items", {}).get(ticker, {})
        
        # Auto-generate snippets
        auto_thesis = (
            f"{company} ({ticker}) - Investment Thesis\n\n"
            f"• Sektor: {sector}\n"
            f"• Cena: {fmt_money(current_price)} | Verdikt: {verdict}\n"
            f"• DCF Fair Value: {fmt_money(fair_value_dcf)} (MOS: {fmt_pct(mos_dcf)})\n"
            f"• Scorecard: {scorecard:.0f}/100\n"
            f"• Insider Signal: {insider_signal.get('label', '—')} ({float(insider_signal.get('signal', 0)):.0f}/100)"
        )
        
        # Memo form
        st.markdown("### 📄 Investment Memo")
        
        thesis = st.text_area(
            "Investiční teze",
            value=memo.get("marketCap") or auto_thesis,
            height=120
        )
        
        drivers = st.text_area(
            "Klíčové faktory úspěchu",
            value=memo.get("marketCap") or "- Růst tržeb\n- Zlepšení marží\n- Inovace",
            height=100
        )
        
        risks = st.text_area(
            "Rizika",
            value=memo.get("marketCap") or "- Konkurence\n- Regulace\n- Makro",
            height=100
        )
        
        catalysts = st.text_area(
            "Katalyzátory",
            value=memo.get("marketCap") or "",
            height=80
        )
        
        buy_conditions = st.text_area(
            "Buy podmínky",
            value=memo.get("marketCap") or f"- Entry < {fmt_money(fair_value_dcf * 0.95) if fair_value_dcf else '—'}",
            height=80
        )
        
        notes = st.text_area(
            "Poznámky",
            value=memo.get("marketCap") or "",
            height=80
        )
        
        # Save/Export buttons
        memo_col1, memo_col2 = st.columns(2)
        
        with memo_col1:
            if st.button("💾 Uložit Memo", use_container_width=True):
                memos.setdefault("memos", {})[ticker] = {
                    "thesis": thesis,
                    "drivers": drivers,
                    "risks": risks,
                    "catalysts": catalysts,
                    "buy_conditions": buy_conditions,
                    "notes": notes,
                    "updated_at": dt.datetime.now().isoformat(),
                }
                set_memos(memos)
                st.success("✅ Memo uloženo!")
        
        with memo_col2:
            if _HAS_PDF and st.button("📄 Export PDF", use_container_width=True):
                summary = {
                    "Price": fmt_money(current_price),
                    "DCF Fair": fmt_money(fair_value_dcf),
                    "Score": f"{scorecard:.0f}/100",
                    "Verdict": verdict
                }
                pdf_bytes = export_memo_pdf(ticker, company, {
                    "thesis": thesis,
                    "drivers": drivers,
                    "risks": risks,
                    "catalysts": catalysts,
                    "buy_conditions": buy_conditions,
                    "notes": notes
                }, summary)
                
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Stáhnout PDF",
                        data=pdf_bytes,
                        file_name=f"memo_{ticker}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        
        # Watchlist
        st.markdown("---")
        st.markdown("### ⭐ Watchlist")
        
        target_buy = st.number_input(
            "Cílová nákupní cena",
            value=float(wl.get("target_buy", 0.0)) if wl else 0.0,
            step=1.0
        )
        
        wl_col1, wl_col2 = st.columns(2)
        
        with wl_col1:
            if st.button("⭐ Přidat/Aktualizovat", use_container_width=True):
                watch.setdefault("items", {})[ticker] = {
                    "target_buy": target_buy,
                    "added_at": wl.get("marketCap") or dt.datetime.now().isoformat(),
                    "updated_at": dt.datetime.now().isoformat(),
                }
                set_watchlist(watch)
                st.success("✅ Watchlist aktualizován!")
        
        with wl_col2:
            if st.button("🗑️ Odebrat", use_container_width=True):
                if ticker in watch.get("items", {}):
                    watch["items"].pop(ticker, None)
                    set_watchlist(watch)
                    st.success("✅ Odebráno!")
        
        # Show watchlist
        st.markdown("#### 📋 Moje Watchlist")
        items = watch.get("items", {})
        
        if items:
            rows = []
            for tkr, item in items.items():
                inf = fetch_ticker_info(tkr)
                price_now = safe_float(inf.get("marketCap") or inf.get("marketCap"))
                tgt = safe_float(item.get("marketCap"))
                hit = (price_now is not None and tgt is not None and tgt > 0 and price_now <= tgt)
                
                rows.append({
                    "Ticker": tkr,
                    "Current": fmt_money(price_now),
                    "Target": fmt_money(tgt),
                    "Status": "🟢 BUY!" if hit else "⏳ Wait",
                    "Updated": item.get("updated_at", "")[:10]
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Watchlist je prázdný")
    

    # ------------------------------------------------------------------------
    # TAB 8: Social & Guru
    # ------------------------------------------------------------------------
    with tabs[7]:
        st.markdown('<div class="section-header">🐦 Social & Guru</div>', unsafe_allow_html=True)

        # Flatten options
        options = []
        option_map = {}
        for cat, people in GURUS.items():
            for name, handle in people.items():
                label = f"{cat} | {name}"
                options.append(label)
                option_map[label] = (cat, name, handle)

        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown("### 👤 Výběr Guru")
            sel = st.selectbox(
                "Vyber guru účet",
                options=options,
                index=0 if options else None,
                key="guru_selectbox"
            )
            cat, name, handle = option_map.get(sel, ("", "", ""))
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">Kategorie</div>'
                f'<div class="metric-value" style="font-size:1.1rem;">{cat or "—"}</div>'
                f'<div class="metric-delta" style="opacity:0.8;">@{handle}</div></div>',
                unsafe_allow_html=True
            )
            st.caption("Tip: Text tweetu pro AI analýzu vlož ručně níže (bez Twitter API).")

        with right:
            st.markdown(f"### 🐦 Timeline: {name or '—'}")
            guru_handle = handle
            st.markdown("### 📡 Přímý přenos")
            st.warning("⚠️ X (Twitter) blokuje náhledy v cizích aplikacích. Použij přímý odkaz níže.")
            st.markdown(f"""
            <div style="
                padding: 20px; 
                border-radius: 12px; 
                border: 1px solid rgba(255,255,255,0.1); 
                background: linear-gradient(135deg, rgba(29,161,242,0.1) 0%, rgba(0,0,0,0) 100%);
                text-align: center;
            ">
                <div style="font-size: 50px; margin-bottom: 10px;">🐦</div>
                <h3>@{guru_handle}</h3>
                <p>Klikni pro zobrazení nejnovějších analýz a komentářů přímo na X.</p>
                <a href="https://twitter.com/{guru_handle}" target="_blank" style="text-decoration: none;">
                    <button style="background-color: #1DA1F2; color: white; border: none; padding: 10px 20px; border-radius: 20px; font-weight: bold; cursor: pointer;">
                        Otevřít profil @{guru_handle} ↗
                    </button>
                </a>
                <br><br>
                <div style="text-align: left; font-size: 0.8em; opacity: 0.7;">
                    <strong>Tip:</strong> Otevři profil, najdi zajímavý tweet, zkopíruj text a vlož ho vlevo do AI analýzy.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f"#### 🔎 Hledat **${ticker}** na X")
            st.markdown(f"""
                <a href="https://twitter.com/search?q=%24{ticker}&src=typed_query&f=top" target="_blank">
                    <button style="background: transparent; border: 1px solid #1DA1F2; color: #1DA1F2; padding: 5px 15px; border-radius: 15px; cursor: pointer;">
                        Nejlepší tweety o ${ticker} ↗
                    </button>
                </a>
            """, unsafe_allow_html=True)

            social_text = st.text_area(
                "Vlož text tweetu nebo komentáře k analýze",
                height=140,
                key="social_text_area"
            )

            analyze_col1, analyze_col2 = st.columns([1, 3])
            with analyze_col1:
                do_analyze = st.button("Analyzovat Sentiment", use_container_width=True, key="btn_analyze_social")
            with analyze_col2:
                st.caption("Použije Gemini (pokud je nastaven GEMINI_API_KEY).")

            if do_analyze:
                if not social_text.strip():
                    st.warning("Vlož prosím text tweetu/komentáře k analýze.")
                else:
                    with st.spinner("Analyzuji…"):
                        result = analyze_social_text_with_gemini(social_text)

                    st.markdown(
                        '<div class="metric-card"><div class="metric-label">Výstup AI</div></div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(result)


    # Footer
    st.markdown("---")
    st.caption(f"📊 Data: Yahoo Finance | {APP_NAME} {APP_VERSION} | Toto není investiční doporučení")


def display_welcome_screen():
    """Display welcome screen when no ticker is selected."""
    st.title("Vítej v Stock Picker Pro v2.0! 🚀")
    
    st.markdown("""
    ### Pokročilá kvantitativní analýza akcií
    
    **🆕 Co je nového ve v2.0:**
    - ✅ **Smart Header** - 5 klíčových karet s responzivním layoutem
    - ✅ **Market Watch** - Makro kalendář (Fed, CPI, NFP) + earnings termíny
    - ✅ **AI Analyst** - Hloubkový Gemini report s bull/bear scénáři a konkrétní "wait for" cenou
    - ✅ **Auto-Peer Comparison** - Automatické srovnání s 3-5 konkurenty
    - ✅ **Insider Trading Pro** - Vážení rolí (CEO/CFO), cluster buying detection
    - ✅ **Scorecard Pro (0-100)** - Rozpad: Valuace, Kvalita, Růst, Fin. zdraví
    - ✅ **Mismatch Warning** - Upozornění když analytici vs DCF nesouhlasí
    
    **Jak začít:**
    1. ⬅️ Zadej ticker symbol v levém panelu (např. AAPL, MSFT, TSLA)
    2. Klikni na "🔍 Analyzovat"
    3. Prohlédni si všechny taby s pokročilými analýzami
    
    """)
    
    # Sample tickers
    st.markdown("### 💡 Populární tickery na vyzkoušení")
    cols = st.columns(4)
    samples = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    for i, ticker in enumerate(samples):
        with cols[i % 4]:
            if st.button(ticker, use_container_width=True, key=f"sample_{ticker}"):
                st.session_state["last_ticker"] = ticker
                st.rerun()
    
    st.markdown("---")
    st.info("💡 **Pro AI analýzu** nastav GEMINI_API_KEY v kódu a získej hloubkové AI reporty!")


if __name__ == "__main__":
    main()
