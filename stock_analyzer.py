"""
Stock Picker Pro v2.0
======================
Pokročilá kvantitativní analýza akcií s AI asistencí, makro kalendářem a peer analýzou.

Author: Enhanced by Claude
Version: 2.0
"""

import os
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'google\.generativeai\..*')
import streamlit as st
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
APP_VERSION = "v2.0"
GEMINI_API_KEY = st.secrets  # Add your Gemini API key here
GEMINI_MODEL = "gemini-2.5-flash-lite"
FMP_API_KEY = st.secrets  # Optional: Financial Modeling Prep API key

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
    },
    "Communication Services": {
        "T": ["VZ", "TMUS"],
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
# UTILITIES
# ============================================================================

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
    """Fetch basic info from Yahoo Finance."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
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
def fetch_insider_transactions(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch insider transactions."""
    try:
        t = yf.Ticker(ticker)
        return getattr(t, "insider_transactions", None)
    except Exception:
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
    fcf = safe_float(info.get("freeCashflow"))
    operating_cashflow = safe_float(info.get("operatingCashflow"))
    market_cap = safe_float(info.get("marketCap"))
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
            
            rows.append({
                "Ticker": t,
                "P/E": safe_float(info.get("trailingPE")),
                "Op. Margin": safe_float(info.get("operatingMargins")),
                "Rev. Growth": safe_float(info.get("revenueGrowth")),
                "FCF Yield": safe_div(safe_float(info.get("freeCashflow")), safe_float(info.get("marketCap"))),
                "Market Cap": safe_float(info.get("marketCap")),
            })
        except Exception:
            continue
    
    return pd.DataFrame(rows)


# ============================================================================
# AI ANALYST (GEMINI)
# ============================================================================

def generate_ai_analyst_report(
    ticker: str,
    company: str,
    metrics: Dict[str, Metric],
    info: Dict[str, Any],
    dcf_fair_value: Optional[float],
    current_price: Optional[float],
    scorecard: float,
    insider_signal: Dict[str, Any],
    macro_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate comprehensive AI analyst report using Gemini.
    
    Returns:
    {
        "market_situation": str,
        "bull_case": List[str],
        "bear_case": List[str],
        "verdict": str,
        "wait_for_price": float,
        "reasoning": str,
        "confidence": str
    }
    """
    
    if not GEMINI_API_KEY:
        return {
            "market_situation": "AI analýza není dostupná (chybí Gemini API klíč)",
            "bull_case": ["Nastavte GEMINI_API_KEY pro AI analýzu"],
            "bear_case": [],
            "verdict": "N/A",
            "wait_for_price": None,
            "reasoning": "Konfigurace AI chybí",
            "confidence": "N/A"
        }
    
    # Prepare context
    context = f"""
Analyzuj akci {company} ({ticker}) a poskytni hloubkový investiční report.

AKTUÁLNÍ DATA:
- Cena: {fmt_money(current_price)}
- Férovka (DCF): {fmt_money(dcf_fair_value)}
- Scorecard: {scorecard:.1f}/100
- P/E: {fmt_num(metrics.get('pe').value if metrics.get('pe') else None)}
- Revenue Growth: {fmt_pct(metrics.get('revenue_growth').value if metrics.get('revenue_growth') else None)}
- Operating Margin: {fmt_pct(metrics.get('operating_margin').value if metrics.get('operating_margin') else None)}
- FCF Yield: {fmt_pct(metrics.get('fcf_yield').value if metrics.get('fcf_yield') else None)}
- Insider Signal: {insider_signal.get('label', 'N/A')} ({insider_signal.get('signal', 0):.0f}/100)
- Sektor: {info.get('sector', 'N/A')}

MAKRO UDÁLOSTI (příští 2 měsíce):
{chr(10).join([f"- {e['date']}: {e['event']} ({e['importance']})" for e in macro_events[:5]])}

INSTRUKCE:
Vrať POUZE validní JSON s těmito klíči (žádný další text):
{{
  "market_situation": "1-2 věty o aktuální tržní situaci a co to znamená pro tuto akcii",
  "bull_case": ["důvod 1", "důvod 2", "důvod 3"],
  "bear_case": ["riziko 1", "riziko 2", "riziko 3"],
  "verdict": "BUY/HOLD/SELL",
  "wait_for_price": <číslo - konkrétní cena pro vstup, nebo null>,
  "reasoning": "2-3 věty proč tento verdikt a wait_for_price",
  "confidence": "HIGH/MEDIUM/LOW"
}}
"""
    
    try:
        # Try new google-genai SDK first
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=context
            )
            result_text = response.text
        except ImportError:
            # Fallback to old SDK
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(context)
            result_text = response.text
        
        # Parse JSON
        # Remove markdown code blocks if present
        result_text = re.sub(r'```json\s*', '', result_text)
        result_text = re.sub(r'```\s*', '', result_text)
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        return result
    
    except Exception as e:
        return {
            "market_situation": f"AI analýza selhala: {str(e)[:200]}",
            "bull_case": ["Chyba při generování"],
            "bear_case": [],
            "verdict": "ERROR",
            "wait_for_price": None,
            "reasoning": "Technická chyba",
            "confidence": "N/A"
        }


# ============================================================================
# CALENDAR & EVENTS
# ============================================================================

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
    if scorecard >= 75:
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

def main():
    """Main application entry point."""
    
    # Page configuration - WIDE LAYOUT
    st.set_page_config(
        page_title="Stock Picker Pro v2.0",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
    </style>
    """, unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR - Settings & Controls
    # ========================================================================
    
    with st.sidebar:
        st.title("📈 Stock Picker Pro")
        st.caption("v2.0 - Advanced Quant Analysis")
        st.markdown("---")
        
        # Ticker input
        ticker_input = st.text_input(
            "Ticker Symbol",
            value="AAPL",
            help="Zadej ticker (např. AAPL, MSFT, GOOGL)",
            max_chars=10
        ).upper().strip()
        
        analyze_btn = st.button("🔍 Analyzovat", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        # DCF Settings
        with st.expander("⚙️ DCF Parametry", expanded=False):
            dcf_growth = st.slider(
                "Růst FCF (roční)",
                0.0, 0.50, 0.10, 0.01,
                help="Očekávaný roční růst Free Cash Flow"
            )
            dcf_terminal = st.slider(
                "Terminální růst",
                0.0, 0.10, 0.03, 0.01,
                help="Dlouhodobý růst po projektovaném období"
            )
            dcf_wacc = st.slider(
                "WACC (diskont)",
                0.05, 0.20, 0.10, 0.01,
                help="Vážené průměrné náklady kapitálu"
            )
            dcf_years = st.slider(
                "Projektované roky",
                3, 10, 5, 1,
                help="Počet let pro projekci FCF"
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
    if not analyze_btn and "last_ticker" not in st.session_state:
        display_welcome_screen()
        st.stop()
    
    # Process ticker
    ticker = ticker_input if analyze_btn else st.session_state.get("last_ticker", "AAPL")
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
        insider_df = fetch_insider_transactions(ticker)
        insider_signal = compute_insider_pro_signal(insider_df)
        
        # DCF calculations
        fcf = safe_float(info.get("freeCashflow"))
        shares = safe_float(info.get("sharesOutstanding"))
        current_price = metrics.get("price").value if metrics.get("price") else None
        
        fair_value_dcf = None
        mos_dcf = None
        implied_growth = None
        
        if fcf and shares and fcf > 0:
            fair_value_dcf = calculate_dcf_fair_value(
                fcf, dcf_growth, dcf_terminal, dcf_wacc, dcf_years, shares
            )
            if fair_value_dcf and current_price:
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
        "📝 Memo & Watchlist"
    ])
    
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
                st.metric("P/E", fmt_num(metrics["pe"].value if metrics.get("pe") else None))
                st.metric("ROE", fmt_pct(metrics["roe"].value if metrics.get("roe") else None))
                st.metric("Op. Margin", fmt_pct(metrics["operating_margin"].value if metrics.get("operating_margin") else None))
            
            with m2:
                st.metric("FCF Yield", fmt_pct(metrics["fcf_yield"].value if metrics.get("fcf_yield") else None))
                st.metric("Debt/Equity", fmt_num(metrics["debt_to_equity"].value if metrics.get("debt_to_equity") else None))
                st.metric("Rev. Growth", fmt_pct(metrics["revenue_growth"].value if metrics.get("revenue_growth") else None))
        
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
        
        if insider_signal.get('cluster_detected'):
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
    with tabs[2]:
        st.markdown('<div class="section-header">🤖 AI Analytik - Hloubkový Report</div>', unsafe_allow_html=True)
        
        if not GEMINI_API_KEY:
            st.warning("⚠️ **AI analýza není dostupná**")
            st.info("Nastav GEMINI_API_KEY v kódu pro aktivaci AI analytika.")
        else:
            st.info("🤖 Gemini AI je připraven vygenerovat hloubkovou analýzu")
            
            if st.button("🚀 Vygenerovat AI Report", use_container_width=True, type="primary"):
                with st.spinner("🧠 AI analytik přemýšlí... (může trvat 10-20s)"):
                    ai_report = generate_ai_analyst_report(
                        ticker=ticker,
                        company=company,
                        metrics=metrics,
                        info=info,
                        dcf_fair_value=fair_value_dcf,
                        current_price=current_price,
                        scorecard=scorecard,
                        insider_signal=insider_signal,
                        macro_events=MACRO_CALENDAR
                    )
                    
                    st.session_state['ai_report'] = ai_report
            
            # Display report if available
            if 'ai_report' in st.session_state:
                report = st.session_state['ai_report']
                
                # Market situation
                st.markdown("### 🌐 Tržní situace")
                st.write(report.get('market_situation', 'N/A'))
                
                # Bull/Bear cases
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🐂 Bull Case")
                    for item in report.get('bull_case', []):
                        st.write(f"✅ {item}")
                
                with col2:
                    st.markdown("### 🐻 Bear Case")
                    for item in report.get('bear_case', []):
                        st.write(f"⚠️ {item}")
                
                # Verdict
                st.markdown("---")
                st.markdown("### 🎯 Verdikt AI")
                
                v_col1, v_col2, v_col3 = st.columns(3)
                
                with v_col1:
                    ai_verdict = report.get('verdict', 'N/A')
                    verdict_emoji = "🟢" if ai_verdict == "BUY" else ("🟡" if ai_verdict == "HOLD" else "🔴")
                    st.metric("Doporučení", f"{verdict_emoji} {ai_verdict}")
                
                with v_col2:
                    wait_price = report.get('wait_for_price')
                    st.metric("Wait for Price", fmt_money(wait_price) if wait_price else "N/A")
                
                with v_col3:
                    st.metric("Confidence", report.get('confidence', 'N/A'))
                
                st.markdown("#### 💭 Zdůvodnění")
                st.write(report.get('reasoning', 'N/A'))
    
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
            f"• Insider Signal: {insider_signal.get('label')} ({insider_signal.get('signal'):.0f}/100)"
        )
        
        # Memo form
        st.markdown("### 📄 Investment Memo")
        
        thesis = st.text_area(
            "Investiční teze",
            value=memo.get("thesis") or auto_thesis,
            height=120
        )
        
        drivers = st.text_area(
            "Klíčové faktory úspěchu",
            value=memo.get("drivers") or "- Růst tržeb\n- Zlepšení marží\n- Inovace",
            height=100
        )
        
        risks = st.text_area(
            "Rizika",
            value=memo.get("risks") or "- Konkurence\n- Regulace\n- Makro",
            height=100
        )
        
        catalysts = st.text_area(
            "Katalyzátory",
            value=memo.get("catalysts") or "",
            height=80
        )
        
        buy_conditions = st.text_area(
            "Buy podmínky",
            value=memo.get("buy_conditions") or f"- Entry < {fmt_money(fair_value_dcf * 0.95) if fair_value_dcf else '—'}",
            height=80
        )
        
        notes = st.text_area(
            "Poznámky",
            value=memo.get("notes") or "",
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
                    "added_at": wl.get("added_at") or dt.datetime.now().isoformat(),
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
                price_now = safe_float(inf.get("currentPrice") or inf.get("regularMarketPrice"))
                tgt = safe_float(item.get("target_buy"))
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
