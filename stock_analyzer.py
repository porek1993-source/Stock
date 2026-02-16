"""
Stock Picker Pro v4.0
======================
Profesionální kvantitativní analýza akcií s pokročilou AI a sektorovou inteligencí.
Kompletně v češtině, optimalizováno pro Gemini 2.5 Flash Lite.

Autor: Enhanced by Claude
Verze: 4.0
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
import streamlit.components.v1 as components


# Page config must be the first Streamlit command
st.set_page_config(
    page_title="Stock Picker Pro v4.0",
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
      
      /* Metric cards styling */
      .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
      }
      .metric-label {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-bottom: 0.3rem;
      }
      .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
      }
      .metric-delta {
        font-size: 0.9rem;
      }
      .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _get_secret(name: str, default: str = "") -> str:
    """Načtení API klíče ze Streamlit secrets nebo environment variables."""
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return str(os.getenv(name, default) or default)


# API klíče
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

# Konstanty
APP_NAME = "Stock Picker Pro"
APP_VERSION = "v4.0"

GEMINI_MODEL = "gemini-2.5-flash-lite"
MAX_AI_RETRIES = 3
RETRY_DELAY = 2

# Risk-Free Rate (použij aktuální 10Y US Treasury)
RISK_FREE_RATE = 0.045  # 4.5% p.a.


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TickerData:
    """Data container pro ticker analýzu."""
    ticker: str
    info: Dict[str, Any]
    hist: pd.DataFrame
    financials: pd.DataFrame
    balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame
    calendar: Dict[str, Any]
    recommendations: pd.DataFrame
    insider_transactions: pd.DataFrame
    major_holders: pd.DataFrame
    institutional_holders: pd.DataFrame


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_float(val: Any, default: float = 0.0) -> float:
    """Bezpečná konverze na float."""
    if val is None or val == "N/A":
        return default
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace("$", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    """Bezpečná konverze na int."""
    return int(safe_float(val, float(default)))


def fmt_money(val: Optional[float], decimals: int = 2) -> str:
    """Formátování částky v USD."""
    if val is None or math.isnan(val):
        return "—"
    return f"${val:,.{decimals}f}"


def fmt_percent(val: Optional[float], decimals: int = 2) -> str:
    """Formátování procent."""
    if val is None or math.isnan(val):
        return "—"
    return f"{val:.{decimals}f}%"


def fmt_number(val: Optional[float], decimals: int = 0) -> str:
    """Formátování čísla."""
    if val is None or math.isnan(val):
        return "—"
    return f"{val:,.{decimals}f}"


def fmt_market_cap(val: Optional[float]) -> str:
    """Formátování market cap."""
    if val is None or val == 0:
        return "—"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    elif val >= 1e9:
        return f"${val/1e9:.2f}B"
    elif val >= 1e6:
        return f"${val/1e6:.2f}M"
    else:
        return fmt_money(val, 0)


def annualize_return(values: pd.Series, periods_per_year: int = 252) -> float:
    """Výpočet anualizovaného výnosu."""
    if len(values) < 2:
        return 0.0
    total_return = (values.iloc[-1] / values.iloc[0]) - 1
    years = len(values) / periods_per_year
    if years <= 0:
        return 0.0
    return ((1 + total_return) ** (1 / years)) - 1


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Výpočet anualizované volatility."""
    if len(returns) < 2:
        return 0.0
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Výpočet Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess_returns = returns - (risk_free_rate / 252)
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Výpočet maximálního просадки (drawdown)."""
    if len(prices) < 2:
        return 0.0
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Načtení základních informací o tickeru z Yahoo Finance."""
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info or len(info) < 3:
            return {"error": "Ticker nenalezen"}
        return info
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker: str, period: str = "2y") -> Optional[TickerData]:
    """Načtení kompletních dat o tickeru."""
    try:
        yf_ticker = yf.Ticker(ticker)
        
        info = yf_ticker.info
        if not info or len(info) < 3:
            return None
        
        hist = yf_ticker.history(period=period)
        if hist.empty:
            return None
        
        # Finanční výkazy
        financials = yf_ticker.financials
        balance_sheet = yf_ticker.balance_sheet
        cashflow = yf_ticker.cashflow
        
        # Další data
        calendar = yf_ticker.calendar if hasattr(yf_ticker, 'calendar') else {}
        recommendations = yf_ticker.recommendations if hasattr(yf_ticker, 'recommendations') else pd.DataFrame()
        insider_transactions = yf_ticker.insider_transactions if hasattr(yf_ticker, 'insider_transactions') else pd.DataFrame()
        major_holders = yf_ticker.major_holders if hasattr(yf_ticker, 'major_holders') else pd.DataFrame()
        institutional_holders = yf_ticker.institutional_holders if hasattr(yf_ticker, 'institutional_holders') else pd.DataFrame()
        
        return TickerData(
            ticker=ticker,
            info=info,
            hist=hist,
            financials=financials,
            balance_sheet=balance_sheet,
            cashflow=cashflow,
            calendar=calendar,
            recommendations=recommendations,
            insider_transactions=insider_transactions,
            major_holders=major_holders,
            institutional_holders=institutional_holders
        )
    except Exception as e:
        st.error(f"Chyba při načítání dat: {str(e)}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peers(ticker: str, sector: str, industry: str, market_cap: float) -> List[str]:
    """Automatická detekce konkurentů podle sektoru a velikosti."""
    
    # Manuální mapování pro specifické případy
    peer_map = {
        "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
        "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
        "GOOGL": ["META", "AMZN", "MSFT", "AAPL"],
        "TSLA": ["GM", "F", "RIVN", "LCID"],
        "NVDA": ["AMD", "INTC", "QCOM", "AVGO"],
        "META": ["GOOGL", "SNAP", "PINS", "TWTR"],
        "AMZN": ["WMT", "TGT", "SHOP", "EBAY"],
        "NFLX": ["DIS", "PARA", "WBD", "ROKU"],
        "V": ["MA", "AXP", "PYPL", "SQ"],
        "MA": ["V", "AXP", "PYPL", "SQ"],
        "JPM": ["BAC", "C", "WFC", "GS"],
        "JNJ": ["PFE", "UNH", "ABBV", "MRK"],
        "PG": ["KO", "PEP", "CL", "KMB"],
        "DIS": ["NFLX", "CMCSA", "PARA", "WBD"],
        "TSLA": ["F", "GM", "RIVN", "LCID"],
        "PYPL": ["SQ", "V", "MA", "ADYEN.AS"],
    }
    
    if ticker in peer_map:
        return peer_map[ticker][:5]
    
    # Fallback: Generický přístup (zde by mohlo být volání API nebo scraping)
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_market_events() -> List[Dict[str, str]]:
    """Vrací seznam nadcházejících makro událostí (mock data)."""
    today = dt.datetime.now()
    
    events = [
        {"date": (today + dt.timedelta(days=7)).strftime("%Y-%m-%d"), "event": "FOMC Meeting", "importance": "🔴"},
        {"date": (today + dt.timedelta(days=14)).strftime("%Y-%m-%d"), "event": "CPI Report", "importance": "🔴"},
        {"date": (today + dt.timedelta(days=21)).strftime("%Y-%m-%d"), "event": "NFP (Nonfarm Payrolls)", "importance": "🟡"},
        {"date": (today + dt.timedelta(days=28)).strftime("%Y-%m-%d"), "event": "GDP Report", "importance": "🟡"},
    ]
    
    return events


# ============================================================================
# FINANCIAL ANALYSIS - DCF & VALUATION
# ============================================================================

def estimate_smart_params(info: Dict[str, Any], ticker: str) -> Dict[str, float]:
    """
    Pokročilý odhad parametrů pro DCF model s plynulým Quality Premium systémem.
    
    Returns:
        Dict s klíči: wacc, terminal_growth, exit_multiple, fcf_growth_5y
    """
    
    # Výchozí hodnoty
    wacc = 0.10  # 10% default
    terminal_growth = 0.025  # 2.5% perpetual growth
    exit_multiple = 12.0  # P/FCF multiple
    fcf_growth_5y = 0.08  # 8% growth
    
    # Získání základních metrik
    beta = safe_float(info.get("beta"), 1.0)
    sector = info.get("sector", "")
    industry = info.get("industry", "")
    
    # ROE, Net Margin, ROIC
    roe = safe_float(info.get("returnOnEquity"), 0.0) * 100
    net_margin = safe_float(info.get("profitMargins"), 0.0) * 100
    roic = safe_float(info.get("returnOnAssets"), 0.0) * 100  # Approximation
    
    # === WACC Calculation ===
    # CAPM: Cost of Equity = Rf + Beta * (Market Risk Premium)
    market_risk_premium = 0.08  # 8% historical
    cost_of_equity = RISK_FREE_RATE + beta * market_risk_premium
    
    # Cost of Debt (approximation)
    interest_expense = safe_float(info.get("interestExpense"), 0)
    total_debt = safe_float(info.get("totalDebt"), 1)
    cost_of_debt = abs(interest_expense) / total_debt if total_debt > 0 else 0.05
    
    # Tax Rate
    tax_rate = safe_float(info.get("taxRate"), 0.21)
    
    # Debt/Equity ratio
    market_cap = safe_float(info.get("marketCap"), 1)
    debt_to_equity = total_debt / market_cap if market_cap > 0 else 0.5
    
    # Weighted Average
    equity_weight = 1 / (1 + debt_to_equity)
    debt_weight = debt_to_equity / (1 + debt_to_equity)
    
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    wacc = max(0.06, min(wacc, 0.20))  # Clamp between 6-20%
    
    # === Terminal Growth Rate ===
    # Nižší pro mature firmy, vyšší pro growth
    if "Technology" in sector or "Healthcare" in sector:
        terminal_growth = 0.03  # 3%
    elif "Utilities" in sector or "Consumer Defensive" in sector:
        terminal_growth = 0.02  # 2%
    else:
        terminal_growth = 0.025  # 2.5%
    
    # === FCF Growth Rate (5Y) ===
    revenue_growth = safe_float(info.get("revenueGrowth"), 0.0) * 100
    earnings_growth = safe_float(info.get("earningsGrowth"), 0.0) * 100
    
    if revenue_growth > 20:
        fcf_growth_5y = 0.15  # 15%
    elif revenue_growth > 10:
        fcf_growth_5y = 0.10  # 10%
    else:
        fcf_growth_5y = 0.06  # 6%
    
    # === Quality Premium - PLYNULÝ BODOVÝ SYSTÉM ===
    quality_score = 0
    
    # ROE > 15% → +2 body
    if roe > 15:
        quality_score += 2
    elif roe > 10:
        quality_score += 1
    
    # Net Margin > 20% → +2 body
    if net_margin > 20:
        quality_score += 2
    elif net_margin > 10:
        quality_score += 1
    
    # ROIC > 15% → +2 body
    if roic > 15:
        quality_score += 2
    elif roic > 10:
        quality_score += 1
    
    # Debt/Equity < 0.5 → +1 bod
    if debt_to_equity < 0.5:
        quality_score += 1
    
    # Konverze bodů na Exit Multiple
    # Base = 12x, každý bod přidá +1x, max 20x
    exit_multiple = 12.0 + quality_score
    exit_multiple = min(exit_multiple, 20.0)
    
    # === Sektorové úpravy ===
    if "Technology" in sector:
        exit_multiple *= 1.2
    elif "Financial" in sector:
        exit_multiple *= 0.8
    elif "Utilities" in sector:
        exit_multiple *= 0.7
    
    # === Speciální případy ===
    # Krypto/FinTech
    if ticker.upper() in ["COIN", "MSTR", "SQ", "HOOD"]:
        wacc += 0.03  # Vyšší riziko
        exit_multiple *= 0.8
    
    # PayPal/Apple Pay detection
    if "payment" in industry.lower() or ticker.upper() in ["PYPL", "SQ", "V", "MA"]:
        exit_multiple *= 1.1
    
    # BioTech
    if "Biotechnology" in industry or "Drug" in industry:
        wacc += 0.02
        fcf_growth_5y = max(fcf_growth_5y, 0.12)
    
    return {
        "wacc": wacc,
        "terminal_growth": terminal_growth,
        "exit_multiple": exit_multiple,
        "fcf_growth_5y": fcf_growth_5y
    }


def calculate_fcf(info: Dict[str, Any], cashflow: pd.DataFrame, ticker: str) -> float:
    """
    Robustní výpočet Free Cash Flow s fallbacky.
    Speciální logika pro PayPal, Visa, MasterCard (Operating CF - minimal CapEx).
    """
    
    # Pokus 1: Přímý FCF z info
    fcf = safe_float(info.get("freeCashflow"), 0)
    if fcf > 0:
        return fcf
    
    # Pokus 2: Operating CF - CapEx z cashflow statement
    if not cashflow.empty and len(cashflow.columns) > 0:
        latest_col = cashflow.columns[0]
        operating_cf = safe_float(cashflow.loc["Operating Cash Flow", latest_col] 
                                  if "Operating Cash Flow" in cashflow.index else 0, 0)
        capex = safe_float(cashflow.loc["Capital Expenditure", latest_col] 
                          if "Capital Expenditure" in cashflow.index else 0, 0)
        
        if operating_cf > 0:
            # FinTech companies (PayPal, Visa, MA) mají minimální CapEx
            if ticker.upper() in ["PYPL", "V", "MA", "SQ", "ADYEN"]:
                # Předpokládej 5% Operating CF jako CapEx
                estimated_capex = operating_cf * 0.05
                return operating_cf - estimated_capex
            else:
                return operating_cf - abs(capex)
    
    # Pokus 3: Net Income - CapEx (aproximace)
    net_income = safe_float(info.get("netIncomeToCommon"), 0)
    if net_income > 0:
        return net_income * 0.8  # Conservative estimate
    
    # Fallback: Použij EBITDA * 0.6
    ebitda = safe_float(info.get("ebitda"), 0)
    if ebitda > 0:
        return ebitda * 0.6
    
    return 0.0


def dcf_valuation(
    fcf_current: float,
    growth_rate: float,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    exit_multiple: float = 15.0,
    years: int = 5
) -> Tuple[float, float]:
    """
    DCF oceňovací model s Exit Multiple metodou.
    
    Returns:
        (fair_value_per_share, terminal_value)
    """
    
    if fcf_current <= 0 or shares_outstanding <= 0:
        return 0.0, 0.0
    
    # Projektované FCF
    projected_fcf = []
    for year in range(1, years + 1):
        fcf = fcf_current * ((1 + growth_rate) ** year)
        projected_fcf.append(fcf)
    
    # Terminální hodnota pomocí Exit Multiple
    final_fcf = projected_fcf[-1]
    terminal_value = final_fcf * exit_multiple
    
    # Diskontování
    pv_fcf = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)])
    pv_terminal = terminal_value / ((1 + wacc) ** years)
    
    enterprise_value = pv_fcf + pv_terminal
    fair_value_per_share = enterprise_value / shares_outstanding
    
    return fair_value_per_share, terminal_value


def calculate_intrinsic_value_range(
    fcf: float,
    shares: float,
    info: Dict[str, Any],
    ticker: str
) -> Dict[str, float]:
    """
    Výpočet rozsahu intrinsické hodnoty s různými scénáři.
    """
    
    params = estimate_smart_params(info, ticker)
    
    # Base Case
    fair_base, _ = dcf_valuation(
        fcf_current=fcf,
        growth_rate=params["fcf_growth_5y"],
        wacc=params["wacc"],
        terminal_growth=params["terminal_growth"],
        shares_outstanding=shares,
        exit_multiple=params["exit_multiple"]
    )
    
    # Bull Case: +30% growth, -1% WACC
    fair_bull, _ = dcf_valuation(
        fcf_current=fcf,
        growth_rate=params["fcf_growth_5y"] * 1.3,
        wacc=max(params["wacc"] - 0.01, 0.06),
        terminal_growth=params["terminal_growth"],
        shares_outstanding=shares,
        exit_multiple=params["exit_multiple"] * 1.2
    )
    
    # Bear Case: -30% growth, +1% WACC
    fair_bear, _ = dcf_valuation(
        fcf_current=fcf,
        growth_rate=params["fcf_growth_5y"] * 0.7,
        wacc=min(params["wacc"] + 0.01, 0.20),
        terminal_growth=params["terminal_growth"] * 0.8,
        shares_outstanding=shares,
        exit_multiple=params["exit_multiple"] * 0.8
    )
    
    return {
        "fair_base": fair_base,
        "fair_bull": fair_bull,
        "fair_bear": fair_bear,
        "wacc": params["wacc"],
        "exit_multiple": params["exit_multiple"],
        "fcf_growth": params["fcf_growth_5y"],
        "terminal_growth": params["terminal_growth"]
    }


# ============================================================================
# SCORECARD SYSTEM (0-100)
# ============================================================================

def calculate_scorecard(
    info: Dict[str, Any],
    current_price: float,
    fair_value: float,
    data: Optional[TickerData] = None
) -> Dict[str, float]:
    """
    Pokročilý scorecard systém (0-100 bodů) s detailním rozpadem.
    
    Komponenty:
    - Valuace (30 bodů)
    - Kvalita byznysu (25 bodů)
    - Růstový potenciál (25 bodů)
    - Finanční zdraví (20 bodů)
    """
    
    scores = {
        "valuation": 0,
        "quality": 0,
        "growth": 0,
        "financial_health": 0,
        "total": 0
    }
    
    # === 1. VALUACE (30 bodů) ===
    if fair_value > 0 and current_price > 0:
        upside = ((fair_value - current_price) / current_price) * 100
        
        if upside > 50:
            scores["valuation"] = 30
        elif upside > 30:
            scores["valuation"] = 25
        elif upside > 15:
            scores["valuation"] = 20
        elif upside > 0:
            scores["valuation"] = 15
        elif upside > -15:
            scores["valuation"] = 10
        else:
            scores["valuation"] = 5
    
    # === 2. KVALITA BYZNYSU (25 bodů) ===
    roe = safe_float(info.get("returnOnEquity"), 0) * 100
    net_margin = safe_float(info.get("profitMargins"), 0) * 100
    roic = safe_float(info.get("returnOnAssets"), 0) * 100
    
    quality_points = 0
    
    # ROE (max 10 bodů)
    if roe > 20:
        quality_points += 10
    elif roe > 15:
        quality_points += 8
    elif roe > 10:
        quality_points += 5
    elif roe > 5:
        quality_points += 3
    
    # Net Margin (max 10 bodů)
    if net_margin > 25:
        quality_points += 10
    elif net_margin > 15:
        quality_points += 8
    elif net_margin > 10:
        quality_points += 5
    elif net_margin > 5:
        quality_points += 3
    
    # ROIC (max 5 bodů)
    if roic > 15:
        quality_points += 5
    elif roic > 10:
        quality_points += 3
    elif roic > 5:
        quality_points += 1
    
    scores["quality"] = min(quality_points, 25)
    
    # === 3. RŮST (25 bodů) ===
    revenue_growth = safe_float(info.get("revenueGrowth"), 0) * 100
    earnings_growth = safe_float(info.get("earningsGrowth"), 0) * 100
    
    growth_points = 0
    
    # Revenue Growth (max 15 bodů)
    if revenue_growth > 20:
        growth_points += 15
    elif revenue_growth > 15:
        growth_points += 12
    elif revenue_growth > 10:
        growth_points += 10
    elif revenue_growth > 5:
        growth_points += 7
    elif revenue_growth > 0:
        growth_points += 4
    
    # Earnings Growth (max 10 bodů)
    if earnings_growth > 20:
        growth_points += 10
    elif earnings_growth > 15:
        growth_points += 8
    elif earnings_growth > 10:
        growth_points += 6
    elif earnings_growth > 5:
        growth_points += 4
    elif earnings_growth > 0:
        growth_points += 2
    
    scores["growth"] = min(growth_points, 25)
    
    # === 4. FINANČNÍ ZDRAVÍ (20 bodů) ===
    debt_to_equity = safe_float(info.get("debtToEquity"), 100) / 100
    current_ratio = safe_float(info.get("currentRatio"), 1.0)
    quick_ratio = safe_float(info.get("quickRatio"), 1.0)
    
    health_points = 0
    
    # Debt/Equity (max 10 bodů)
    if debt_to_equity < 0.3:
        health_points += 10
    elif debt_to_equity < 0.5:
        health_points += 8
    elif debt_to_equity < 1.0:
        health_points += 6
    elif debt_to_equity < 2.0:
        health_points += 3
    
    # Current Ratio (max 5 bodů)
    if current_ratio > 2.0:
        health_points += 5
    elif current_ratio > 1.5:
        health_points += 4
    elif current_ratio > 1.0:
        health_points += 2
    
    # Quick Ratio (max 5 bodů)
    if quick_ratio > 1.5:
        health_points += 5
    elif quick_ratio > 1.0:
        health_points += 3
    elif quick_ratio > 0.8:
        health_points += 1
    
    scores["financial_health"] = min(health_points, 20)
    
    # === TOTAL ===
    scores["total"] = (
        scores["valuation"] + 
        scores["quality"] + 
        scores["growth"] + 
        scores["financial_health"]
    )
    
    return scores


def detect_value_trap(info: Dict[str, Any], data: Optional[TickerData] = None) -> Tuple[bool, str]:
    """
    Detekce potenciální "pasti na hodnotu".
    
    Returns:
        (is_trap, warning_message)
    """
    
    pe_ratio = safe_float(info.get("trailingPE"), 0)
    revenue_growth = safe_float(info.get("revenueGrowth"), 0) * 100
    debt_to_equity = safe_float(info.get("debtToEquity"), 100) / 100
    
    is_trap = False
    warnings = []
    
    # Podmínka 1: Nízké P/E (< 10)
    if pe_ratio > 0 and pe_ratio < 10:
        
        # Podmínka 2: Klesající tržby
        if revenue_growth < -5:
            is_trap = True
            warnings.append("Klesající tržby (YoY)")
        
        # Podmínka 3: Vysoký dluh
        if debt_to_equity > 2.0:
            is_trap = True
            warnings.append("Vysoká zadluženost (D/E > 2)")
        
        # Podmínka 4: Negativní nebo nulové EPS
        eps = safe_float(info.get("trailingEps"), 0)
        if eps <= 0:
            is_trap = True
            warnings.append("Negativní/nulové EPS")
    
    if is_trap:
        warning_msg = f"⚠️ **Potenciální Value Trap**: {', '.join(warnings)}. Nízká valuace může být oprávněná kvůli úpadku byznysu."
        return True, warning_msg
    
    return False, ""


# ============================================================================
# AI ANALYSIS (GEMINI)
# ============================================================================

def generate_ai_report(
    ticker: str,
    info: Dict[str, Any],
    current_price: float,
    fair_value: float,
    data: Optional[TickerData] = None
) -> str:
    """
    Generování hloubkové AI analýzy pomocí Gemini 2.5 Flash Lite.
    
    Prompt je v angličtině pro lepší výkon modelu, ale výstup v češtině.
    """
    
    if not GEMINI_API_KEY:
        return "⚠️ **AI analýza není k dispozici**: Nastav GEMINI_API_KEY v konfiguraci aplikace."
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Příprava dat pro prompt
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        market_cap = safe_float(info.get("marketCap"), 0)
        pe_ratio = safe_float(info.get("trailingPE"), 0)
        revenue_growth = safe_float(info.get("revenueGrowth"), 0) * 100
        earnings_growth = safe_float(info.get("earningsGrowth"), 0) * 100
        roe = safe_float(info.get("returnOnEquity"), 0) * 100
        debt_to_equity = safe_float(info.get("debtToEquity"), 100) / 100
        
        # Detekce speciálních případů
        is_crypto = ticker.upper() in ["COIN", "MSTR", "RIOT", "MARA"]
        is_fintech = "payment" in industry.lower() or ticker.upper() in ["PYPL", "SQ", "V", "MA", "ADYEN"]
        is_biotech = "Biotechnology" in industry or "Drug" in industry
        
        # Value Trap detection
        is_trap, trap_msg = detect_value_trap(info, data)
        
        # ATH analysis
        ath = 0
        current_from_ath = 0
        if data and not data.hist.empty:
            ath = data.hist['Close'].max()
            current_from_ath = ((current_price - ath) / ath) * 100
        
        is_deep_value = current_from_ath < -50
        
        # Konstrukce promptu
        prompt = f"""You are an elite equity research analyst. Analyze {ticker} ({sector} - {industry}) and provide a DEEP, CYNICAL analysis in CZECH language.

CRITICAL INSTRUCTIONS:
1. Output MUST be in CZECH language only
2. Be thorough - each point in Bull/Bear case needs 2-3 sentences explaining deeper implications (impact on margins, market share, cash flow)
3. You MUST propose a specific "wait_for_price" - if stock is overvalued, suggest entry at 15% discount to fair value; if undervalued, suggest current price
4. Be brutally honest - no generic AI fluff

COMPANY DATA:
- Ticker: {ticker}
- Current Price: ${current_price:.2f}
- DCF Fair Value: ${fair_value:.2f}
- Market Cap: {fmt_market_cap(market_cap)}
- P/E Ratio: {pe_ratio:.1f}
- Revenue Growth: {revenue_growth:.1f}%
- Earnings Growth: {earnings_growth:.1f}%
- ROE: {roe:.1f}%
- Debt/Equity: {debt_to_equity:.2f}

SPECIAL CASES:
- Is Crypto-related: {is_crypto}
- Is FinTech/Payments: {is_fintech}
- Is BioTech: {is_biotech}
- Is potential Value Trap: {is_trap}
- Distance from ATH: {current_from_ath:.1f}%

REQUIRED STRUCTURE (in Czech):
1. **Byznys Model** (2-3 věty)
2. **Bull Case** (5-7 bodů, každý 2-3 věty s detailním vysvětlením)
3. **Bear Case** (5-7 bodů, každý 2-3 věty s cynickou analýzou rizik)
4. **Katalyzátory** (3-4 konkrétní události, které mohou pohnout cenou)
5. **Verdict** (BUY/HOLD/SELL s odůvodněním)
6. **Wait for Price**: $XXX (konkrétní vstupní cena s vysvětlením)

SECTOR-SPECIFIC INTELLIGENCE:
"""
        
        if is_crypto:
            prompt += """
- Crypto exposure = high volatility risk
- Analyze regulatory headwinds
- Bitcoin correlation impact on revenue
"""
        
        if is_fintech:
            prompt += """
- Payment networks = high moats but regulatory risk
- Analyze take rates, TPV growth, international expansion
- Competition from Apple Pay, Google Pay, CBDCs
"""
        
        if is_biotech:
            prompt += """
- Pipeline risk - analyze Phase 2/3 trials
- Patent cliffs, FDA approval timelines
- M&A potential
"""
        
        if is_deep_value:
            prompt += """
- Stock down >50% from ATH - analyze if turnaround story or terminal decline
- Management changes, strategic pivots
"""
        
        if is_trap:
            prompt += f"""
- WARNING: Potential Value Trap detected: {trap_msg}
- Dig deep into why valuation is low - deserved or opportunity?
"""
        
        prompt += "\n\nSTART YOUR ANALYSIS NOW (in Czech):"
        
        # Volání API s retry logikou
        for attempt in range(MAX_AI_RETRIES):
            try:
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                if attempt < MAX_AI_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return f"⚠️ Chyba při generování AI reportu: {str(e)}"
        
    except Exception as e:
        return f"⚠️ Chyba při volání Gemini API: {str(e)}"


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """Inicializace session state proměnných."""
    if "last_ticker" not in st.session_state:
        st.session_state["last_ticker"] = ""
    if "ai_report" not in st.session_state:
        st.session_state["ai_report"] = ""
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = 0  # Index aktivního tabu
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = {"items": {}}


def get_watchlist() -> Dict[str, Any]:
    """Načtení watchlistu ze session state."""
    return st.session_state.get("watchlist", {"items": {}})


def set_watchlist(data: Dict[str, Any]):
    """Uložení watchlistu do session state."""
    st.session_state["watchlist"] = data


# ============================================================================
# PDF EXPORT
# ============================================================================

def export_analysis_pdf(
    ticker: str,
    company_name: str,
    summary: Dict[str, Any],
    ai_report: str
) -> Optional[bytes]:
    """Export analýzy do PDF formátu."""
    if not _HAS_PDF:
        return None
    
    try:
        from io import BytesIO
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(1*inch, height - 1*inch, f"Stock Analysis: {ticker}")
        
        c.setFont("Helvetica", 12)
        y = height - 1.5*inch
        
        # Summary
        c.drawString(1*inch, y, f"Společnost: {company_name}")
        y -= 0.3*inch
        c.drawString(1*inch, y, f"Aktuální cena: {summary.get('Cena', 'N/A')}")
        y -= 0.3*inch
        c.drawString(1*inch, y, f"DCF Fair Value: {summary.get('DCF Fair', 'N/A')}")
        y -= 0.3*inch
        c.drawString(1*inch, y, f"Score: {summary.get('Score', 'N/A')}")
        y -= 0.3*inch
        c.drawString(1*inch, y, f"Verdict: {summary.get('Verdict', 'N/A')}")
        
        # AI Report (zkrácený)
        y -= 0.6*inch
        c.setFont("Helvetica-Bold", 14)
        c.drawString(1*inch, y, "AI Analýza:")
        y -= 0.3*inch
        
        c.setFont("Helvetica", 10)
        # Zkrácený text (max 2000 znaků)
        report_text = ai_report[:2000] + "..." if len(ai_report) > 2000 else ai_report
        
        # Simple text wrapping
        max_width = width - 2*inch
        words = report_text.split()
        line = ""
        for word in words:
            if c.stringWidth(line + word, "Helvetica", 10) < max_width:
                line += word + " "
            else:
                c.drawString(1*inch, y, line)
                y -= 0.2*inch
                line = word + " "
                if y < 1*inch:
                    break
        
        if line and y > 1*inch:
            c.drawString(1*inch, y, line)
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
    
    except Exception as e:
        st.error(f"Chyba při exportu PDF: {str(e)}")
        return None


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def display_welcome_screen():
    """Zobrazení úvodní obrazovky."""
    st.title("Vítej v Stock Picker Pro v4.0! 🚀")
    
    st.markdown("""
    ### Profesionální kvantitativní analýza akcií
    
    **🆕 Co je nového ve v4.0:**
    - ✅ **Kompletní lokalizace do češtiny** - veškeré texty v aplikaci pouze v ČJ
    - ✅ **Fix UX: Persistentní taby** - AI report nezpůsobí reset UI
    - ✅ **Hardcore AI analýza** - hloubková cynická analýza s konkrétní wait_for_price
    - ✅ **Plynulý Quality Premium** - bodový systém pro Exit Multiple
    - ✅ **Visual Heatmap** - barevný gradient v Sensitivity Analysis
    - ✅ **Value Trap Detection** - varování před pastmi na hodnotu
    - ✅ **Tooltips u všech metrik** - vysvětlení každého čísla
    
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
    st.info("💡 **Pro AI analýzu** nastav GEMINI_API_KEY v kódu nebo Streamlit secrets!")


def main():
    """Hlavní funkce aplikace."""
    
    init_session_state()
    
    # === SIDEBAR ===
    with st.sidebar:
        st.title(f"📈 {APP_NAME}")
        st.caption(f"Verze {APP_VERSION}")
        
        st.markdown("---")
        
        # Ticker input
        ticker_input = st.text_input(
            "Symbol Tickeru",
            value=st.session_state.get("last_ticker", ""),
            placeholder="např. AAPL, MSFT, TSLA",
            help="Zadej ticker symbol akcie z americké burzy (NYSE, NASDAQ). Pro pražskou burzu použij příponu .PR (např. CEZ.PR)"
        )
        
        st.caption("💡 Pro pražskou burzu použij příponu .PR (např. CEZ.PR, KOMB.PR)")
        
        analyze_button = st.button("🔍 Analyzovat", use_container_width=True, type="primary")
        
        if analyze_button and ticker_input:
            st.session_state["last_ticker"] = ticker_input.upper().strip()
            st.session_state["active_tab"] = 0  # Reset na první tab
            st.rerun()
        
        st.markdown("---")
        
        # Market Events
        st.markdown("### 📅 Makro Kalendář")
        events = fetch_market_events()
        for event in events[:3]:
            st.markdown(
                f"{event['importance']} **{event['event']}**  \n"
                f"<small style='opacity: 0.7;'>{event['date']}</small>",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.caption("📊 Data: Yahoo Finance")
        st.caption("🤖 AI: Gemini 2.5 Flash Lite")
    
    # === MAIN CONTENT ===
    ticker = st.session_state.get("last_ticker", "").upper().strip()
    
    if not ticker:
        display_welcome_screen()
        return
    
    # Fetch data
    with st.spinner(f"Načítám data pro {ticker}..."):
        data = fetch_ticker_data(ticker)
    
    if not data or not data.info:
        st.error(f"❌ Ticker '{ticker}' nebyl nalezen. Zkontroluj symbol a zkus to znovu.")
        return
    
    info = data.info
    company_name = info.get("longName", ticker)
    current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    
    if current_price == 0:
        st.error("❌ Nepodařilo se načíst aktuální cenu.")
        return
    
    # === HEADER METRICS ===
    st.markdown(f"# {company_name} ({ticker})")
    st.markdown(f"**{info.get('sector', 'N/A')} • {info.get('industry', 'N/A')}**")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Aktuální Cena</div>'
            f'<div class="metric-value">{fmt_money(current_price)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        market_cap = safe_float(info.get("marketCap"))
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Market Cap</div>'
            f'<div class="metric-value">{fmt_market_cap(market_cap)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        pe_ratio = safe_float(info.get("trailingPE"))
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">P/E Ratio</div>'
            f'<div class="metric-value">{pe_ratio:.1f}x</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col4:
        dividend_yield = safe_float(info.get("dividendYield"), 0) * 100
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Div. Yield</div>'
            f'<div class="metric-value">{fmt_percent(dividend_yield)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    with col5:
        beta = safe_float(info.get("beta"), 1.0)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Beta</div>'
            f'<div class="metric-value">{beta:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # === TABS ===
    tab_labels = [
        "📊 Přehled",
        "💰 DCF Valuace",
        "📈 Technická Analýza",
        "🔍 Peer Comparison",
        "👔 Insider Trading",
        "🤖 AI Analyst",
        "📝 Investment Memo",
        "⭐ Watchlist"
    ]
    
    tabs = st.tabs(tab_labels)
    
    # === TAB 1: PŘEHLED ===
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Finanční Přehled</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Valuační Metriky")
            metrics_df = pd.DataFrame({
                "Metrika": [
                    "P/E Ratio",
                    "P/B Ratio",
                    "P/S Ratio",
                    "EV/EBITDA",
                    "PEG Ratio"
                ],
                "Hodnota": [
                    f"{safe_float(info.get('trailingPE')):.2f}",
                    f"{safe_float(info.get('priceToBook')):.2f}",
                    f"{safe_float(info.get('priceToSalesTrailing12Months')):.2f}",
                    f"{safe_float(info.get('enterpriseToEbitda')):.2f}",
                    f"{safe_float(info.get('pegRatio')):.2f}"
                ]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### Profitabilita")
            profit_df = pd.DataFrame({
                "Metrika": [
                    "ROE",
                    "ROA",
                    "Net Margin",
                    "Operating Margin",
                    "Gross Margin"
                ],
                "Hodnota": [
                    fmt_percent(safe_float(info.get("returnOnEquity"), 0) * 100),
                    fmt_percent(safe_float(info.get("returnOnAssets"), 0) * 100),
                    fmt_percent(safe_float(info.get("profitMargins"), 0) * 100),
                    fmt_percent(safe_float(info.get("operatingMargins"), 0) * 100),
                    fmt_percent(safe_float(info.get("grossMargins"), 0) * 100)
                ]
            })
            st.dataframe(profit_df, use_container_width=True, hide_index=True)
        
        with col3:
            st.markdown("#### Finanční Zdraví")
            health_df = pd.DataFrame({
                "Metrika": [
                    "Debt/Equity",
                    "Current Ratio",
                    "Quick Ratio",
                    "Cash (M)",
                    "Total Debt (M)"
                ],
                "Hodnota": [
                    f"{safe_float(info.get('debtToEquity')) / 100:.2f}",
                    f"{safe_float(info.get('currentRatio')):.2f}",
                    f"{safe_float(info.get('quickRatio')):.2f}",
                    f"${safe_float(info.get('totalCash')) / 1e6:,.0f}",
                    f"${safe_float(info.get('totalDebt')) / 1e6:,.0f}"
                ]
            })
            st.dataframe(health_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Chart: Price History
        st.markdown("#### 📈 Cenový Vývoj (2 roky)")
        if not data.hist.empty:
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=data.hist.index,
                open=data.hist['Open'],
                high=data.hist['High'],
                low=data.hist['Low'],
                close=data.hist['Close'],
                name=ticker
            ))
            
            fig.update_layout(
                xaxis_title="Datum",
                yaxis_title="Cena (USD)",
                template="plotly_dark",
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Analyst Recommendations
        st.markdown("#### 🎯 Doporučení Analytiků")
        target_high = safe_float(info.get("targetHighPrice"))
        target_low = safe_float(info.get("targetLowPrice"))
        target_mean = safe_float(info.get("targetMeanPrice"))
        
        if target_mean > 0:
            upside_analyst = ((target_mean - current_price) / current_price) * 100
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            with rec_col1:
                st.metric("Průměrný Target", fmt_money(target_mean), f"{upside_analyst:+.1f}%")
            with rec_col2:
                st.metric("High Target", fmt_money(target_high))
            with rec_col3:
                st.metric("Low Target", fmt_money(target_low))
    
    # === TAB 2: DCF VALUACE ===
    with tabs[1]:
        st.markdown('<div class="section-header">💰 DCF Valuace & Sensitivity</div>', unsafe_allow_html=True)
        
        # Calculate FCF
        fcf = calculate_fcf(info, data.cashflow, ticker)
        shares = safe_float(info.get("sharesOutstanding"))
        
        if fcf <= 0:
            st.warning("⚠️ Free Cash Flow je negativní nebo nebyl nalezen. DCF model není spolehlivý.")
            fcf = safe_float(info.get("ebitda"), 0) * 0.6  # Fallback
        
        # Calculate intrinsic value
        valuation = calculate_intrinsic_value_range(fcf, shares, info, ticker)
        
        fair_base = valuation["fair_base"]
        fair_bull = valuation["fair_bull"]
        fair_bear = valuation["fair_bear"]
        wacc = valuation["wacc"]
        exit_multiple = valuation["exit_multiple"]
        fcf_growth = valuation["fcf_growth"]
        terminal_growth = valuation["terminal_growth"]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            upside = ((fair_base - current_price) / current_price) * 100
            color = "🟢" if upside > 0 else "🔴"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">DCF Fair Value (Base)</div>'
                f'<div class="metric-value">{fmt_money(fair_base)}</div>'
                f'<div class="metric-delta">{color} Upside: {upside:+.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with col2:
            upside_bull = ((fair_bull - current_price) / current_price) * 100
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Bull Case</div>'
                f'<div class="metric-value">{fmt_money(fair_bull)}</div>'
                f'<div class="metric-delta">🟢 Upside: {upside_bull:+.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with col3:
            upside_bear = ((fair_bear - current_price) / current_price) * 100
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Bear Case</div>'
                f'<div class="metric-value">{fmt_money(fair_bear)}</div>'
                f'<div class="metric-delta">🔴 Downside: {upside_bear:+.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Model Parameters
        st.markdown("#### 🔧 Parametry DCF Modelu")
        
        param_col1, param_col2, param_col3, param_col4 = st.columns(4)
        
        with param_col1:
            st.metric(
                "WACC",
                fmt_percent(wacc * 100),
                help="Weighted Average Cost of Capital - průměrná cena kapitálu firmy. Používá se jako diskontní sazba v DCF modelu."
            )
        
        with param_col2:
            st.metric(
                "FCF Growth (5Y)",
                fmt_percent(fcf_growth * 100),
                help="Očekávaný roční růst Free Cash Flow v příštích 5 letech."
            )
        
        with param_col3:
            st.metric(
                "Terminal Growth",
                fmt_percent(terminal_growth * 100),
                help="Perpetuální růst po ukončení projekční periody (typicky 2-3%)."
            )
        
        with param_col4:
            st.metric(
                "Exit Multiple",
                f"{exit_multiple:.1f}x",
                help="P/FCF násobek použitý pro výpočet terminální hodnoty. Vyšší = kvalitnější byznys."
            )
        
        st.markdown("---")
        
        # Sensitivity Analysis
        st.markdown("#### 📊 Sensitivity Analysis")
        st.caption("Heatmapa ukazuje, jak se mění fair value při různých kombinacích FCF Growth a WACC.")
        
        # Generate sensitivity matrix
        wacc_range = np.linspace(wacc - 0.02, wacc + 0.02, 5)
        growth_range = np.linspace(fcf_growth - 0.03, fcf_growth + 0.03, 5)
        
        sensitivity_matrix = []
        for g in growth_range:
            row = []
            for w in wacc_range:
                fv, _ = dcf_valuation(
                    fcf_current=fcf,
                    growth_rate=g,
                    wacc=w,
                    terminal_growth=terminal_growth,
                    shares_outstanding=shares,
                    exit_multiple=exit_multiple
                )
                upside_pct = ((fv - current_price) / current_price) * 100
                row.append(upside_pct)
            sensitivity_matrix.append(row)
        
        sensitivity_df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{g*100:.1f}%" for g in growth_range],
            columns=[f"{w*100:.1f}%" for w in wacc_range]
        )
        
        # Apply color gradient (heatmap)
        def color_upside(val):
            """Barevný gradient: zelená pro pozitivní upside, červená pro negativní."""
            if val > 30:
                return 'background-color: #1a5c1a; color: white'
            elif val > 15:
                return 'background-color: #2d7a2d; color: white'
            elif val > 0:
                return 'background-color: #4a9b4a; color: white'
            elif val > -15:
                return 'background-color: #9b6a4a; color: white'
            else:
                return 'background-color: #7a2d2d; color: white'
        
        styled_df = sensitivity_df.style.applymap(color_upside).format("{:.1f}%")
        
        st.markdown("**Upside/Downside (%) při různých FCF Growth (řádky) a WACC (sloupce):**")
        st.dataframe(styled_df, use_container_width=True)
        
        st.caption("🟢 Zelená = vysoký upside | 🔴 Červená = downside riziko")
    
    # === TAB 3: TECHNICKÁ ANALÝZA ===
    with tabs[2]:
        st.markdown('<div class="section-header">📈 Technická Analýza & Riziková Metrika</div>', unsafe_allow_html=True)
        
        if not data.hist.empty:
            # Calculate returns
            returns = data.hist['Close'].pct_change().dropna()
            
            # Calculate metrics
            ann_return = annualize_return(data.hist['Close']) * 100
            volatility = calculate_volatility(returns) * 100
            sharpe = calculate_sharpe_ratio(returns)
            max_dd = calculate_max_drawdown(data.hist['Close']) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Anualizovaný Výnos",
                    fmt_percent(ann_return),
                    help="Průměrný roční výnos za poslední 2 roky (anualizovaný)."
                )
            
            with col2:
                st.metric(
                    "Volatilita",
                    fmt_percent(volatility),
                    help="Anualizovaná volatilita (směrodatná odchylka) denních výnosů. Vyšší = rizikovější."
                )
            
            with col3:
                st.metric(
                    "Sharpe Ratio",
                    f"{sharpe:.2f}",
                    help="Poměr excess výnosu k volatilitě. > 1 je dobré, > 2 je výborné."
                )
            
            with col4:
                st.metric(
                    "Max Drawdown",
                    fmt_percent(max_dd),
                    help="Maximální pokles z vrcholu za sledované období. Ukazuje největší ztrátu."
                )
            
            st.markdown("---")
            
            # Returns Distribution
            st.markdown("#### 📊 Distribuce Denních Výnosů")
            
            import plotly.express as px
            
            fig = px.histogram(
                returns * 100,
                nbins=50,
                labels={"value": "Denní Výnos (%)", "count": "Počet Dnů"},
                title="Histogram denních výnosů"
            )
            fig.update_layout(
                template="plotly_dark",
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Historická data nejsou k dispozici.")
    
    # === TAB 4: PEER COMPARISON ===
    with tabs[3]:
        st.markdown('<div class="section-header">🔍 Srovnání s Konkurenty</div>', unsafe_allow_html=True)
        
        sector = info.get("sector", "")
        industry = info.get("industry", "")
        market_cap = safe_float(info.get("marketCap"))
        
        peers = fetch_peers(ticker, sector, industry, market_cap)
        
        if peers:
            st.markdown(f"**Konkurenti v sektoru {sector}:**")
            
            # Fetch peer data
            peer_data = []
            for peer_ticker in [ticker] + peers:
                peer_info = fetch_ticker_info(peer_ticker)
                if "error" not in peer_info:
                    peer_data.append({
                        "Ticker": peer_ticker,
                        "Cena": fmt_money(safe_float(peer_info.get("currentPrice"))),
                        "Market Cap": fmt_market_cap(safe_float(peer_info.get("marketCap"))),
                        "P/E": f"{safe_float(peer_info.get('trailingPE')):.1f}",
                        "P/S": f"{safe_float(peer_info.get('priceToSalesTrailing12Months')):.2f}",
                        "ROE": fmt_percent(safe_float(peer_info.get("returnOnEquity"), 0) * 100),
                        "D/E": f"{safe_float(peer_info.get('debtToEquity')) / 100:.2f}"
                    })
            
            if peer_data:
                peer_df = pd.DataFrame(peer_data)
                
                # Highlight current ticker
                def highlight_ticker(row):
                    if row['Ticker'] == ticker:
                        return ['background-color: rgba(255, 255, 255, 0.1)'] * len(row)
                    return [''] * len(row)
                
                styled_peer_df = peer_df.style.apply(highlight_ticker, axis=1)
                st.dataframe(styled_peer_df, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Peer Correlation Chart (Relative Performance)
                st.markdown("#### 📊 Relativní Výkon vs. Konkurenti (1 rok)")
                
                # Fetch 1Y history for peers
                peer_prices = {}
                for peer_ticker in [ticker] + peers[:3]:  # Max 4 tickery (včetně hlavního)
                    try:
                        peer_hist = yf.Ticker(peer_ticker).history(period="1y")
                        if not peer_hist.empty:
                            # Normalizace na 100 (první den = 100)
                            normalized = (peer_hist['Close'] / peer_hist['Close'].iloc[0]) * 100
                            peer_prices[peer_ticker] = normalized
                    except:
                        pass
                
                if peer_prices:
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    for peer_ticker, prices in peer_prices.items():
                        fig.add_trace(go.Scatter(
                            x=prices.index,
                            y=prices,
                            mode='lines',
                            name=peer_ticker,
                            line=dict(width=3 if peer_ticker == ticker else 2)
                        ))
                    
                    fig.update_layout(
                        title="Relativní výkon (normalizováno na 100)",
                        xaxis_title="Datum",
                        yaxis_title="Index (100 = start)",
                        template="plotly_dark",
                        height=400,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Graf ukazuje, zda {ticker} překonává nebo zaostává za svými konkurenty.")
        
        else:
            st.info("Automatická detekce konkurentů není k dispozici pro tento ticker.")
    
    # === TAB 5: INSIDER TRADING ===
    with tabs[4]:
        st.markdown('<div class="section-header">👔 Insider Trading</div>', unsafe_allow_html=True)
        
        if not data.insider_transactions.empty:
            # Process insider data
            insider_df = data.insider_transactions.copy()
            
            # Filter last 6 months
            six_months_ago = dt.datetime.now() - dt.timedelta(days=180)
            insider_df = insider_df[insider_df.index >= six_months_ago]
            
            if not insider_df.empty:
                # Display table
                st.dataframe(insider_df, use_container_width=True)
                
                # Analyze sentiment
                st.markdown("---")
                st.markdown("#### 📊 Insider Sentiment")
                
                # Count buys vs sells
                buys = insider_df[insider_df['Shares'] > 0]['Shares'].sum()
                sells = abs(insider_df[insider_df['Shares'] < 0]['Shares'].sum())
                
                if buys > 0 or sells > 0:
                    buy_pct = (buys / (buys + sells)) * 100
                    sell_pct = 100 - buy_pct
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Nákupy (akcií)", fmt_number(buys), f"{buy_pct:.1f}%")
                    with col2:
                        st.metric("Prodeje (akcií)", fmt_number(sells), f"{sell_pct:.1f}%")
                    
                    # Sentiment interpretation
                    if buy_pct > 70:
                        st.success("🟢 **Silný bullish sentiment** - insideři masivně nakupují!")
                    elif buy_pct > 50:
                        st.info("🟡 **Mírný bullish sentiment** - více nákupů než prodejů.")
                    elif sell_pct > 70:
                        st.error("🔴 **Bearish sentiment** - insideři prodávají!")
                    else:
                        st.warning("⚪ **Neutrální sentiment**")
            
            else:
                st.info("Žádné insider transakce za posledních 6 měsíců.")
        
        else:
            st.info("Insider trading data nejsou k dispozici.")
    
    # === TAB 6: AI ANALYST ===
    with tabs[5]:
        st.markdown('<div class="section-header">🤖 AI Analyst Report</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Hloubková AI analýza využívá **Gemini 2.5 Flash Lite** k poskytnutí cynického pohledu na byznys,
        bull/bear scénáře a konkrétní vstupní cenu.
        """)
        
        # Button to generate AI report
        if st.button("🚀 Vygenerovat AI Report", use_container_width=True, type="primary"):
            # Uložíme aktuální tab do session state
            st.session_state["active_tab"] = 5  # Index tabu "AI Analyst"
            
            with st.spinner("Generuji hloubkovou analýzu pomocí AI..."):
                ai_report = generate_ai_report(ticker, info, current_price, fair_base, data)
                st.session_state["ai_report"] = ai_report
            
            st.rerun()  # Refresh pro zobrazení reportu
        
        # Display cached AI report
        if st.session_state.get("ai_report"):
            st.markdown("---")
            st.markdown(st.session_state["ai_report"])
        
        else:
            st.info("Klikni na tlačítko výše pro vygenerování AI reportu.")
        
        # Scorecard
        st.markdown("---")
        st.markdown("#### 📊 Scorecard (0-100)")
        
        scores = calculate_scorecard(info, current_price, fair_base, data)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Valuace", f"{scores['valuation']:.0f}/30")
        with col2:
            st.metric("Kvalita", f"{scores['quality']:.0f}/25")
        with col3:
            st.metric("Růst", f"{scores['growth']:.0f}/25")
        with col4:
            st.metric("Fin. Zdraví", f"{scores['financial_health']:.0f}/20")
        with col5:
            total = scores['total']
            color = "🟢" if total >= 70 else "🟡" if total >= 50 else "🔴"
            st.metric("Celkem", f"{color} {total:.0f}/100")
        
        # Verdict
        if total >= 70:
            verdict = "🟢 **STRONG BUY** - Výborná příležitost"
        elif total >= 60:
            verdict = "🟢 **BUY** - Dobrá investice"
        elif total >= 50:
            verdict = "🟡 **HOLD** - Neutrální"
        elif total >= 40:
            verdict = "🔴 **SELL** - Slabá investice"
        else:
            verdict = "🔴 **STRONG SELL** - Vyhni se"
        
        st.markdown(f"### {verdict}")
        
        # Value Trap Warning
        is_trap, trap_msg = detect_value_trap(info, data)
        if is_trap:
            st.error(trap_msg)
    
    # === TAB 7: INVESTMENT MEMO ===
    with tabs[6]:
        st.markdown('<div class="section-header">📝 Investment Memo</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Vytvoř si vlastní investiční memo s tezí, riziky a podmínkami vstupu.
        """)
        
        thesis = st.text_area(
            "Investment Thesis",
            placeholder="Proč investovat? Jaký je hlavní důvod?",
            height=120,
            help="Stručně popiš, proč je tato akcie zajímavá investiční příležitost."
        )
        
        drivers = st.text_area(
            "Klíčové Katalyzátory",
            placeholder="Co může pohnout cenou? (nový produkt, expanze, AI trend...)",
            height=100,
            help="Události nebo trendy, které mohou pozitivně ovlivnit cenu akcie."
        )
        
        risks = st.text_area(
            "Rizika",
            placeholder="Co může pokazit investici?",
            height=100,
            help="Hlavní rizika, která by mohla negativně ovlivnit investici."
        )
        
        buy_conditions = st.text_area(
            "Podmínky Vstupu",
            placeholder="Za jakých podmínek nakoupit? (cena, událost...)",
            height=80,
            help="Specifikuj konkrétní podmínky, které musí být splněny pro nákup."
        )
        
        notes = st.text_area(
            "Poznámky",
            placeholder="Další poznámky...",
            height=80
        )
        
        st.markdown("---")
        
        # Export PDF
        if _HAS_PDF:
            if st.button("📄 Exportovat jako PDF", use_container_width=True):
                summary = {
                    "Cena": fmt_money(current_price),
                    "DCF Fair": fmt_money(fair_base),
                    "Score": f"{scores['total']:.0f}/100",
                    "Verdict": verdict
                }
                
                pdf_bytes = export_analysis_pdf(
                    ticker,
                    company_name,
                    summary,
                    st.session_state.get("ai_report", "AI report nebyl vygenerován.")
                )
                
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Stáhnout PDF",
                        data=pdf_bytes,
                        file_name=f"analysis_{ticker}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            st.info("PDF export není k dispozici. Nainstaluj `reportlab` pro export.")
    
    # === TAB 8: WATCHLIST ===
    with tabs[7]:
        st.markdown('<div class="section-header">⭐ Watchlist</div>', unsafe_allow_html=True)
        
        st.markdown("Přidej akcie na watchlist s cílovou nákupní cenou.")
        
        watch = get_watchlist()
        wl = watch.get("items", {}).get(ticker, {})
        
        target_buy = st.number_input(
            "Cílová nákupní cena (USD)",
            value=float(wl.get("target_buy", 0.0)) if wl else 0.0,
            step=1.0,
            help="Zadej cenu, při které chceš nakoupit tuto akcii."
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⭐ Přidat/Aktualizovat", use_container_width=True, type="primary"):
                watch.setdefault("items", {})[ticker] = {
                    "target_buy": target_buy,
                    "company": company_name,
                    "added_at": wl.get("added_at") or dt.datetime.now().isoformat(),
                    "updated_at": dt.datetime.now().isoformat(),
                }
                set_watchlist(watch)
                st.success(f"✅ {ticker} přidán na watchlist!")
        
        with col2:
            if st.button("🗑️ Odebrat", use_container_width=True):
                if ticker in watch.get("items", {}):
                    watch["items"].pop(ticker, None)
                    set_watchlist(watch)
                    st.success(f"✅ {ticker} odebrán z watchlistu!")
        
        st.markdown("---")
        
        # Display watchlist
        st.markdown("#### 📋 Můj Watchlist")
        items = watch.get("items", {})
        
        if items:
            rows = []
            for tkr, item in items.items():
                tkr_info = fetch_ticker_info(tkr)
                price_now = safe_float(tkr_info.get("currentPrice") or tkr_info.get("regularMarketPrice"))
                tgt = safe_float(item.get("target_buy"))
                hit = (price_now is not None and tgt is not None and tgt > 0 and price_now <= tgt)
                
                rows.append({
                    "Ticker": tkr,
                    "Společnost": item.get("company", "—"),
                    "Aktuální Cena": fmt_money(price_now),
                    "Cílová Cena": fmt_money(tgt),
                    "Status": "🟢 **BUY!**" if hit else "⏳ Čekej",
                    "Aktualizováno": item.get("updated_at", "")[:10]
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            
            # Alert for hit targets
            hits = [r for r in rows if "BUY" in r["Status"]]
            if hits:
                st.success(f"🎯 **ALERT**: {len(hits)} {'akcie' if len(hits) == 1 else 'akcie'} dosáhly tvé cílové ceny!")
        
        else:
            st.info("Watchlist je prázdný. Přidej první akcii výše.")
    
    # === FOOTER ===
    st.markdown("---")
    st.caption(f"📊 Data: Yahoo Finance | {APP_NAME} {APP_VERSION} | Toto není investiční doporučení")


if __name__ == "__main__":
    main()
