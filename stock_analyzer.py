"""
Stock Picker Pro v3.0 ULTIMATE
===============================
Robustní, dvojjazyčná (CZ/EN) aplikace optimalizovaná pro Gemini 2.5 Flash Lite (Free Tier)
s pokročilou finanční analýzou, sektorovou inteligencí a perfektním UX.

Author: Enhanced by Claude
Version: 3.0 ULTIMATE
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
    page_title="Stock Picker Pro v3.0",
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


# ============================================================================
# TRANSLATIONS DICTIONARY
# ============================================================================
TRANSLATIONS = {
    "en": {
        "app_name": "Stock Picker Pro",
        "language": "Language",
        "ticker_input": "Ticker Symbol",
        "ticker_placeholder": "e.g., AAPL, MSFT, TSLA",
        "analyze_button": "🔍 Analyze",
        "czech_stocks_help": "For Prague Stock Exchange use .PR suffix (e.g., CEZ.PR, KOMB.PR)",
        "overview": "📊 Overview",
        "scorecard": "💯 Scorecard",
        "valuation": "💰 Valuation",
        "financials": "💸 Financials",
        "technicals": "📈 Technicals",
        "insider": "👔 Insider Trading",
        "ai_analyst": "🤖 AI Analyst",
        "social": "🐦 Social & Guru",
        "price": "Price",
        "market_cap": "Market Cap",
        "pe_ratio": "P/E Ratio",
        "beta": "Beta",
        "dividend_yield": "Dividend Yield",
        "dcf_fair_value": "DCF Fair Value",
        "analyst_target": "Analyst Target",
        "total_score": "Total Score",
        "verdict": "Verdict",
        "strong_buy": "🟢 STRONG BUY",
        "buy": "🟢 BUY",
        "hold": "🟡 HOLD",
        "sell": "🔴 SELL",
        "strong_sell": "🔴 STRONG SELL",
        "value_trap": "⚠️ VALUE TRAP?",
        "generate_ai_report": "🚀 Generate AI Report",
        "ai_loading": "Generating AI analysis...",
        "ai_error": "⚠️ AI is overloaded (Rate Limit). Try again in a moment.",
        "ai_no_key": "⚠️ GEMINI_API_KEY not configured. Set it in Streamlit secrets or environment.",
        "welcome_title": "Welcome to Stock Picker Pro v3.0! 🚀",
        "how_to_start": "How to start:",
        "how_to_1": "⬅️ Enter ticker symbol in the left panel",
        "how_to_2": "Click on \"🔍 Analyze\"",
        "how_to_3": "Explore all tabs with advanced analytics",
        "popular_tickers": "💡 Popular tickers to try",
        "data_source": "Data: Yahoo Finance",
        "disclaimer": "This is not investment advice",
        "pe_help": "Price-to-Earnings ratio. Compares stock price to earnings. Lower is typically cheaper.",
        "beta_help": "Volatility measure. Beta > 1 = more volatile than market. Beta < 1 = less volatile.",
        "div_yield_help": "Annual dividend as % of stock price. Higher = more income for investors.",
        "dcf_help": "Discounted Cash Flow valuation. Intrinsic value based on future cash flows.",
        "wacc_help": "Weighted Average Cost of Capital. Risk measure. Higher WACC = lower fair value.",
        "roe_help": "Return on Equity. Shows how efficiently company uses shareholder money. Higher is better.",
        "roic_help": "Return on Invested Capital. Efficiency of capital deployment. >15% is excellent.",
        "debt_to_equity_help": "Debt/Equity ratio. <1 is conservative, >2 may be risky (sector-dependent).",
        "fcf_yield_help": "Free Cash Flow Yield. Annual FCF as % of market cap. Higher = better value.",
        "score_breakdown": "Score Breakdown",
        "valuation_score": "Valuation",
        "quality_score": "Quality",
        "growth_score": "Growth",
        "health_score": "Financial Health",
        "sector_detection": "Sector Detection",
        "crypto_detected": "🪙 Crypto detected - ignoring P/E, focusing on adoption & regulation",
        "bank_detected": "🏦 Bank/FinTech detected - focusing on interest rates & competition",
        "biotech_detected": "💊 BioTech detected - checking patent expiration risks",
        "value_trap_detected": "⚠️ Price down >50% from ATH - is this a discount or collapse?",
        "negative_fcf": "N/A (Negative Cash Flow)",
        "dcf_not_applicable": "N/A (DCF doesn't work for banks/negative FCF companies)",
        "mismatch_warning": "⚠️ Mismatch Warning",
        "analyst_vs_dcf": "Analysts see {analyst_upside:.0f}% upside but DCF shows {dcf_upside:.0f}%. Big divergence!",
    },
    "cz": {
        "app_name": "Stock Picker Pro",
        "language": "Jazyk",
        "ticker_input": "Ticker Symbol",
        "ticker_placeholder": "např. AAPL, MSFT, TSLA",
        "analyze_button": "🔍 Analyzovat",
        "czech_stocks_help": "Pro pražskou burzu použij .PR příponu (např. CEZ.PR, KOMB.PR)",
        "overview": "📊 Přehled",
        "scorecard": "💯 Hodnocení",
        "valuation": "💰 Valuace",
        "financials": "💸 Finance",
        "technicals": "📈 Technická Analýza",
        "insider": "👔 Insider Trading",
        "ai_analyst": "🤖 AI Analytik",
        "social": "🐦 Social & Guru",
        "price": "Cena",
        "market_cap": "Tržní Kapitalizace",
        "pe_ratio": "P/E Poměr",
        "beta": "Beta",
        "dividend_yield": "Dividendový Výnos",
        "dcf_fair_value": "DCF Férová Hodnota",
        "analyst_target": "Cíl Analytiků",
        "total_score": "Celkové Skóre",
        "verdict": "Verdikt",
        "strong_buy": "🟢 SILNÉ NÁKUP",
        "buy": "🟢 NÁKUP",
        "hold": "🟡 DRŽET",
        "sell": "🔴 PRODAT",
        "strong_sell": "🔴 SILNÉ PRODAT",
        "value_trap": "⚠️ VALUE TRAP?",
        "generate_ai_report": "🚀 Vygenerovat AI Report",
        "ai_loading": "Generuji AI analýzu...",
        "ai_error": "⚠️ AI je přetížená (Rate Limit). Zkuste to za chvíli.",
        "ai_no_key": "⚠️ GEMINI_API_KEY není nastaven. Nastav ho v Streamlit secrets nebo env proměnných.",
        "welcome_title": "Vítej v Stock Picker Pro v3.0! 🚀",
        "how_to_start": "Jak začít:",
        "how_to_1": "⬅️ Zadej ticker symbol v levém panelu",
        "how_to_2": "Klikni na \"🔍 Analyzovat\"",
        "how_to_3": "Prohlédni si všechny taby s pokročilými analýzami",
        "popular_tickers": "💡 Populární tickery na vyzkoušení",
        "data_source": "Data: Yahoo Finance",
        "disclaimer": "Toto není investiční doporučení",
        "pe_help": "Poměr Cena/Zisk. Porovnává cenu akcie k zisku. Nižší = obvykle levnější.",
        "beta_help": "Míra volatility. Beta > 1 = volatilnější než trh. Beta < 1 = stabilnější.",
        "div_yield_help": "Roční dividenda jako % z ceny akcie. Vyšší = více příjmů pro investory.",
        "dcf_help": "Diskontovaný Cash Flow. Vnitřní hodnota založená na budoucích peněžních tocích.",
        "wacc_help": "Vážený průměr nákladů kapitálu. Míra rizika. Vyšší WACC = nižší férová hodnota.",
        "roe_help": "Návratnost vlastního kapitálu. Ukazuje efektivitu využití peněz akcionářů. Vyšší je lepší.",
        "roic_help": "Návratnost investovaného kapitálu. Efektivita nasazení kapitálu. >15% je výborné.",
        "debt_to_equity_help": "Poměr Dluh/Vlastní kapitál. <1 je konzervativní, >2 může být rizikové (závisí na sektoru).",
        "fcf_yield_help": "Výnos volných peněžních toků. Roční FCF jako % tržní kapitalizace. Vyšší = lepší value.",
        "score_breakdown": "Rozpad Skóre",
        "valuation_score": "Valuace",
        "quality_score": "Kvalita",
        "growth_score": "Růst",
        "health_score": "Finanční Zdraví",
        "sector_detection": "Detekce Sektoru",
        "crypto_detected": "🪙 Krypto detekováno - ignoruji P/E, zaměřuji se na adopci & regulaci",
        "bank_detected": "🏦 Banka/FinTech detekována - zaměřuji se na úrokové sazby & konkurenci",
        "biotech_detected": "💊 BioTech detekován - kontroluji rizika expirace patentů",
        "value_trap_detected": "⚠️ Cena spadla >50% z ATH - je to sleva nebo krach?",
        "negative_fcf": "N/A (Záporný Cash Flow)",
        "dcf_not_applicable": "N/A (DCF nefunguje u bank/firem se záporným FCF)",
        "mismatch_warning": "⚠️ Varování o nesouladu",
        "analyst_vs_dcf": "Analytici vidí {analyst_upside:.0f}% růst, ale DCF ukazuje {dcf_upside:.0f}%. Velký rozdíl!",
    }
}


def t(key: str, lang: str = "cz", **kwargs) -> str:
    """Translate key to selected language with optional formatting."""
    text = TRANSLATIONS.get(lang, TRANSLATIONS["cz"]).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text


# ============================================================================
# SECRETS & CONFIG
# ============================================================================
def _get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return str(os.getenv(name, default) or default)

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
APP_VERSION = "v3.0 ULTIMATE"
GEMINI_MODEL = "gemini-2.5-flash-lite"  # Optimized for Free Tier
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


# ============================================================================
# SECTOR PEERS - Extended with Czech stocks
# ============================================================================
SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META"],
    "Financial Services": ["JPM", "BAC", "WFC", "C", "GS", "KOMB.PR", "MONETA.PR"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO"],
    "Consumer Cyclical": ["AMZN", "TSLA", "NKE", "HD", "MCD"],
    "Communication Services": ["META", "GOOGL", "DIS", "NFLX", "T"],
    "Utilities": ["NEE", "DUK", "SO", "D", "CEZ.PR"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Industrials": ["BA", "CAT", "HON", "UPS", "GE"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "Basic Materials": ["LIN", "APD", "ECL", "SHW", "NEM"],
}


# ============================================================================
# GURUS - Social handles
# ============================================================================
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
        "Macro Charts": "MacroCharts",
        "The Kobeissi Letter": "KobeissiLetter",
        "Bespoke": "bespokeinvest",
    },
}


# ============================================================================
# SECTOR DETECTION & CONTEXT
# ============================================================================
def detect_sector_context(ticker: str, info: dict, lang: str = "cz") -> Dict[str, Any]:
    """
    Detect sector-specific context for smarter analysis.
    Returns: {
        "is_crypto": bool,
        "is_bank": bool,
        "is_biotech": bool,
        "is_value_trap": bool,
        "context_message": str,
        "prompt_additions": str
    }
    """
    sector = (info.get("sector") or "").lower()
    industry = (info.get("industry") or "").lower()
    name = (info.get("longName") or "").lower()
    
    # Crypto detection
    is_crypto = (
        ticker.endswith("-USD") or 
        "crypto" in sector or 
        "crypto" in industry or
        "bitcoin" in name or 
        "ethereum" in name
    )
    
    # Bank/FinTech detection
    is_bank = (
        "bank" in sector or
        "financial" in sector or
        "fintech" in industry or
        "bank" in industry
    )
    
    # BioTech detection
    is_biotech = (
        "biotech" in sector or
        "biotech" in industry or
        "pharmaceutical" in industry
    )
    
    # Value Trap detection (price down >50% from ATH)
    current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    fifty_two_week_high = safe_float(info.get("fiftyTwoWeekHigh"))
    is_value_trap = False
    if current_price and fifty_two_week_high and fifty_two_week_high > 0:
        pct_from_high = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
        is_value_trap = pct_from_high < -50
    
    # Build context message
    messages = []
    if is_crypto:
        messages.append(t("crypto_detected", lang))
    if is_bank:
        messages.append(t("bank_detected", lang))
    if is_biotech:
        messages.append(t("biotech_detected", lang))
    if is_value_trap:
        messages.append(t("value_trap_detected", lang))
    
    context_message = "\n".join(messages) if messages else ""
    
    # Build prompt additions for AI
    prompt_additions = ""
    if is_crypto:
        prompt_additions += "\n- CRYPTO: Ignore P/E ratio. Focus on: adoption metrics, regulatory risks, blockchain fundamentals."
    if is_bank:
        prompt_additions += "\n- BANK/FINTECH: Focus on: interest rate sensitivity, competition from Apple Pay/neobanks, loan quality."
    if is_biotech:
        prompt_additions += "\n- BIOTECH: Check: patent expiration dates, FDA approval pipeline, R&D efficiency."
    if is_value_trap:
        prompt_additions += "\n- VALUE TRAP WARNING: Price down >50% from ATH. Question: Is this a discount or permanent impairment?"
    
    return {
        "is_crypto": is_crypto,
        "is_bank": is_bank,
        "is_biotech": is_biotech,
        "is_value_trap": is_value_trap,
        "context_message": context_message,
        "prompt_additions": prompt_additions
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_float(val: Any, default: float = 0.0) -> Optional[float]:
    """Safely convert value to float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return float(val)
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def fmt_money(val: Optional[float], lang: str = "cz") -> str:
    """Format money values."""
    if val is None:
        return "—"
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.2f}"


def fmt_pct(val: Optional[float], decimals: int = 2) -> str:
    """Format percentage values."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}%"


@st.cache_data(ttl=3600)
def fetch_ticker_info(ticker: str) -> dict:
    """Fetch ticker info from yfinance with caching."""
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def fetch_ticker_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch price history with caching."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ============================================================================
# DCF VALUATION
# ============================================================================
def calculate_dcf_fair_value(info: dict, sector_context: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate DCF fair value with sector-specific adjustments.
    Returns: (fair_value, wacc)
    """
    # Don't calculate DCF for banks or negative FCF companies
    if sector_context.get("is_bank"):
        return None, None
    
    fcf = safe_float(info.get("freeCashflow"))
    if not fcf or fcf <= 0:
        return None, None
    
    shares = safe_float(info.get("sharesOutstanding"))
    if not shares or shares <= 0:
        return None, None
    
    beta = safe_float(info.get("beta")) or 1.0
    risk_free = 0.04  # 4% risk-free rate
    market_premium = 0.08  # 8% market risk premium
    cost_of_equity = risk_free + beta * market_premium
    
    debt = safe_float(info.get("totalDebt")) or 0
    equity = safe_float(info.get("marketCap")) or 1
    total_cap = debt + equity
    
    cost_of_debt = 0.05  # Assume 5% cost of debt
    tax_rate = 0.21  # 21% corporate tax
    
    if total_cap > 0:
        wacc = (equity/total_cap * cost_of_equity) + (debt/total_cap * cost_of_debt * (1 - tax_rate))
    else:
        wacc = cost_of_equity
    
    wacc = max(0.06, min(wacc, 0.20))  # Cap WACC between 6-20%
    
    # Project 5-year FCF with conservative growth
    growth = min(safe_float(info.get("revenueGrowth")) or 0.10, 0.15)
    terminal_growth = 0.025  # 2.5% perpetual growth
    
    pv_fcf = 0
    for year in range(1, 6):
        future_fcf = fcf * ((1 + growth) ** year)
        discount_factor = (1 + wacc) ** year
        pv_fcf += future_fcf / discount_factor
    
    terminal_fcf = fcf * ((1 + growth) ** 5) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** 5)
    
    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - debt
    fair_value_per_share = equity_value / shares
    
    return fair_value_per_share, wacc


# ============================================================================
# SCORECARD CALCULATION
# ============================================================================
def calculate_scorecard(info: dict, fair_value_dcf: Optional[float], sector_context: dict) -> Tuple[float, dict]:
    """
    Calculate 0-100 scorecard with breakdown.
    Returns: (total_score, breakdown_dict)
    """
    scores = {
        "valuation": 0,
        "quality": 0,
        "growth": 0,
        "health": 0
    }
    
    current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    
    # VALUATION SCORE (0-25)
    val_score = 0
    if fair_value_dcf and current_price and not sector_context.get("is_bank"):
        upside = ((fair_value_dcf - current_price) / current_price) * 100
        if upside > 50:
            val_score = 25
        elif upside > 30:
            val_score = 20
        elif upside > 10:
            val_score = 15
        elif upside > -10:
            val_score = 10
        else:
            val_score = 5
    else:
        # Alternative valuation for banks/crypto
        pe = safe_float(info.get("trailingPE"))
        if pe and pe > 0 and not sector_context.get("is_crypto"):
            if pe < 15:
                val_score = 20
            elif pe < 20:
                val_score = 15
            elif pe < 25:
                val_score = 10
            else:
                val_score = 5
    
    scores["valuation"] = val_score
    
    # QUALITY SCORE (0-25)
    roe = safe_float(info.get("returnOnEquity"))
    profit_margin = safe_float(info.get("profitMargins"))
    
    qual_score = 0
    if roe:
        if roe > 0.20:
            qual_score += 15
        elif roe > 0.15:
            qual_score += 10
        elif roe > 0.10:
            qual_score += 5
    
    if profit_margin:
        if profit_margin > 0.20:
            qual_score += 10
        elif profit_margin > 0.10:
            qual_score += 5
    
    scores["quality"] = min(qual_score, 25)
    
    # GROWTH SCORE (0-25)
    rev_growth = safe_float(info.get("revenueGrowth"))
    earnings_growth = safe_float(info.get("earningsGrowth"))
    
    growth_score = 0
    if rev_growth:
        if rev_growth > 0.20:
            growth_score += 15
        elif rev_growth > 0.10:
            growth_score += 10
        elif rev_growth > 0:
            growth_score += 5
    
    if earnings_growth:
        if earnings_growth > 0.20:
            growth_score += 10
        elif earnings_growth > 0.10:
            growth_score += 5
    
    scores["growth"] = min(growth_score, 25)
    
    # HEALTH SCORE (0-25)
    current_ratio = safe_float(info.get("currentRatio"))
    debt_to_equity = safe_float(info.get("debtToEquity"))
    
    health_score = 0
    if current_ratio:
        if current_ratio > 2.0:
            health_score += 15
        elif current_ratio > 1.5:
            health_score += 10
        elif current_ratio > 1.0:
            health_score += 5
    
    if debt_to_equity is not None:
        if debt_to_equity < 50:
            health_score += 10
        elif debt_to_equity < 100:
            health_score += 5
    
    scores["health"] = min(health_score, 25)
    
    total = sum(scores.values())
    return total, scores


def get_verdict(score: float, lang: str = "cz") -> str:
    """Get verdict based on score."""
    if score >= 80:
        return t("strong_buy", lang)
    elif score >= 65:
        return t("buy", lang)
    elif score >= 45:
        return t("hold", lang)
    elif score >= 30:
        return t("sell", lang)
    else:
        return t("strong_sell", lang)


# ============================================================================
# AI ANALYST with RETRY LOGIC
# ============================================================================
def generate_ai_analyst_report_with_retry(
    ticker: str,
    info: dict,
    sector_context: dict,
    fair_value_dcf: Optional[float],
    current_price: Optional[float],
    lang: str = "cz"
) -> str:
    """
    Generate AI report with retry logic for Free Tier rate limits.
    """
    if not GEMINI_API_KEY:
        return t("ai_no_key", lang)
    
    for attempt in range(MAX_RETRIES):
        try:
            result = _generate_ai_report(ticker, info, sector_context, fair_value_dcf, current_price, lang)
            return result
        except Exception as e:
            error_msg = str(e)
            # Check for rate limit errors
            if "429" in error_msg or "quota" in error_msg.lower() or "rate" in error_msg.lower():
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return t("ai_error", lang)
            else:
                # Other errors - don't retry
                return f"⚠️ AI Error: {error_msg}"
    
    return t("ai_error", lang)


def _generate_ai_report(
    ticker: str,
    info: dict,
    sector_context: dict,
    fair_value_dcf: Optional[float],
    current_price: Optional[float],
    lang: str
) -> str:
    """Internal function to generate AI report using Gemini."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        return f"⚠️ Gemini import error: {e}"
    
    company = info.get("longName", ticker)
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    
    # Build hardcore prompt with persona
    prompt_lang = "Czech" if lang == "cz" else "English"
    
    prompt = f"""You are a cynical hedge fund manager analyzing {ticker} ({company}).

CRITICAL INSTRUCTIONS:
- Write in {prompt_lang} language
- DO NOT repeat marketing phrases or company slogans
- Your job is to find reasons NOT to buy
- Look for RISKS and red flags
- Be brutally honest about valuation
- Question every bullish narrative

SECTOR CONTEXT:{sector_context.get('prompt_additions', '')}

COMPANY DATA:
- Sector: {sector}
- Industry: {industry}
- Current Price: ${current_price or 'N/A'}
- DCF Fair Value: ${fair_value_dcf or 'N/A'}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- Revenue Growth: {info.get('revenueGrowth', 'N/A')}
- Profit Margin: {info.get('profitMargins', 'N/A')}
- Debt/Equity: {info.get('debtToEquity', 'N/A')}
- ROE: {info.get('returnOnEquity', 'N/A')}

ANALYSIS STRUCTURE:
1. **Investment Thesis** (2-3 sentences - what's the bull case?)
2. **Key Risks** (3-5 bullet points - what could go wrong?)
3. **Valuation Reality Check** (Is the price justified? Be skeptical!)
4. **Specific Wait-For Price** (Give ONE specific entry price with rationale)
5. **Bear Case Scenario** (What if everything goes wrong?)

Be concise. No fluff. Show me why I should be cautious."""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        raise e  # Re-raise to trigger retry logic


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================
def init_session_state():
    """Initialize session state for tab persistence."""
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = 0
    if "ai_report" not in st.session_state:
        st.session_state.ai_report = None
    if "ai_generated_for_ticker" not in st.session_state:
        st.session_state.ai_generated_for_ticker = None
    if "language" not in st.session_state:
        st.session_state.language = "cz"


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"# 📈 {t('app_name', st.session_state.language)}")
        st.markdown(f"**{APP_VERSION}**")
        st.markdown("---")
        
        # Language selector
        lang_options = {"🇨🇿 Čeština": "cz", "🇺🇸 English": "en"}
        selected_lang_label = st.selectbox(
            t("language", st.session_state.language),
            options=list(lang_options.keys()),
            index=0 if st.session_state.language == "cz" else 1
        )
        st.session_state.language = lang_options[selected_lang_label]
        lang = st.session_state.language
        
        st.markdown("---")
        
        # Ticker input
        ticker_input = st.text_input(
            t("ticker_input", lang),
            value=st.session_state.get("last_ticker", ""),
            placeholder=t("ticker_placeholder", lang),
            help=t("czech_stocks_help", lang)
        )
        
        analyze_btn = st.button(t("analyze_button", lang), use_container_width=True, type="primary")
        
        if analyze_btn and ticker_input:
            st.session_state["last_ticker"] = ticker_input.strip().upper()
            st.session_state.active_tab = 0  # Reset to first tab
            st.session_state.ai_report = None  # Clear old AI report
            st.session_state.ai_generated_for_ticker = None
            st.rerun()
    
    # Main content
    ticker = st.session_state.get("last_ticker", "").strip().upper()
    
    if not ticker:
        display_welcome_screen(lang)
        return
    
    # Fetch data
    with st.spinner(f"Loading {ticker}..."):
        info = fetch_ticker_info(ticker)
        history = fetch_ticker_history(ticker, period="1y")
    
    if not info or not info.get("longName"):
        st.error(f"❌ Could not fetch data for {ticker}. Check ticker symbol.")
        return
    
    # Sector context detection
    sector_context = detect_sector_context(ticker, info, lang)
    
    # Calculate metrics
    current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    fair_value_dcf, wacc = calculate_dcf_fair_value(info, sector_context)
    scorecard, score_breakdown = calculate_scorecard(info, fair_value_dcf, sector_context)
    verdict = get_verdict(scorecard, lang)
    
    # Header
    company = info.get("longName", ticker)
    st.title(f"{ticker} - {company}")
    
    # Display sector context warnings
    if sector_context["context_message"]:
        st.info(sector_context["context_message"])
    
    # Smart Header - 5 Key Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            t("price", lang),
            fmt_money(current_price, lang),
            help=t("pe_help", lang)
        )
    
    with col2:
        if fair_value_dcf and not sector_context.get("is_bank"):
            dcf_display = fmt_money(fair_value_dcf, lang)
        else:
            dcf_display = t("dcf_not_applicable", lang)
        st.metric(
            t("dcf_fair_value", lang),
            dcf_display,
            help=t("dcf_help", lang)
        )
    
    with col3:
        target_price = safe_float(info.get("targetMeanPrice"))
        st.metric(
            t("analyst_target", lang),
            fmt_money(target_price, lang),
            help=t("wacc_help", lang)
        )
    
    with col4:
        st.metric(
            t("total_score", lang),
            f"{scorecard:.0f}/100",
            help=t("score_breakdown", lang)
        )
    
    with col5:
        st.metric(
            t("verdict", lang),
            verdict.split()[1] if len(verdict.split()) > 1 else verdict
        )
    
    # Mismatch Warning
    if fair_value_dcf and target_price and current_price:
        dcf_upside = ((fair_value_dcf - current_price) / current_price) * 100
        analyst_upside = ((target_price - current_price) / current_price) * 100
        
        if abs(dcf_upside - analyst_upside) > 20:
            st.warning(
                f"⚠️ {t('mismatch_warning', lang)}: " +
                t("analyst_vs_dcf", lang, analyst_upside=analyst_upside, dcf_upside=dcf_upside)
            )
    
    st.markdown("---")
    
    # Tabs
    tab_labels = [
        t("overview", lang),
        t("scorecard", lang),
        t("valuation", lang),
        t("financials", lang),
        t("technicals", lang),
        t("insider", lang),
        t("ai_analyst", lang),
        t("social", lang)
    ]
    
    tabs = st.tabs(tab_labels)
    
    # TAB 0: Overview
    with tabs[0]:
        st.markdown("### 📊 " + t("overview", lang))
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                t("market_cap", lang),
                fmt_money(safe_float(info.get("marketCap")), lang)
            )
        
        with col2:
            pe = safe_float(info.get("trailingPE"))
            st.metric(
                t("pe_ratio", lang),
                f"{pe:.2f}" if pe else "—",
                help=t("pe_help", lang)
            )
        
        with col3:
            beta = safe_float(info.get("beta"))
            st.metric(
                t("beta", lang),
                f"{beta:.2f}" if beta else "—",
                help=t("beta_help", lang)
            )
        
        with col4:
            div_yield = safe_float(info.get("dividendYield"))
            st.metric(
                t("dividend_yield", lang),
                fmt_pct(div_yield * 100) if div_yield else "—",
                help=t("div_yield_help", lang)
            )
        
        # Price chart
        if not history.empty:
            st.line_chart(history["Close"])
    
    # TAB 1: Scorecard
    with tabs[1]:
        st.markdown("### 💯 " + t("scorecard", lang))
        
        st.markdown(f"**{t('total_score', lang)}:** {scorecard:.0f}/100")
        st.markdown(f"**{t('verdict', lang)}:** {verdict}")
        
        st.markdown("---")
        st.markdown(f"#### {t('score_breakdown', lang)}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(t("valuation_score", lang), f"{score_breakdown['valuation']:.0f}/25")
        
        with col2:
            st.metric(t("quality_score", lang), f"{score_breakdown['quality']:.0f}/25")
        
        with col3:
            st.metric(t("growth_score", lang), f"{score_breakdown['growth']:.0f}/25")
        
        with col4:
            st.metric(t("health_score", lang), f"{score_breakdown['health']:.0f}/25")
        
        # Progress bars
        for key, value in score_breakdown.items():
            st.progress(value / 25, text=f"{t(key + '_score', lang)}: {value:.0f}/25")
    
    # TAB 2: Valuation
    with tabs[2]:
        st.markdown("### 💰 " + t("valuation", lang))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### DCF Valuation")
            if fair_value_dcf and not sector_context.get("is_bank"):
                st.metric("Fair Value", fmt_money(fair_value_dcf, lang), help=t("dcf_help", lang))
                st.metric("WACC", fmt_pct(wacc * 100) if wacc else "—", help=t("wacc_help", lang))
                
                if current_price:
                    upside = ((fair_value_dcf - current_price) / current_price) * 100
                    st.metric("Upside/Downside", fmt_pct(upside))
            else:
                st.info(t("dcf_not_applicable", lang))
        
        with col2:
            st.markdown("#### Multiples")
            pe = safe_float(info.get("trailingPE"))
            pb = safe_float(info.get("priceToBook"))
            ps = safe_float(info.get("priceToSalesTrailing12Months"))
            
            if pe:
                st.metric("P/E", f"{pe:.2f}", help=t("pe_help", lang))
            if pb:
                st.metric("P/B", f"{pb:.2f}")
            if ps:
                st.metric("P/S", f"{ps:.2f}")
    
    # TAB 3: Financials
    with tabs[3]:
        st.markdown("### 💸 " + t("financials", lang))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Profitability")
            roe = safe_float(info.get("returnOnEquity"))
            roic = safe_float(info.get("returnOnAssets"))  # Approximation
            margin = safe_float(info.get("profitMargins"))
            
            if roe:
                st.metric("ROE", fmt_pct(roe * 100), help=t("roe_help", lang))
            if roic:
                st.metric("ROIC (approx)", fmt_pct(roic * 100), help=t("roic_help", lang))
            if margin:
                st.metric("Profit Margin", fmt_pct(margin * 100))
        
        with col2:
            st.markdown("#### Financial Health")
            debt_equity = safe_float(info.get("debtToEquity"))
            current_ratio = safe_float(info.get("currentRatio"))
            fcf = safe_float(info.get("freeCashflow"))
            mcap = safe_float(info.get("marketCap"))
            
            if debt_equity is not None:
                st.metric("Debt/Equity", f"{debt_equity:.2f}%", help=t("debt_to_equity_help", lang))
            if current_ratio:
                st.metric("Current Ratio", f"{current_ratio:.2f}")
            
            if fcf and mcap and fcf > 0:
                fcf_yield = (fcf / mcap) * 100
                st.metric("FCF Yield", fmt_pct(fcf_yield), help=t("fcf_yield_help", lang))
            elif fcf and fcf < 0:
                st.metric("FCF", t("negative_fcf", lang), help=t("fcf_yield_help", lang))
    
    # TAB 4: Technicals (placeholder)
    with tabs[4]:
        st.markdown("### 📈 " + t("technicals", lang))
        st.info("Technical analysis coming soon!")
    
    # TAB 5: Insider (placeholder)
    with tabs[5]:
        st.markdown("### 👔 " + t("insider", lang))
        st.info("Insider trading analysis coming soon!")
    
    # TAB 6: AI Analyst
    with tabs[6]:
        st.markdown("### 🤖 " + t("ai_analyst", lang))
        
        # Check if report already exists for this ticker
        if st.session_state.ai_generated_for_ticker != ticker:
            st.session_state.ai_report = None
        
        # Generate button
        generate_col1, generate_col2 = st.columns([1, 3])
        with generate_col1:
            generate_btn = st.button(
                t("generate_ai_report", lang),
                use_container_width=True,
                type="primary"
            )
        
        with generate_col2:
            st.caption("Uses Gemini 2.5 Flash Lite with retry logic for rate limits")
        
        # Generate AI report
        if generate_btn:
            with st.spinner(t("ai_loading", lang)):
                report = generate_ai_analyst_report_with_retry(
                    ticker, info, sector_context, fair_value_dcf, current_price, lang
                )
                st.session_state.ai_report = report
                st.session_state.ai_generated_for_ticker = ticker
        
        # Display AI report if exists
        if st.session_state.ai_report:
            st.markdown(st.session_state.ai_report)
    
    # TAB 7: Social & Guru (placeholder)
    with tabs[7]:
        st.markdown("### 🐦 " + t("social", lang))
        st.info("Social sentiment analysis coming soon!")
    
    # Footer
    st.markdown("---")
    st.caption(f"📊 {t('data_source', lang)} | {APP_NAME} {APP_VERSION} | {t('disclaimer', lang)}")


def display_welcome_screen(lang: str):
    """Display welcome screen when no ticker is selected."""
    st.title(t("welcome_title", lang))
    
    st.markdown(f"""
    ### {t("how_to_start", lang)}
    
    1. {t("how_to_1", lang)}
    2. {t("how_to_2", lang)}
    3. {t("how_to_3", lang)}
    
    **🆕 New in v3.0 ULTIMATE:**
    - ✅ **Bilingual Support (CZ/EN)** - Full app translation
    - ✅ **Gemini 2.5 Flash Lite** - Optimized with retry logic for Free Tier
    - ✅ **Sector Intelligence** - Crypto, Bank, BioTech detection
    - ✅ **Value Trap Warning** - Alerts when price down >50% from ATH
    - ✅ **Fixed Tab State** - Tabs no longer reset after AI generation
    - ✅ **Czech Stocks Support** - Extended with .PR suffix tips
    - ✅ **Hardcore AI Prompts** - Cynical analyst persona for critical thinking
    - ✅ **Enhanced Tooltips** - Every metric explained
    """)
    
    # Sample tickers
    st.markdown(f"### {t('popular_tickers', lang)}")
    cols = st.columns(4)
    samples = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
    
    for i, ticker in enumerate(samples):
        with cols[i % 4]:
            if st.button(ticker, use_container_width=True, key=f"sample_{ticker}"):
                st.session_state["last_ticker"] = ticker
                st.rerun()
    
    st.markdown("---")
    st.info("💡 **For AI Analysis**: Set GEMINI_API_KEY in Streamlit secrets or environment variables!")


if __name__ == "__main__":
    main()
