"""
Stock Picker Pro v2.0 - Hlavní Streamlit aplikace.

Modernizovaná architektura s:
- Modulární data layer (YFinance/FMP/Intrinio)
- Pokročilé analytické modely (Piotroski, Altman, Beneish)
- Multi-stage DCF
- AI sentiment analysis
- Decision engine s vizuálním semaforem
- Football field chart
"""
import streamlit as st
import logging
from datetime import datetime
from typing import Optional

# Import modulů
from config import config, api_keys, scoring_weights
from utils import setup_logging, DataValidator, format_money, format_percent, format_number
from data_providers import DataProvider, DataProviderError
from yfinance_provider import YFinanceProvider
from quality_scores import calculate_all_quality_scores
from valuation import DCFCalculator, calculate_historical_fcf_cagr, estimate_wacc
from sentiment import SentimentAnalyzer
from decision_engine import DecisionEngine, Decision
from football_field import FootballFieldChart


# Setup
setup_logging("INFO")
logger = logging.getLogger(__name__)


# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title=f"{config.APP_NAME} {config.APP_VERSION}",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Complete CSS - All styles inline (no external files needed)
st.markdown("""
    <style>
    /* =================================================================
       CRITICAL FIX: Solid sidebar background (not transparent)
       ================================================================= */
    
    /* Force solid background on ALL sidebar elements */
    section[data-testid="stSidebar"] {
        background-color: #262730 !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #262730 !important;
        opacity: 1 !important;
    }
    
    section[data-testid="stSidebar"] > div > div {
        background-color: #262730 !important;
        opacity: 1 !important;
    }
    
    /* Light theme support */
    @media (prefers-color-scheme: light) {
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        section[data-testid="stSidebar"] > div > div {
            background-color: #FFFFFF !important;
        }
    }
    
    /* =================================================================
       MOBILE OPTIMIZATIONS (screens < 768px)
       ================================================================= */
    @media (max-width: 768px) {
        /* Sidebar as modal overlay when open */
        section[data-testid="stSidebar"][aria-expanded="true"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 85% !important;
            max-width: 350px !important;
            height: 100vh !important;
            background-color: #262730 !important;
            z-index: 999999 !important;
            box-shadow: 4px 0 20px rgba(0, 0, 0, 0.8) !important;
            overflow-y: auto !important;
        }
        
        /* Dim main content when sidebar is open */
        section[data-testid="stSidebar"][aria-expanded="true"] ~ div {
            opacity: 0.3 !important;
            pointer-events: none !important;
        }
        
        /* Form in sidebar - clearly visible */
        section[data-testid="stSidebar"] [data-testid="stForm"] {
            background-color: rgba(255, 255, 255, 0.08) !important;
            border: 2px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
            padding: 1.5rem !important;
            margin: 1rem 0 !important;
        }
        
        /* Input fields in sidebar */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] textarea {
            background-color: rgba(255, 255, 255, 0.12) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            color: white !important;
        }
        
        /* Submit button prominent */
        section[data-testid="stSidebar"] button[kind="primary"] {
            background-color: #FF4B4B !important;
            color: white !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4) !important;
        }
        
        /* Compact main content padding */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            max-width: 100% !important;
        }
        
        /* Compact metrics */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        /* Scrollable tabs */
        [data-baseweb="tab-list"] {
            overflow-x: auto !important;
            overflow-y: hidden !important;
            flex-wrap: nowrap !important;
            -webkit-overflow-scrolling: touch !important;
        }
        
        /* Touch-friendly buttons */
        button {
            min-height: 44px !important;
            padding: 0.75rem 1.5rem !important;
        }
        
        .stButton button {
            width: 100%;
            font-size: 1rem;
            padding: 0.75rem;
        }
    }
    
    /* =================================================================
       GENERAL UI IMPROVEMENTS
       ================================================================= */
    
    /* Decision card styling */
    .decision-card {
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Expander headers */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    
    /* Better hover states */
    button:hover {
        transform: translateY(-1px);
        transition: transform 0.2s;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if "ticker" not in st.session_state:
    st.session_state.ticker = "MSFT"
if "data_provider" not in st.session_state:
    st.session_state.data_provider = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def get_data_provider() -> DataProvider:
    """Získá data provider (s cachováním)"""
    # V budoucnu: Fallback chain FMP → Intrinio → YFinance
    # Pro teď: jen YFinance
    return YFinanceProvider()


# ============================================================================
# WATCHLIST & MEMO FUNCTIONS
# ============================================================================

def get_watchlist() -> dict:
    """Načte watchlist z JSON"""
    import os
    import json
    data_dir = os.path.join(os.path.expanduser("~"), config.DATA_DIR)
    watchlist_path = os.path.join(data_dir, "watchlist.json")
    
    if os.path.exists(watchlist_path):
        try:
            with open(watchlist_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    
    return {"items": {}}


def set_watchlist(data: dict):
    """Uloží watchlist do JSON"""
    import os
    import json
    data_dir = os.path.join(os.path.expanduser("~"), config.DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    watchlist_path = os.path.join(data_dir, "watchlist.json")
    
    with open(watchlist_path, "w") as f:
        json.dump(data, f, indent=2)


def get_memos() -> dict:
    """Načte memos z JSON"""
    import os
    import json
    data_dir = os.path.join(os.path.expanduser("~"), config.DATA_DIR)
    memos_path = os.path.join(data_dir, "memos.json")
    
    if os.path.exists(memos_path):
        try:
            with open(memos_path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    
    return {"memos": {}}


def set_memos(data: dict):
    """Uloží memos do JSON"""
    import os
    import json
    data_dir = os.path.join(os.path.expanduser("~"), config.DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    memos_path = os.path.join(data_dir, "memos.json")
    
    with open(memos_path, "w") as f:
        json.dump(data, f, indent=2)


def display_decision_card(signal):
    """Zobrazí vizuální decision card s semaforem - mobile-friendly"""
    
    # Responzivní layout: 1 sloupec na mobilu, 3 na desktopu
    st.markdown(
        f"""
        <div class="decision-card" style="
            padding: 20px;
            border-radius: 16px;
            background: linear-gradient(135deg, {signal.color}22 0%, {signal.color}11 100%);
            border: 3px solid {signal.color};
            text-align: center;
            margin-bottom: 20px;
        ">
            <div style="font-size: 48px; margin-bottom: 10px;">{signal.emoji}</div>
            <div style="font-size: 28px; font-weight: 800; color: {signal.color}; margin-bottom: 10px;">
                {signal.decision.value}
            </div>
            <div style="font-size: 16px; opacity: 0.8; margin-bottom: 15px;">
                Quality Grade: <strong>{signal.quality_grade}</strong> | 
                Confidence: <strong>{signal.confidence*100:.0f}%</strong>
            </div>
            <div style="font-size: 14px; opacity: 0.7;">
                Score: <strong>{signal.final_score:.1f}/100</strong>
                {f' | MOS: <strong>{signal.margin_of_safety*100:.1f}%</strong>' if signal.margin_of_safety is not None else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Component scores jako expander (šetří místo na mobilu)
    with st.expander("📊 Rozpad skóre podle komponent"):
        for component, score in signal.component_scores.items():
            # Progress bar style
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(component.replace("_", " ").title())
            with col2:
                st.caption(f"**{score:.0f}**")
            st.progress(min(score / 100, 1.0))


def display_quality_scores(quality_results):
    """Zobrazí quality scores v přehledné formě"""
    st.subheader("📊 Quality Scores")
    
    col1, col2, col3 = st.columns(3)
    
    # Piotroski F-Score
    with col1:
        piotroski = quality_results.get("piotroski")
        if piotroski:
            st.metric(
                "Piotroski F-Score",
                f"{piotroski.score}/9",
                help="9-bodový systém finančního zdraví (0=nejhorší, 9=nejlepší)"
            )
            st.caption(piotroski.interpretation)
            
            # Detail breakdown
            with st.expander("Detail kritérií"):
                st.write(f"✅ Profitability: {piotroski.profitability}/4")
                st.write(f"✅ Leverage/Liquidity: {piotroski.leverage_liquidity}/3")
                st.write(f"✅ Operating Efficiency: {piotroski.operating_efficiency}/2")
                for key, val in piotroski.details.items():
                    icon = "✅" if val else "❌"
                    st.caption(f"{icon} {key.replace('_', ' ').title()}")
        else:
            st.info("Nedostatek dat pro F-Score")
    
    # Altman Z-Score
    with col2:
        altman = quality_results.get("altman")
        if altman:
            color_map = {"Safe": "green", "Grey": "orange", "Distress": "red"}
            st.metric(
                "Altman Z-Score",
                f"{altman.score:.2f}",
                help="Predikce rizika bankrotu"
            )
            st.markdown(f"**Zone:** :{color_map[altman.zone]}[{altman.zone}]")
            st.caption(f"Bankruptcy Risk: {altman.bankruptcy_risk}")
            
            with st.expander("Detail komponent"):
                for key, val in altman.components.items():
                    if val is not None:
                        st.caption(f"{key}: {val:.3f}")
        else:
            st.info("Nedostatek dat pro Z-Score")
    
    # Beneish M-Score
    with col3:
        beneish = quality_results.get("beneish")
        if beneish:
            st.metric(
                "Beneish M-Score",
                f"{beneish.score:.2f}",
                help="Detekce účetní manipulace (> -2.22 = riziko)"
            )
            st.caption(f"Manipulation: {beneish.likelihood}")
            st.caption(f"Risk: {beneish.manipulation_risk}")
            
            with st.expander("Detail proměnných"):
                for key, val in beneish.variables.items():
                    if val is not None:
                        st.caption(f"{key.upper()}: {val:.3f}")
        else:
            st.info("Nedostatek dat pro M-Score")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Hlavní aplikace"""
    
    # Header
    st.title(f"📊 {config.APP_NAME} {config.APP_VERSION}")
    st.markdown("**Profesionální analýza akcií s AI a pokročilými valuačními modely**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Nastavení")
        
        # Ticker input with form for Enter support
        with st.form("ticker_form", clear_on_submit=False):
            ticker_input = st.text_input(
                "Stock Ticker",
                value=st.session_state.ticker,
                help="Např. MSFT, AAPL, GOOGL",
                key="ticker_input"
            ).upper().strip()
            
            submitted = st.form_submit_button("🔍 Analyzovat", use_container_width=True)
            
            if submitted and ticker_input and ticker_input != st.session_state.ticker:
                st.session_state.ticker = ticker_input
                st.rerun()
        
        ticker = st.session_state.ticker
        
        st.markdown("---")
        
        # DCF Parameters - collapsible on mobile
        with st.expander("⚙️ DCF Parametry", expanded=False):
            wacc = st.slider("WACC (%)", 4.0, 15.0, 8.0, 0.5) / 100
            terminal_growth = st.slider("Terminal Growth (%)", 1.0, 4.0, 2.5, 0.1) / 100
            projection_years = st.number_input("Projekční roky", 5, 15, 10)
        
        st.markdown("---")
        
        # Feature toggles
        st.subheader("Features")
        use_three_stage_dcf = st.checkbox("3-Stage DCF", value=True, help="Přesnější pro growth akcie")
        use_ai_sentiment = st.checkbox("AI Sentiment", value=api_keys.has_ai_sentiment, 
                                       disabled=not api_keys.has_ai_sentiment)
        show_football_field = st.checkbox("Football Field Chart", value=True)
        
        st.markdown("---")
        st.caption(f"Data provider: {get_data_provider().get_provider_name()}")
    
    # Main content
    if not ticker:
        st.info("👈 Zadejte ticker symbol v sidebar")
        return
    
    # Loading data
    with st.spinner(f"Načítám data pro {ticker}..."):
        try:
            provider = get_data_provider()
            
            # Fetch all data
            company_info = provider.get_company_info(ticker)
            price_data = provider.get_price_data(ticker)
            metrics = provider.get_financial_metrics(ticker)
            statements = provider.get_financial_statements(ticker, "annual", limit=5)
            analyst_estimates = provider.get_analyst_estimates(ticker)
            insider_txns = provider.get_insider_transactions(ticker, limit=100)
            news = provider.get_news(ticker, limit=10)
            
            if not company_info or not price_data:
                st.error(f"❌ Nepodařilo se načíst data pro {ticker}")
                return
            
        except DataProviderError as e:
            st.error(f"❌ Chyba při načítání dat: {e}")
            return
        except Exception as e:
            st.error(f"❌ Neočekávaná chyba: {e}")
            logger.exception("Unexpected error")
            return
    
    # Validate data
    _, price_warnings = DataValidator.validate_price_data(price_data)
    _, stmt_warnings = DataValidator.validate_financial_statements(statements)
    
    if price_warnings:
        st.warning("⚠️ Varování u cenových dat: " + ", ".join(price_warnings))
    if stmt_warnings:
        st.warning("⚠️ Varování u finančních dat: " + ", ".join(stmt_warnings))
    
    # Company Header - responsive
    st.header(f"{company_info.name} ({ticker})")
    
    # Na mobilu: 2x2 grid, na desktopu: 4x1
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Price", format_money(price_data.current_price))
        st.metric("52W High", format_money(price_data.fifty_two_week_high))
    with col2:
        market_cap_display = (
            format_money(company_info.market_cap / 1e9, decimals=2) + "B" 
            if company_info.market_cap else "—"
        )
        st.metric("Market Cap", market_cap_display)
        st.metric("52W Low", format_money(price_data.fifty_two_week_low))
    
    st.caption(f"**Sector:** {company_info.sector} | **Industry:** {company_info.industry}")
    
    # Tabs
    tabs = st.tabs([
        "🎯 Decision Dashboard",
        "💰 Valuation",
        "📊 Quality Scores",
        "📈 Sentiment",
        "📰 News & Fundamentals",
        "📝 Memo & Watchlist"
    ])
    
    # --- TAB 1: DECISION DASHBOARD ---
    with tabs[0]:
        st.header("🎯 Investment Decision")
        
        with st.spinner("Analyzuji..."):
            # Calculate all scores
            quality_results = calculate_all_quality_scores(
                statements, metrics, company_info.market_cap
            )
            
            # DCF
            dcf_result = None
            reverse_dcf = None
            if statements and statements[0].free_cash_flow:
                base_fcf = statements[0].free_cash_flow
                
                calculator = DCFCalculator()
                
                if use_three_stage_dcf:
                    # Three-stage DCF
                    dcf_result = calculator.calculate_three_stage_dcf(
                        base_fcf=base_fcf,
                        high_growth_rate=0.15,  # 15% high growth
                        high_growth_years=5,
                        transition_growth_rate=0.08,
                        transition_years=5,
                        wacc=wacc,
                        terminal_growth=terminal_growth,
                        shares_outstanding=int(company_info.market_cap / price_data.current_price) if company_info.market_cap and price_data.current_price else None,
                        net_debt=statements[0].total_debt - statements[0].cash if statements[0].total_debt and statements[0].cash else 0
                    )
                else:
                    # Single-stage DCF
                    dcf_result = calculator.calculate_single_stage_dcf(
                        base_fcf=base_fcf,
                        growth_rate=0.08,
                        wacc=wacc,
                        terminal_growth=terminal_growth,
                        projection_years=projection_years,
                        shares_outstanding=int(company_info.market_cap / price_data.current_price) if company_info.market_cap and price_data.current_price else None,
                        net_debt=statements[0].total_debt - statements[0].cash if statements[0].total_debt and statements[0].cash else 0
                    )
                
                # Reverse DCF
                historical_cagr = calculate_historical_fcf_cagr(statements, years=5)
                if dcf_result.shares_outstanding:
                    reverse_dcf = calculator.calculate_reverse_dcf(
                        current_price=price_data.current_price,
                        base_fcf=base_fcf,
                        wacc=wacc,
                        terminal_growth=terminal_growth,
                        projection_years=projection_years,
                        shares_outstanding=dcf_result.shares_outstanding,
                        net_debt=dcf_result.equity_value - dcf_result.enterprise_value if dcf_result.equity_value and dcf_result.enterprise_value else 0,
                        historical_fcf_cagr=historical_cagr
                    )
            
            # Sentiment
            sentiment_result = None
            if use_ai_sentiment and news:
                analyzer = SentimentAnalyzer(use_finbert=True, use_llm=True)
                sentiment_result = analyzer.analyze_news_articles(news, use_llm_for_context=True)
            
            # Insider trading net flow
            insider_net_flow = None
            if insider_txns:
                buys = sum(t.value for t in insider_txns if t.transaction_type.lower() in ["buy", "purchase"] and t.value)
                sells = sum(t.value for t in insider_txns if t.transaction_type.lower() in ["sell", "sale"] and t.value)
                insider_net_flow = buys - sells
            
            # Decision Engine
            engine = DecisionEngine()
            decision_signal = engine.make_decision(
                price_data=price_data,
                metrics=metrics,
                dcf_result=dcf_result,
                reverse_dcf=reverse_dcf,
                piotroski=quality_results.get("piotroski"),
                altman=quality_results.get("altman"),
                beneish=quality_results.get("beneish"),
                sentiment=sentiment_result,
                insider_net_flow=insider_net_flow,
                analyst_target=analyst_estimates.target_price_mean if analyst_estimates else None
            )
        
        # Display decision
        display_decision_card(decision_signal)
        
        st.markdown("---")
        
        # Reasoning - jako expanders (šetří prostor)
        if decision_signal.strengths:
            with st.expander("💪 Strengths", expanded=True):
                for strength in decision_signal.strengths:
                    st.success(strength, icon="✅")
        
        if decision_signal.reasoning:
            with st.expander("📝 Reasoning", expanded=False):
                for reason in decision_signal.reasoning:
                    st.info(reason, icon="ℹ️")
        
        if decision_signal.warnings:
            with st.expander("⚠️ Warnings", expanded=True):
                for warning in decision_signal.warnings:
                    st.warning(warning, icon="⚠️")
        
        # Reverse DCF interpretation
        if reverse_dcf:
            st.markdown("---")
            st.subheader("🔄 Reverse DCF Analysis")
            
            st.markdown(reverse_dcf.interpretation)
            
            col1, col2 = st.columns(2)
            with col1:
                if reverse_dcf.implied_fcf_growth:
                    st.metric("Implied FCF Growth", format_percent(reverse_dcf.implied_fcf_growth))
            with col2:
                if reverse_dcf.historical_5y_cagr:
                    st.metric("Historical 5Y CAGR", format_percent(reverse_dcf.historical_5y_cagr))
    
    # --- TAB 2: VALUATION ---
    with tabs[1]:
        st.header("💰 Valuation Analysis")
        
        if dcf_result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DCF Fair Value", format_money(dcf_result.fair_value_per_share))
            with col2:
                st.metric("Enterprise Value", format_money(dcf_result.enterprise_value / 1e9, decimals=2) + "B")
            with col3:
                st.metric("Model Type", dcf_result.model_type.replace("_", " ").title())
            
            if show_football_field:
                st.markdown("### Football Field Chart")
                
                # Prepare data
                analyst_data = None
                if analyst_estimates:
                    analyst_data = (
                        analyst_estimates.target_price_low,
                        analyst_estimates.target_price_mean,
                        analyst_estimates.target_price_high
                    )
                
                chart = FootballFieldChart.create_chart(
                    current_price=price_data.current_price,
                    dcf_value=dcf_result.fair_value_per_share,
                    dcf_range=(dcf_result.fair_value_per_share * 0.85, dcf_result.fair_value_per_share * 1.15),
                    analyst_low=analyst_data[0] if analyst_data else None,
                    analyst_mean=analyst_data[1] if analyst_data else None,
                    analyst_high=analyst_data[2] if analyst_data else None,
                    title=f"{ticker} Valuation Football Field"
                )
                st.plotly_chart(chart, use_container_width=True)
        else:
            st.warning("DCF valuace není k dispozici (chybí FCF data)")
    
    # --- TAB 3: QUALITY SCORES ---
    with tabs[2]:
        display_quality_scores(quality_results)
    
    # --- TAB 4: SENTIMENT ---
    with tabs[3]:
        st.header("📈 Sentiment Analysis")
        
        if sentiment_result:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Sentiment", sentiment_result.overall_sentiment)
            with col2:
                st.metric("Score", f"{sentiment_result.overall_score:.2f}")
            with col3:
                st.metric("Confidence", format_percent(sentiment_result.confidence))
            with col4:
                st.metric("Articles", sentiment_result.num_articles)
            
            # Distribution
            st.markdown("### Sentiment Distribution")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Positive", sentiment_result.positive_count, delta=None)
            with col2:
                st.metric("Neutral", sentiment_result.neutral_count, delta=None)
            with col3:
                st.metric("Negative", sentiment_result.negative_count, delta=None)
            
            if sentiment_result.key_themes:
                st.markdown("### Key Themes")
                st.write(", ".join(sentiment_result.key_themes))
        else:
            st.info("Sentiment analýza není dostupná")
    
    # --- TAB 5: NEWS & FUNDAMENTALS ---
    with tabs[4]:
        st.header("📰 Recent News")
        if news:
            for article in news[:5]:
                with st.expander(f"📄 {article['title']}", expanded=False):
                    st.caption(f"**Publisher:** {article['publisher']} | **Date:** {article['published_date']}")
                    st.write(article.get('summary', ''))
                    st.markdown(f"[Read more]({article['link']})")
        
        st.markdown("---")
        st.header("📊 Key Fundamentals")
        
        if metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("P/E Ratio", format_number(metrics.pe_ratio))
                st.metric("P/B Ratio", format_number(metrics.price_to_book))
                st.metric("P/S Ratio", format_number(metrics.price_to_sales))
            with col2:
                st.metric("Operating Margin", format_percent(metrics.operating_margin))
                st.metric("Net Margin", format_percent(metrics.net_margin))
                st.metric("ROE", format_percent(metrics.roe))
            with col3:
                st.metric("Current Ratio", format_number(metrics.current_ratio))
                st.metric("Debt/Equity", format_number(metrics.debt_to_equity))
                st.metric("FCF Yield", format_percent(metrics.fcf_yield))
    
    # --- TAB 6: MEMO & WATCHLIST ---
    with tabs[5]:
        st.header("📝 Investment Memo & Watchlist")
        
        # Load data
        memos = get_memos()
        watch = get_watchlist()
        memo = memos.get("memos", {}).get(ticker, {})
        wl = watch.get("items", {}).get(ticker, {})
        
        # Auto-draft snippets
        auto_thesis = f"{company_info.name} — Firma v sektoru {company_info.sector}, {company_info.industry}. Hlavní investiční teze a konkurenční výhoda."
        auto_drivers = "- Růst tržeb a earnings\n- Provozní marže a efficiency\n- Kapitálová alokace (buybacks/dividends)\n- Makro tailwinds a produktové cykly"
        auto_risks = "- Vysoká valuace vyžaduje rychlý růst\n- Konkurenční tlaky\n- Regulační rizika\n- Cyklická poptávka"
        
        if dcf_result and dcf_result.fair_value_per_share:
            auto_buy = f"- Buy zone: pod {format_money(dcf_result.fair_value_per_share * 0.85)} (MOS ~20%)\n- Watch: kolem {format_money(dcf_result.fair_value_per_share)}\n- Avoid: výrazně nad fair value bez zrychlení fundamentů"
        else:
            auto_buy = "- Buy zone: TBD\n- Watch: TBD\n- Avoid: TBD"
        
        # Memo Section
        st.subheader("📄 Investment Memo")
        
        col1, col2 = st.columns(2)
        with col1:
            thesis = st.text_area(
                "Investiční teze",
                value=memo.get("thesis", auto_thesis),
                height=100,
                help="Proč by firma měla vyhrát?"
            )
            
            drivers = st.text_area(
                "Key Drivers",
                value=memo.get("drivers", auto_drivers),
                height=100,
                help="Co musí platit, aby teze vyšla?"
            )
            
            risks = st.text_area(
                "Rizika",
                value=memo.get("risks", auto_risks),
                height=100,
                help="Co může pokazit investici?"
            )
        
        with col2:
            catalysts = st.text_area(
                "Catalysts",
                value=memo.get("catalysts", ""),
                height=100,
                help="Co může pohnout cenou nahoru?"
            )
            
            buy_conditions = st.text_area(
                "Buy Podmínky",
                value=memo.get("buy_conditions", auto_buy),
                height=100,
                help="Za jakých podmínek koupit?"
            )
            
            notes = st.text_area(
                "Poznámky",
                value=memo.get("notes", ""),
                height=100,
                help="Další poznámky"
            )
        
        # Save button
        if st.button("💾 Uložit Memo", use_container_width=True, type="primary"):
            if "memos" not in memos:
                memos["memos"] = {}
            
            memos["memos"][ticker] = {
                "thesis": thesis,
                "drivers": drivers,
                "risks": risks,
                "catalysts": catalysts,
                "buy_conditions": buy_conditions,
                "notes": notes,
                "updated_at": datetime.now().isoformat(),
            }
            set_memos(memos)
            st.success("✅ Memo uloženo!")
        
        st.markdown("---")
        
        # Watchlist Section
        st.subheader("⭐ Watchlist")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            target_buy = st.number_input(
                "Cílová nákupní cena",
                value=float(wl.get("target_buy", 0.0)) if wl else 0.0,
                step=10.0,
                help="Cena, za kterou chceš koupit"
            )
        
        with col2:
            if st.button("⭐ Přidat do watchlistu", use_container_width=True):
                if "items" not in watch:
                    watch["items"] = {}
                
                watch["items"][ticker] = {
                    "target_buy": target_buy,
                    "added_at": wl.get("added_at") or datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                }
                set_watchlist(watch)
                st.success("✅ Přidáno!")
                st.rerun()
        
        with col3:
            if st.button("🗑️ Odebrat", use_container_width=True):
                if ticker in watch.get("items", {}):
                    watch["items"].pop(ticker, None)
                    set_watchlist(watch)
                    st.success("✅ Odebráno!")
                    st.rerun()
        
        # Display watchlist
        st.markdown("### Moje Watchlist")
        items = watch.get("items", {})
        
        if not items:
            st.info("💡 Watchlist je prázdný. Přidej první ticker!")
        else:
            import pandas as pd
            rows = []
            
            for tkr, item in items.items():
                try:
                    provider = get_data_provider()
                    price_data_wl = provider.get_price_data(tkr)
                    price_now = price_data_wl.current_price if price_data_wl else None
                except Exception:
                    price_now = None
                
                tgt = item.get("target_buy")
                hit = (price_now is not None and tgt is not None and tgt > 0 and price_now <= tgt)
                
                rows.append({
                    "Ticker": tkr,
                    "Current Price": format_money(price_now) if price_now else "—",
                    "Target Buy": format_money(tgt) if tgt else "—",
                    "Alert": "🔔 BUY NOW!" if hit else "⏳ Wait",
                    "Updated": item.get("updated_at", "")[:10],  # Just date
                })
            
            df_watchlist = pd.DataFrame(rows)
            st.dataframe(df_watchlist, use_container_width=True, hide_index=True)
            st.caption("🔔 = Cena dosáhla nebo je pod target buy price!")
    
    # Footer
    st.markdown("---")
    st.caption(f"Data: {provider.get_provider_name()} | Verze: {config.APP_VERSION} | ⚠️ Toto není investiční doporučení")


if __name__ == "__main__":
    main()
