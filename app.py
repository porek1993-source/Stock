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

# Load external CSS for mobile fixes
try:
    import os
    css_path = os.path.join(os.path.dirname(__file__), "mobile_fix.css")
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception as e:
    logger.warning(f"Could not load mobile_fix.css: {e}")

# Mobile-friendly CSS (inline backup)
st.markdown("""
    <style>
    /* CRITICAL: Fix transparent sidebar on mobile */
    [data-testid="stSidebar"] {
        background-color: rgb(14, 17, 23) !important;
        z-index: 999999 !important;
    }
    
    /* Dark theme sidebar background */
    [data-testid="stSidebar"] > div:first-child {
        background-color: rgb(14, 17, 23) !important;
    }
    
    /* Light theme sidebar background */
    @media (prefers-color-scheme: light) {
        [data-testid="stSidebar"] {
            background-color: rgb(255, 255, 255) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: rgb(255, 255, 255) !important;
        }
    }
    
    /* Sidebar overlay on mobile */
    @media (max-width: 768px) {
        /* Force sidebar to cover main content when open */
        [data-testid="stSidebar"][aria-expanded="true"] {
            position: fixed !important;
            left: 0 !important;
            top: 0 !important;
            height: 100vh !important;
            width: 80% !important;
            max-width: 300px !important;
            background-color: rgb(14, 17, 23) !important;
            z-index: 999999 !important;
            box-shadow: 2px 0 10px rgba(0, 0, 0, 0.5) !important;
        }
        
        /* Blur main content when sidebar is open */
        [data-testid="stSidebar"][aria-expanded="true"] ~ * {
            filter: blur(3px);
            pointer-events: none;
        }
        
        /* Form styling in sidebar */
        [data-testid="stSidebar"] [data-testid="stForm"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            padding: 1.5rem !important;
            border-radius: 0.75rem !important;
            margin: 1rem 0 !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }
        
        /* Make form inputs more visible */
        [data-testid="stSidebar"] input {
            background-color: rgba(255, 255, 255, 0.15) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }
        
        /* Reduce padding on mobile main content */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        /* Make metrics more compact */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        /* Make tabs scrollable on mobile */
        [data-baseweb="tab-list"] {
            overflow-x: auto;
            flex-wrap: nowrap;
        }
        
        /* Improve button sizes on mobile */
        .stButton button {
            width: 100%;
            font-size: 1rem;
            padding: 0.75rem;
        }
    }
    
    /* Better decision card on all devices */
    .decision-card {
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Improve expander visibility */
    .streamlit-expanderHeader {
        font-size: 1rem !important;
        font-weight: 600 !important;
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
    
    # Mobile sidebar overlay dim effect
    st.markdown("""
        <script>
        // Add dark overlay when sidebar is open on mobile
        const checkSidebar = () => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            const main = document.querySelector('.main');
            if (sidebar && main) {
                const isOpen = sidebar.getAttribute('aria-expanded') === 'true';
                if (window.innerWidth <= 768 && isOpen) {
                    main.style.opacity = '0.3';
                } else {
                    main.style.opacity = '1';
                }
            }
        };
        setInterval(checkSidebar, 100);
        </script>
    """, unsafe_allow_html=True)
    
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
        "📰 News & Fundamentals"
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
    
    # Footer
    st.markdown("---")
    st.caption(f"Data: {provider.get_provider_name()} | Verze: {config.APP_VERSION} | ⚠️ Toto není investiční doporučení")


if __name__ == "__main__":
    main()
