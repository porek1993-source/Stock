"""
Stock Analysis Pro - Masterpiece v19.0
Ultimate Terminal: Gemini 1.5 Flash (1500 RPD), DCF Priority, Altman Z-Score, 10Y History.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

warnings.filterwarnings('ignore')

# ============================================================================
# 1. DESIGN A DARK TERMINAL STYLING
# ============================================================================
st.set_page_config(page_title="Stock Analyzer Master Pro 2026", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .main-header { font-size: 3.2rem; font-weight: 800; background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; padding: 25px; }
    div[data-testid="stMetric"] { background-color: #1e293b !important; border-radius: 12px; padding: 20px !important; border: 1px solid #334155 !important; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
    div[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.8rem !important; }
    div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color: #0f172a; gap: 10px; }
    .stTabs [data-baseweb="tab"] { background-color: #1e293b; color: #94a3b8; border-radius: 8px 8px 0 0; padding: 10px 20px; border: 1px solid #334155 !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 2. ANALYTIKŮV MOZEK (VÝPOČTY A LOGIKA)
# ============================================================================

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return float(str(x).replace(",", "").strip())
    except: return None

def classify_insider_action(text_value: str, txn_value: str) -> str:
    s = f"{txn_value or ''} {text_value or ''}".lower()
    if any(k in s for k in ["sale", "sell", "sold", "dispose"]): return "SELL"
    if any(k in s for k in ["buy", "purchase", "acquire"]): return "BUY"
    if any(k in s for k in ["award", "grant", "rsu", "option"]): return "GRANT"
    return "OTHER"

def calculate_intrinsic_value_pro(info: Dict) -> Tuple[Optional[float], str]:
    """DCF Prioritní výpočet vnitřní hodnoty."""
    try:
        fcf = _safe_float(info.get("freeCashflow"))
        shares = _safe_float(info.get("sharesOutstanding"))
        if fcf and shares and fcf > 0:
            discount, growth, terminal, years = 0.10, 0.05, 0.02, 5
            pv_fcf = sum([(fcf * (1 + growth)**i) / (1 + discount)**i for i in range(1, years + 1)])
            tv = (fcf * (1 + growth)**years * (1 + terminal)) / (discount - terminal)
            pv_tv = tv / (1 + discount)**years
            return (pv_fcf + pv_tv) / shares, "DCF Model (Intrinsic)"
    except: pass
    t_mean = _safe_float(info.get("targetMeanPrice"))
    return t_mean, "Analyst Consensus" if t_mean else "N/A"

def get_altman_z_score(stock_obj, info):
    """Altman Z-Score pro predikci úpadku."""
    try:
        bs = stock_obj.balance_sheet.iloc[:, 0]
        is_stmt = stock_obj.financials.iloc[:, 0]
        assets = _safe_float(bs.get('Total Assets', 1))
        ebit = _safe_float(is_stmt.get('EBIT', 0))
        wc = _safe_float(bs.get('Working Capital', 0))
        re = _safe_float(bs.get('Retained Earnings', 0))
        mc = _safe_float(info.get('marketCap', 0))
        liab = _safe_float(bs.get('Total Liabilities Net Minority Interest', 1))
        rev = _safe_float(is_stmt.get('Total Revenue', 0))
        z = (wc/assets * 1.2) + (re/assets * 1.4) + (ebit/assets * 3.3) + (mc/liab * 0.6) + (rev/assets * 1.0)
        return round(z, 2)
    except: return None

# ============================================================================
# 3. HLAVNÍ UI APLIKACE
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Analyzer Master Pro</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Ovládací panel")
        ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
        history_len = st.selectbox("Historie", ["2y", "5y", "10y", "max"], index=2)
        st.markdown("---")
        ai_on = st.checkbox("🤖 AI Hloubková analýza", value=False)
        # Priorita: Načtení z Koyeb Env Variables, jinak text input
        api_key = os.environ.get("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
        
        st.subheader("🧮 Vlastní DCF model")
        user_g = st.slider("Očekávaný růst (5y)", 0.0, 0.4, 0.12)
        user_d = st.slider("Diskontní sazba", 0.07, 0.18, 0.10)
        
        run = st.button("SPUSTIT ANALÝZU", type="primary", use_container_width=True)

    if run or ticker:
        with st.spinner(f"Analyzuji data pro {ticker}..."):
            stock = yf.Ticker(ticker)
            df = stock.history(period="2y")
            info = stock.info
            
            if df.empty:
                st.error("Data nebyla nalezena. Zkontrolujte ticker.")
                return

            # --- TOP HEADER METRIKY ---
            h1, h2, h3, h4 = st.columns(4)
            curr_p = info.get('currentPrice', df['Close'].iloc[-1])
            h1.metric("🏢 Firma", info.get('shortName', ticker))
            h2.metric("💰 Cena", f"${curr_p:.2f}", f"{((curr_p - info.get('previousClose', curr_p))/info.get('previousClose', 1)*100):.2f}%")
            
            fair_v, fair_m = calculate_intrinsic_value_pro(info)
            if fair_v:
                upside = ((fair_v / curr_p) - 1) * 100
                h3.metric("🎯 Férová cena", f"${fair_v:.2f}", f"{upside:+.1f}%")
                st.caption(f"Metoda: {fair_m}")
            
            z_score = get_altman_z_score(stock, info)
            h4.metric("🛡️ Altman Z-Score", z_score if z_score else "N/A", "Safe" if z_score and z_score > 2.99 else "Risk")

            # --- ANALYTICKÉ TABY ---
            tabs = st.tabs(["📈 Graf", "📊 10Y Fundamenty", "📰 AI Sentiment", "💼 Insiders", "🏥 Zdraví", "🌍 Makro & Peers", "🧾 Investor Dashboard"])

            with tabs[0]: # Technická analýza
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Cena'), row=1, col=1)
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(50).mean(), name='SMA 50', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Objem', opacity=0.3), row=2, col=1)
                fig.update_layout(height=600, template='plotly_dark', xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]: # 10letá historie
                st.subheader("🏛️ Hloubková historie (Tržby, Zisk, FCF)")
                fin = stock.financials.T
                cf = stock.cashflow.T
                if not fin.empty:
                    hist_data = pd.DataFrame({
                        "Tržby": fin['Total Revenue'] if 'Total Revenue' in fin.columns else 0,
                        "Zisk": fin['Net Income'] if 'Net Income' in fin.columns else 0,
                        "FCF": cf['Free Cash Flow'] if not cf.empty and 'Free Cash Flow' in cf.columns else 0
                    }).sort_index()
                    st.bar_chart(hist_data)
                    st.dataframe(hist_data.T, use_container_width=True)

            with tabs[2]: # AI Sentiment - PŘEPNUTO NA GEMINI 1.5 FLASH
                st.subheader("📰 AI Analýza novinek (Gemini 1.5 Flash)")
                news_items = stock.news[:10]
                if ai_on and api_key:
                    try:
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        titles = [n.get('title') for n in news_items if n.get('title')]
                        prompt = f"Analyzuj sentiment pro {ticker} z těchto zpráv:\n" + "\n".join(titles[:8])
                        # POUŽITÍ MODELU 1.5 FLASH PRO VYSOKÉ LIMITY
                        resp = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                        st.info(f"### 🤖 AI Analýza:\n{resp.text}")
                    except Exception as e: st.error(f"AI Chyba: {e}")
                
                for n in news_items:
                    with st.expander(f"📰 {n.get('title')}"):
                        st.write(f"Zdroj: {n.get('publisher')}")
                        st.markdown(f"[Odkaz na zprávu]({n.get('link')})")

            with tabs[4]: # Finanční zdraví
                st.subheader("🏥 Detailní analýza zdraví")
                c1, c2, c3 = st.columns(3)
                c1.metric("Debt/Equity", f"{info.get('debtToEquity', 'N/A')}")
                c2.metric("Current Ratio", f"{info.get('currentRatio', 'N/A')}")
                c3.metric("Profit Margin", f"{info.get('profitMargins', 0)*100:.2f}%")

            with tabs[6]: # Investor Dashboard
                st.subheader("🧾 Dashboard & Scénáře")
                s1, s2 = st.columns(2)
                scen_growth = s1.slider("Scénář: Růst tržeb p.a.", 0.0, 0.5, 0.10)
                scen_margin = s2.slider("Scénář: FCF marže", 0.0, 0.5, 0.20)
                # Sanity check výpočet
                fcf_est = info.get('totalRevenue', 0) * (1 + scen_growth)**5 * scen_margin
                st.metric("Odhad FCF (5 let)", f"${fcf_est/1e9:.2f}B")
                
                st.markdown("### 🚩 Red Flags")
                if info.get('debtToEquity', 0) > 150: st.warning("Vysoký dluh vůči vlastnímu kapitálu.")
                if info.get('currentRatio', 0) < 1: st.error("Firma může mít problém s krátkodobou likviditou.")
                else: st.success("Žádné kritické fundamentální Red Flags nenalezeny.")

            st.markdown('<div style="text-align:center; padding:20px; color:#64748b;">Stock Analyzer Master Pro v19.0 | Data: Yahoo Finance & Gemini 1.5 Flash</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
