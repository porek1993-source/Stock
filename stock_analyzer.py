


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

# Optional PDF export
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False


APP_NAME = "Stock Picker Pro"
APP_VERSION = "v1.0"
# ---------------- Optional AI (Gemini) ----------------
# Put your key directly here (hardcoded) if you don't want ENV:
GEMINI_API_KEY = ""  # e.g. "AIza..."
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Safety: AI calls are opt-in via a button in the UI to avoid quota / 429 issues.

DATA_DIR = os.path.join(os.path.dirname(__file__), ".stock_picker_pro")
WATCHLIST_PATH = os.path.join(DATA_DIR, "watchlist.json")
MEMOS_PATH = os.path.join(DATA_DIR, "memos.json")


# ---------------- FMP (Financial Modeling Prep) ----------------
# Optional: improves missing fundamentals vs Yahoo (Current Ratio, Debt/Assets, FCF Yield, etc.).
# You can hardcode your key here for local testing.
FMP_API_KEY = ""  # <-- paste your FMP key (or leave blank to disable)

def _fmp_get_json(endpoint: str, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Optional[Any]:
    """Small helper to call FMP without adding external deps (requests)."""
    if not FMP_API_KEY:
        return None
    try:
        import urllib.parse
        import urllib.request
        q = dict(params or {})
        q["apikey"] = FMP_API_KEY
        url = "https://financialmodelingprep.com/api/v3/" + endpoint.lstrip("/")
        url = url + "?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return json.loads(raw) if raw else None
    except Exception:
        return None

def enrich_metrics_with_fmp(ticker: str, metrics: Dict[str, "Metric"]) -> Tuple[Dict[str, "Metric"], List[str]]:
    """Fill missing ratios from FMP if available."""
    notes: List[str] = []
    if not FMP_API_KEY:
        return metrics, notes

    # Ratios (TTM)
    ratios = _fmp_get_json(f"ratios-ttm/{ticker}") or []
    if isinstance(ratios, list) and ratios:
        r0 = ratios[0] or {}
        # Current ratio
        if metrics.get("current_ratio") and metrics["current_ratio"].value is None:
            v = safe_float(r0.get("currentRatioTTM"))
            if v is not None:
                metrics["current_ratio"].value = v
                metrics["current_ratio"].source = "FMP"
                notes.append("Current Ratio doplněn z FMP (ratios-ttm).")
        # Debt ratio (Total liabilities / total assets) ~ close to Debt/Assets
        if metrics.get("leverage") and metrics["leverage"].value is None:
            v = safe_float(r0.get("debtRatioTTM"))
            if v is not None:
                metrics["leverage"].value = v
                metrics["leverage"].source = "FMP"
                notes.append("Debt/Assets doplněno z FMP (debtRatioTTM).")

    # Key metrics TTM (FCF yield)
    km = _fmp_get_json(f"key-metrics-ttm/{ticker}") or []
    if isinstance(km, list) and km:
        k0 = km[0] or {}
        if metrics.get("fcf_yield") and metrics["fcf_yield"].value is None:
            v = safe_float(k0.get("freeCashFlowYieldTTM"))
            if v is None:
                v = safe_float(k0.get("fcfYieldTTM"))
            if v is not None:
                metrics["fcf_yield"].value = v
                metrics["fcf_yield"].source = "FMP"
                notes.append("FCF Yield doplněn z FMP (key-metrics-ttm).")

    return metrics, notes



# ---------------------------
# Utilities
# ---------------------------
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
        if isinstance(x, (int, float)) and (math.isfinite(float(x))):
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
    if a is None or b is None:
        return None
    if b == 0:
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



@st.cache_data(show_spinner=False, ttl=60*60)
def get_all_time_high(ticker: str) -> Optional[float]:
    """All‑time high price based on Yahoo historical data (max daily High).

    Note: Requires a separate `period='max'` fetch.
    """
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="max", interval="1d", auto_adjust=False)
        if h is None or h.empty:
            return None
        # prefer High, fallback to Close
        col = "High" if "High" in h.columns else ("Close" if "Close" in h.columns else None)
        if not col:
            return None
        return float(pd.to_numeric(h[col], errors="coerce").max())
    except Exception:
        return None


def clamp(v: Optional[float], lo: float, hi: float) -> Optional[float]:
    if v is None:
        return None
    return max(lo, min(hi, v))


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def to_date(x: Any) -> Optional[dt.date]:
    if x is None:
        return None
    if isinstance(x, dt.date) and not isinstance(x, dt.datetime):
        return x
    if isinstance(x, dt.datetime):
        return x.date()
    try:
        return pd.to_datetime(x).date()
    except Exception:
        return None


# ---------------------------
# Data fetch (cached)
# ---------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    # yfinance can sometimes return MultiIndex columns; flatten for single ticker
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # normalize common column variants
    if isinstance(df, pd.DataFrame):
        cols_lower = {str(c).lower().replace(' ', ''): c for c in df.columns}
        if 'close' not in df.columns and 'close' in cols_lower:
            df.rename(columns={cols_lower['close']: 'Close'}, inplace=True)
        if 'Close' not in df.columns and 'adjclose' in cols_lower:
            df.rename(columns={cols_lower['adjclose']: 'Close'}, inplace=True)
        if 'Adj Close' in df.columns and 'Close' not in df.columns:
            df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    # standardize time column
    if "Date" in df.columns:
        df.rename(columns={"Date": "Datetime"}, inplace=True)
    elif "Datetime" not in df.columns and df.columns[0] not in ("Datetime",):
        df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
    return df


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        return t.info or {}
    except Exception:
        return {}


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_ticker_objects(ticker: str) -> Dict[str, Any]:
    """Fetch financial tables that yfinance provides. Cache it to reduce Yahoo rate issues."""
    t = yf.Ticker(ticker)
    out = {}
    for attr in ["financials", "balance_sheet", "cashflow", "quarterly_financials",
                 "quarterly_balance_sheet", "quarterly_cashflow", "earnings", "quarterly_earnings",
                 "recommendations", "calendar"]:
        try:
            out[attr] = getattr(t, attr)
        except Exception:
            out[attr] = None
    # Insider transactions dataframe (can be missing)
    try:
        out["insider_transactions"] = getattr(t, "insider_transactions", None)
    except Exception:
        out["insider_transactions"] = None
    # Some yfinance versions have insider_purchases/sales
    for attr in ["insider_purchases", "insider_roster_holders", "major_holders", "institutional_holders"]:
        try:
            out[attr] = getattr(t, attr)
        except Exception:
            out[attr] = None
    return out


# ---------------------------
# News extraction (robust)
# ---------------------------
def _extract_title(item: Any) -> Optional[str]:
    if isinstance(item, str):
        s = item.strip()
        return s or None
    if not isinstance(item, dict):
        return None
    # common direct keys
    for k in ("title", "headline", "text", "summary"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    content = item.get("content")
    if isinstance(content, dict):
        for k in ("title", "headline", "summary", "description"):
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def _extract_publisher(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for k in ("publisher", "source", "provider"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    content = item.get("content")
    if isinstance(content, dict):
        prov = content.get("provider")
        if isinstance(prov, dict):
            name = prov.get("displayName") or prov.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def _extract_url(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for k in ("link", "url"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    content = item.get("content")
    if isinstance(content, dict):
        cu = content.get("canonicalUrl")
        if isinstance(cu, dict):
            u = cu.get("url")
            if isinstance(u, str) and u.strip():
                return u.strip()
        u = content.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()
    return None


def _extract_time(item: Any) -> Optional[dt.datetime]:
    if not isinstance(item, dict):
        return None
    for k in ("providerPublishTime", "published", "pubDate", "time"):
        v = item.get(k)
        if isinstance(v, (int, float)) and v > 0:
            try:
                return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc)
            except Exception:
                pass
        if isinstance(v, str) and v.strip():
            try:
                return pd.to_datetime(v, utc=True).to_pydatetime()
            except Exception:
                pass
    content = item.get("content")
    if isinstance(content, dict):
        v = content.get("pubDate") or content.get("publishedAt")
        if isinstance(v, str) and v.strip():
            try:
                return pd.to_datetime(v, utc=True).to_pydatetime()
            except Exception:
                pass
    return None


def fetch_news(ticker: str, limit: int = 12) -> List[Dict[str, Any]]:
    """Try multiple Yahoo/yfinance pathways to get readable headlines + url."""
    items: List[Dict[str, Any]] = []
    try:
        t = yf.Ticker(ticker)
        raw = getattr(t, "news", None) or []
        for it in raw:
            title = _extract_title(it)
            if not title:
                continue
            items.append({
                "title": title,
                "publisher": _extract_publisher(it),
                "url": _extract_url(it),
                "published": _extract_time(it),
                "raw": it
            })
    except Exception:
        pass

    # Fallback: yfinance Search
    if len(items) == 0:
        try:
            s = yf.Search(ticker)
            raw = getattr(s, "news", None) or []
            for it in raw:
                title = _extract_title(it)
                if not title:
                    continue
                items.append({
                    "title": title,
                    "publisher": _extract_publisher(it),
                    "url": _extract_url(it),
                    "published": _extract_time(it),
                    "raw": it
                })
        except Exception:
            pass

    # Deduplicate by title
    seen = set()
    out = []
    for it in items:
        key = it["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out


# ---------------- News relevance (ticker filtering) ----------------
_CLICKBAIT_PATTERNS = [
    r"\b3\s+stocks\b", r"\b\d+\s+stocks\b", r"\bhere['’]s\b", r"\bone\s+reason\b",
    r"\bshould\s+you\s+buy\b", r"\bthink\s+it['’]s\s+too\s+late\b", r"\btop\s+\d+\b",
]

def _news_relevance_score(title: str, ticker: str, company: str) -> int:
    t = (title or "").strip()
    if not t:
        return -5
    up = t.upper()
    score = 0
    tk = (ticker or "").upper().strip()
    comp = (company or "").strip()

    if tk and (tk in up or f"({tk})" in up):
        score += 3
    if comp:
        # check main company tokens (first 2 words) + full
        comp_up = comp.upper()
        if comp_up in up:
            score += 2
        else:
            parts = [p for p in re.split(r"\s+", comp_up) if p]
            for p in parts[:2]:
                if len(p) >= 4 and p in up:
                    score += 1
                    break

    # penalize if headline explicitly mentions another ticker in parentheses
    for m in re.finditer(r"\(([A-Z]{1,5})\)", up):
        other = m.group(1)
        if tk and other != tk:
            score -= 2

    # clickbait penalty
    low = t.lower()
    if any(re.search(p, low) for p in _CLICKBAIT_PATTERNS):
        score -= 1

    return score

def split_relevant_news(news_items: List[Dict[str, Any]], ticker: str, company: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    relevant: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []
    for it in (news_items or []):
        title = it.get("title") or ""
        s = _news_relevance_score(title, ticker, company)
        it["rel_score"] = s
        it["rel_label"] = "RELEVANT" if s >= 2 else ("MAYBE" if s == 1 else "NOISE")
        (relevant if s >= 2 else other).append(it)
    return relevant, other

def google_search_url(query: str) -> str:
    # keep it simple; no direct raw URL in markdown unless within code - but st.link_button expects plain URL
    q = re.sub(r"\s+", "+", query.strip())
    return f"https://www.google.com/search?q={q}"

# ---------------- Sentiment helpers ----------------
_POS_WORDS = {"beat","beats","surge","rally","record","upgrade","raised","raises","outperform","buy","strong","growth","profit","profits","wins","soar","soars","guidance"}
POS_WORDS = _POS_WORDS  # alias used by explain functions
_NEG_WORDS = {"miss","misses","plunge","drop","downgrade","cut","cuts","lawsuit","probe","sec","weak","fall","recall","fraud","warning","decline","selloff","sell-off"}
NEG_WORDS = _NEG_WORDS  # alias used by explain functions
_NEGATIONS = {"not","no","without","never","none","n't"}
NEGATIONS = _NEGATIONS  # backward-compat for helper functions

def _tokenize_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (s or "").lower())


def _contains_negation(tokens: List[str]) -> bool:
    """True if any negation token is present (simple heuristic)."""
    return any(t in NEGATIONS for t in (tokens or []))


def headline_sentiment_score(news_items: List[Dict[str, Any]], max_items: int = 10) -> Tuple[int, str, float]:
    """
    Returns: (score_0_100, label, confidence_0_1)
    Uses simple finance keyword scoring + negation + recency weighting.
    """
    if not news_items:
        return 50, "NEUTRÁLNÍ", 0.0

    now = dt.datetime.now(dt.timezone.utc)
    total = 0.0
    wsum = 0.0
    used = 0

    for it in (news_items or [])[:max_items]:
        title = it.get("title") or ""
        toks = _tokenize_words(title)
        if not toks:
            continue

        score = 0
        for i, t in enumerate(toks):
            if t in _POS_WORDS:
                neg = any(toks[j] in _NEGATIONS for j in range(max(0, i-2), i))
                score += -1 if neg else 1
            elif t in _NEG_WORDS:
                neg = any(toks[j] in _NEGATIONS for j in range(max(0, i-2), i))
                score += 1 if neg else -1

        # Recency weight (newer matters more)
        w = 1.0
        pub = it.get("providerPublishTime") or it.get("published")
        if isinstance(pub, (int, float)):
            days = max(0.0, (now - dt.datetime.utcfromtimestamp(pub)).total_seconds() / 86400.0)
            w = 1.0 / (1.0 + days/3.0)

        total += score * w
        wsum += w
        used += 1

    if used == 0:
        return 50, "NEUTRÁLNÍ", 0.0

    avg = total / (wsum or 1.0)   # typical -2..+2
    score_0_100 = int(max(0, min(100, round(50 + avg * 15))))
    if score_0_100 >= 60:
        label = "POZITIVNÍ"
    elif score_0_100 <= 40:
        label = "NEGATIVNÍ"
    else:
        label = "NEUTRÁLNÍ"

    # Confidence: more items + more recency -> higher
    confidence = min(1.0, 0.25 + 0.07 * used + 0.15 * (wsum / max(1.0, used)))
    return score_0_100, label, confidence



# ---------------------------
# Insider trading parsing
# ---------------------------
_TXN_BUY = re.compile(r"\b(buy|bought|purchase|acquire|acquired|open market purchase)\b", re.I)
_TXN_SELL = re.compile(r"\b(sell|sold|sale|dispose|disposed)\b", re.I)
_TXN_GRANT = re.compile(r"\b(award|grant|rsu|restricted stock|stock award|vesting|vested)\b", re.I)
_TXN_OPTION = re.compile(r"\b(option|exercise|exercised)\b", re.I)
_TXN_10B5 = re.compile(r"\b10b5\-?1\b|\brule\s*10b5\-?1\b", re.I)
_TXN_TAX = re.compile(r"\b(tax|withhold|withholding|cover)\b", re.I)
_TXN_SELL_TO_COVER = re.compile(r"\b(sell\s*to\s*cover|to\s*cover\s*tax|cover\s*taxes?)\b", re.I)




def _tokenize(text: str):
    return re.findall(r"[A-Za-z0-9']+", str(text or "").lower())

def _parse_published_dt(published: Any) -> Optional[dt.datetime]:
    """Parse 'published' from yfinance news item into timezone-aware UTC datetime.

    Handles:
      - unix seconds (int/float)
      - unix milliseconds
      - ISO strings
      - datetime objects (naive -> assume UTC)
      - None -> None
    """
    if published is None:
        return None
    try:
        if isinstance(published, dt.datetime):
            return published.replace(tzinfo=dt.timezone.utc) if published.tzinfo is None else published.astimezone(dt.timezone.utc)
        if isinstance(published, (int, float)):
            ts = float(published)
            # Heuristic: > 10^12 is likely ms
            if ts > 1e12:
                ts = ts / 1000.0
            return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        if isinstance(published, str):
            s = published.strip()
            if not s:
                return None
            # Try ISO format
            try:
                d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
                return d.astimezone(dt.timezone.utc) if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
            except Exception:
                return None
    except Exception:
        return None
    return None


def headline_sentiment_explain(news_items: List[Dict[str, Any]], max_items: int = 10) -> Tuple[int, str, float, List[str], Dict[str, List[str]]]:
    """Explainable fallback sentiment from headlines.

    Returns: (score_0_100, label, confidence_0_1, reasons, highlights)
    highlights: {"positive": [...], "negative": [...]} (top titles driving score)
    """
    if not news_items:
        return 50, "NEUTRÁLNÍ", 0.0, ["Žádné titulky k vyhodnocení."], {"positive": [], "negative": []}

    now = dt.datetime.now(dt.timezone.utc)
    scored: List[Tuple[float, float, float, str]] = []  # (contrib, raw_score, weight, title)

    total = 0.0
    wsum = 0.0
    used = 0

    for it in (news_items or [])[:max_items]:
        title = it.get("title") or ""
        toks = _tokenize(title)
        raw = 0.0
        neg = False
        # negation window: if "not/no/never/n't" exists, invert within headline (simple)
        if _contains_negation(toks):
            neg = True
        for tok in toks:
            if tok in POS_WORDS:
                raw += 1.0
            elif tok in NEG_WORDS:
                raw -= 1.0

        if neg:
            raw *= -1.0

        when = it.get("published_dt")
        if when is None:
            when = _parse_published_dt(it.get("published"))

        age_hours = 24.0
        if isinstance(when, dt.datetime):
            try:
                age_hours = max(0.0, (now - when).total_seconds() / 3600.0)
            except Exception:
                age_hours = 24.0

        # recency weighting: fresh news weighs more, half-life ~36h
        weight = math.exp(-age_hours / 36.0)
        contrib = raw * weight

        total += contrib
        wsum += weight
        used += 1
        scored.append((contrib, raw, weight, title))

    if used == 0 or wsum == 0:
        return 50, "NEUTRÁLNÍ", 0.0, ["Žádné použitelné titulky."], {"positive": [], "negative": []}

    avg = total / wsum
    score_0_100 = int(max(0, min(100, round(50 + avg * 15.0))))

    if score_0_100 >= 60:
        label = "POZITIVNÍ"
    elif score_0_100 <= 40:
        label = "NEGATIVNÍ"
    else:
        label = "NEUTRÁLNÍ"

    confidence = min(1.0, used / 6.0)

    # Top drivers
    pos = [t for t in sorted(scored, key=lambda x: x[0], reverse=True) if t[0] > 0][:3]
    negs = [t for t in sorted(scored, key=lambda x: x[0]) if t[0] < 0][:3]
    highlights = {
        "positive": [t[3] for t in pos],
        "negative": [t[3] for t in negs],
    }

    reasons = [
        f"Hodnoceno {used} titulků, čerstvější mají vyšší váhu.",
        "Jednoduchá slovní heuristika s negací (např. 'not good').",
        "Nečte celý článek – jen titulek (pro přesnější výsledky použij AI analýzu).",
    ]
    return score_0_100, label, confidence, reasons, highlights

def _gemini_available() -> bool:
    return bool(GEMINI_API_KEY)

def gemini_sentiment_from_headlines(news_items: List[Dict[str, Any]], max_items: int = 10) -> Tuple[Optional[int], Optional[str], Optional[float], List[str]]:
    """
    Returns: (score_0_100, label, confidence_0_1, bullet_points)

    IMPORTANT:
    - Always requests GEMINI_MODEL (default: gemini-2.5-flash-lite).
    - Uses the newer `google-genai` SDK if installed; otherwise falls back to deprecated `google-generativeai`.
    - Designed to be called explicitly (button) to avoid quota issues.
    """
    if not _gemini_available():
        return None, None, None, ["Gemini API key not set – using fallback."]

    items = (news_items or [])[:max_items]
    headlines = [it.get("title") or "" for it in items if (it.get("title") or "").strip()]
    if not headlines:
        return None, None, None, ["No headlines available."]

    prompt = (
        "You are a careful financial news sentiment classifier.\n"
        "Classify overall sentiment for the company based ONLY on the following headlines.\n"
        "Output STRICT JSON with keys: score_0_100 (int), label (one of POSITIVE/NEUTRAL/NEGATIVE), confidence_0_1 (float), bullets (array of <=4 short bullets).\n"
        "Important: ignore generic macro headlines unless clearly relevant to the company.\n"
        "Headlines:\n- " + "\n- ".join(headlines)
    )

    # 1) Prefer the new SDK: `google-genai`
    try:
        from google import genai as genai_new  # type: ignore
        client = genai_new.Client(api_key=GEMINI_API_KEY)
        resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        txt = getattr(resp, "text", None) or ""
    except Exception as e_new:
        # 2) Fallback to deprecated SDK if installed
        try:
            import google.generativeai as genai_old  # type: ignore
            genai_old.configure(api_key=GEMINI_API_KEY)
            model = genai_old.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            txt = getattr(resp, "text", None) or ""
        except Exception as e_old:
            msg = str(e_old) or str(e_new)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                return None, None, None, ["AI quota exceeded – using fallback sentiment."]
            # If SDK missing, give actionable hint
            if "No module named" in msg and ("google" in msg or "genai" in msg or "generativeai" in msg):
                return None, None, None, ["Gemini SDK missing. Install: pip install google-genai (preferred) or google-generativeai."]
            return None, None, None, [f"AI error – using fallback: {msg[:160]}"]

    # Extract JSON from the model output
    try:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None, None, None, ["Gemini response did not contain JSON – using fallback."]
        data = json.loads(m.group(0))

        score = int(data.get("score_0_100", 50))
        score = max(0, min(100, score))
        label_raw = str(data.get("label", "NEUTRAL")).upper()
        label_map = {"POSITIVE": "POZITIVNÍ", "NEGATIVE": "NEGATIVNÍ", "NEUTRAL": "NEUTRÁLNÍ"}
        label = label_map.get(label_raw, "NEUTRÁLNÍ")
        conf = float(data.get("confidence_0_1", 0.5))
        conf = max(0.0, min(1.0, conf))
        bullets = data.get("bullets") or []
        if not isinstance(bullets, list):
            bullets = []
        bullets = [str(b) for b in bullets[:4] if str(b).strip()]

        # Make the requested model explicit in the bullets (helps debugging the Google console view)
        if bullets:
            bullets.append(f"Model requested: {GEMINI_MODEL}")
        else:
            bullets = [f"Model requested: {GEMINI_MODEL}"]

        return score, label, conf, bullets
    except Exception as e:
        msg = str(e)
        return None, None, None, [f"AI parse error – using fallback: {msg[:160]}"]


def classify_insider_row(row: pd.Series) -> Tuple[str, str]:
    """Return (Type, Tag) where Tag explains common reasons (planned/tax/award/etc.)."""
    tx = str(row.get("Transaction", "") or "")
    txt = str(row.get("Text", "") or "")
    blob = f"{tx} {txt}".strip()
    if not blob:
        return "Unknown", ""

    # Awards / RSU vesting
    if _TXN_GRANT.search(blob):
        return "Grant/Award", "RSU/Award (vesting)"

    # Options/exercise (may be paired with sell-to-cover)
    if _TXN_OPTION.search(blob) and _TXN_SELL.search(blob):
        if _TXN_SELL_TO_COVER.search(blob) or _TXN_TAX.search(blob):
            return "Sell", "Sell-to-cover (tax)"
        return "Sell", "Option-related"
    if _TXN_OPTION.search(blob) and _TXN_BUY.search(blob):
        return "Buy", "Option-related"

    # Buys
    if _TXN_BUY.search(blob):
        return "Buy", "Open market"

    # Sells
    if _TXN_SELL.search(blob):
        if _TXN_SELL_TO_COVER.search(blob) or _TXN_TAX.search(blob):
            return "Sell", "Sell-to-cover (tax)"
        if _TXN_10B5.search(blob):
            return "Sell", "10b5-1 planned"
        return "Sell", "Open market"

    return "Other", ""


def normalize_insiders_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    out = df.copy()
    # standard column names we expect
    for col in ["Shares", "Value", "URL", "Text", "Insider", "Position", "Transaction", "Start Date", "Ownership"]:
        if col not in out.columns:
            # try best-effort mapping
            for c in out.columns:
                if c.lower() == col.lower():
                    out.rename(columns={c: col}, inplace=True)
                    break
    # If Start Date not present, try 'Start Date' variants
    date_col = None
    for c in out.columns:
        if c.lower() in ("start date", "startdate", "date", "reported"):
            date_col = c
            break
    if date_col and date_col != "Start Date":
        out.rename(columns={date_col: "Start Date"}, inplace=True)

    # classification
    classified = out.apply(classify_insider_row, axis=1)
    # classify_insider_row returns (Type, Tag)
    out["Type"] = classified.apply(lambda x: x[0] if isinstance(x, tuple) else x)
    out["Tag"] = classified.apply(lambda x: x[1] if isinstance(x, tuple) and len(x) > 1 else "")
    # numeric cleanup
    for c in ["Shares", "Value"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    if "Start Date" in out.columns:
        out["Start Date"] = pd.to_datetime(out["Start Date"], errors="coerce")
    return out


# ---------------------------
# Financial metrics & scoring
# ---------------------------
@dataclass
class Metric:
    value: Optional[float]
    label: str
    help: str
    fmt: str = "num"  # num, pct, money
    good_high: Optional[bool] = None  # for quick coloring


def get_latest_col(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    # yfinance columns are dates; take first column as most recent
    try:
        return df.iloc[:, 0]
    except Exception:
        return None


def get_row(df: pd.DataFrame, names: List[str]) -> Optional[float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for n in names:
        if n in df.index:
            return safe_float(df.loc[n].iloc[0] if isinstance(df.loc[n], pd.Series) else df.loc[n])
    # fuzzy match
    idx = [str(i).lower() for i in df.index]
    for n in names:
        nl = n.lower()
        for i, s in enumerate(idx):
            if nl == s:
                try:
                    return safe_float(df.iloc[i, 0])
                except Exception:
                    pass
    return None


def calc_cagr(first: float, last: float, years: float) -> Optional[float]:
    if years <= 0:
        return None
    if first <= 0 or last <= 0:
        return None
    return (last / first) ** (1 / years) - 1


def derive_fcf_ttm(objects: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort Free Cash Flow (FCF).

    Priority:
    1) quarterly_cashflow: sum the most recent 4 quarters of (Operating Cash Flow + CapEx)
       - CapEx is usually negative in Yahoo/yfinance, so we ADD it.
    2) annual cashflow: most recent year (Operating Cash Flow + CapEx)

    Notes:
    - yfinance cashflow data can arrive with columns in either order. We explicitly sort columns
      (newest first) before taking the last 4 quarters.
    """
    qcf = objects.get("quarterly_cashflow")
    acf = objects.get("cashflow")

    def fcf_from(cf: pd.DataFrame, sum_rows: bool) -> Optional[float]:
        if cf is None or not isinstance(cf, pd.DataFrame) or cf.empty:
            return None

        # Identify row names across variants
        op = None
        capex = None
        for r in (
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Net Cash Provided by Operating Activities",
            "Cash Flow From Continuing Operating Activities",
        ):
            if r in cf.index:
                op = cf.loc[r]
                break

        for r in ("Capital Expenditures", "Capital Expenditure", "CapEx"):
            if r in cf.index:
                capex = cf.loc[r]
                break

        if op is None or capex is None:
            return None

        try:
            opv = pd.to_numeric(op, errors="coerce")
            capv = pd.to_numeric(capex, errors="coerce")

            # Sort by column/index label if it looks like datetimes (newest first)
            try:
                opv = opv.sort_index(ascending=False)
                capv = capv.sort_index(ascending=False)
            except Exception:
                pass

            if sum_rows:
                # take up to 4 most recent periods
                f = (opv.iloc[:4].sum() + capv.iloc[:4].sum())  # capex usually negative
                f = safe_float(f)
            else:
                f = safe_float(opv.iloc[0] + capv.iloc[0])

            # If Yahoo returns 0/NaN or absurdly tiny FCF, treat as missing
            if f is None or (isinstance(f, (int, float)) and abs(f) < 1.0):
                return None

            return f
        except Exception:
            return None

    f = fcf_from(qcf, sum_rows=True)
    if f is not None:
        return f

    return fcf_from(acf, sum_rows=False)



def derive_revenue_ttm_or_fy(objects: Dict[str, Any]) -> Optional[float]:
    qf = objects.get("quarterly_financials")
    af = objects.get("financials")
    # Prefer annual Total Revenue
    def get_rev(df: pd.DataFrame, sum_quarters: bool) -> Optional[float]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return None
        row_names = ["Total Revenue", "TotalRevenue", "Revenue"]
        rname = None
        for rn in row_names:
            if rn in df.index:
                rname = rn
                break
        if rname is None:
            return None
        ser = pd.to_numeric(df.loc[rname], errors="coerce")
        if sum_quarters:
            return safe_float(ser.iloc[:4].sum())
        return safe_float(ser.iloc[0])
    rev = get_rev(af, sum_quarters=False)
    if rev is not None:
        return rev
    return get_rev(qf, sum_quarters=True)


def derive_margins(objects: Dict[str, Any]) -> Dict[str, Optional[float]]:
    info = {}  # placeholders for future; use yf.info where possible
    return info


def compute_metrics(ticker: str, info: Dict[str, Any], objects: Dict[str, Any]) -> Dict[str, Metric]:
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    mcap = safe_float(info.get("marketCap"))
    shares = safe_float(info.get("sharesOutstanding"))
    ev = safe_float(info.get("enterpriseValue"))
    pe = safe_float(info.get("trailingPE"))
    fpe = safe_float(info.get("forwardPE"))
    pb = safe_float(info.get("priceToBook"))
    beta = safe_float(info.get("beta"))

    fcf = derive_fcf_ttm(objects)
    rev = derive_revenue_ttm_or_fy(objects)

    fcf_yield = safe_div(fcf, mcap)
    fcf_margin = safe_div(fcf, rev)

    # Balance sheet health
    bsq = objects.get("quarterly_balance_sheet")
    bsa = objects.get("balance_sheet")
    bs = bsq if isinstance(bsq, pd.DataFrame) and not bsq.empty else bsa
    total_assets = get_row(bs, ["Total Assets"])
    total_liab = get_row(bs, ["Total Liab", "Total Liabilities"])
    cash = get_row(bs, ["Cash", "Cash And Cash Equivalents", "Cash And Cash Equivalents, Beginning of Period", "Cash And Short Term Investments"])
    debt = get_row(bs, ["Long Term Debt", "Short Long Term Debt", "Short Term Debt", "Total Debt"])
    cur_assets = get_row(bs, ["Total Current Assets"])
    cur_liab = get_row(bs, ["Total Current Liabilities"])
    current_ratio = safe_div(cur_assets, cur_liab)

    # leverage: Debt/Assets (guarded)
    leverage = None
    if debt is not None and total_assets is not None and total_assets > 0:
        leverage = debt / total_assets

    # dilution / share change (from info if available)
    float_shares = safe_float(info.get("floatShares"))
    held_pct_insiders = safe_float(info.get("heldPercentInsiders"))
    held_pct_inst = safe_float(info.get("heldPercentInstitutions"))

    # Analyst targets
    target_mean = safe_float(info.get("targetMeanPrice"))
    target_median = safe_float(info.get("targetMedianPrice"))
    target_low = safe_float(info.get("targetLowPrice"))
    target_high = safe_float(info.get("targetHighPrice"))
    rec_key = info.get("recommendationKey")
    rec_mean = safe_float(info.get("recommendationMean"))

    # Company name
    company_name = info.get("longName") or info.get("shortName") or info.get("displayName") or ticker

    metrics: Dict[str, Metric] = {
        "company": Metric(company_name, "Firma", "Název společnosti dle Yahoo (longName/shortName).", fmt="num"),
        "price": Metric(price, "Aktuální cena", "Poslední dostupná tržní cena.", fmt="money"),
        "market_cap": Metric(mcap, "Market Cap", "Tržní kapitalizace = cena × počet akcií. Velikost firmy.", fmt="money"),
        "enterprise_value": Metric(ev, "Enterprise Value", "EV ≈ market cap + dluh − cash. Lepší pro porovnání valuace mezi firmami.", fmt="money"),
        "pe": Metric(pe, "P/E (TTM)", "Cena / zisk. Čím vyšší, tím víc trh platí za 1 jednotku zisku.", fmt="num"),
        "forward_pe": Metric(fpe, "Forward P/E", "P/E založené na očekávaném zisku (odhad analytiků).", fmt="num"),
        "pb": Metric(pb, "P/B", "Cena / účetní hodnota. U bank/pojišťoven často důležitější než P/E.", fmt="num"),
        "beta": Metric(beta, "Beta", "Citlivost vůči trhu (S&P 500 = 1). Vyšší beta = větší výkyvy.", fmt="num"),
        "fcf": Metric(fcf, "FCF (TTM)", "Free Cash Flow (TTM) = cashflow z provozu − capex. Peníze dostupné pro buyback/dividendy/splácení dluhu.", fmt="money"),
        "fcf_yield": Metric(fcf_yield, "FCF Yield", "FCF / Market cap. Čím vyšší, tím 'levnější' firma vzhledem k cashflow.", fmt="pct", good_high=True),
        "fcf_margin": Metric(fcf_margin, "FCF Margin", "FCF / Tržby. Kvalita monetizace a efektivita.", fmt="pct", good_high=True),
        "cash": Metric(cash, "Cash", "Hotovost a ekvivalenty (poslední dostupný report).", fmt="money"),
        "debt": Metric(debt, "Debt", "Dluh (poslední dostupný report).", fmt="money"),
        "current_ratio": Metric(current_ratio, "Current Ratio", "Likvidita: krátkodobá aktiva / krátkodobé závazky. <1 může být varování.", fmt="num", good_high=True),
        "leverage": Metric(leverage, "Leverage (Debt/Assets)", "Dluh / aktiva. Hrubý indikátor zadlužení. (Ošetřeno proti nesmyslům z prázdných dat.)", fmt="pct"),
        "target_mean": Metric(target_mean, "Cílová cena (mean)", "Průměrná cílová cena od analytiků (pokud dostupné).", fmt="money"),
        "target_median": Metric(target_median, "Cílová cena (median)", "Medián cílových cen od analytiků (pokud dostupné).", fmt="money"),
        "target_low": Metric(target_low, "Cílová cena (low)", "Nejnižší cílová cena z pokrytí analytiků (pokud dostupné).", fmt="money"),
        "target_high": Metric(target_high, "Cílová cena (high)", "Nejvyšší cílová cena z pokrytí analytiků (pokud dostupné).", fmt="money"),
        "rec_mean": Metric(rec_mean, "Doporučení (mean)", "Nižší = lepší (1=Strong Buy, 3=Hold, 5=Sell) – pokud Yahoo poskytuje.", fmt="num"),
        "shares": Metric(shares, "Shares Outstanding", "Počet vydaných akcií. Důležité pro přepočet na 'per share'.", fmt="num"),
        "float_shares": Metric(float_shares, "Float Shares", "Volně obchodované akcie. Nižší float = větší volatilita.", fmt="num"),
        "held_insiders": Metric(held_pct_insiders, "% drží insideři", "Podíl akcií držený vedením/insidery.", fmt="pct"),
        "held_institutions": Metric(held_pct_inst, "% drží instituce", "Podíl akcií držený institucionálními investory.", fmt="pct"),
    }
    # attach non-metric strings
    metrics["_rec_key"] = Metric(None, "Recommendation key", "", fmt="num")
    metrics["_rec_key"].help = str(rec_key) if rec_key else "—"
    return metrics


# ---------------------------
# DCF & Reverse DCF
# ---------------------------
def dcf_fair_value_per_share(
    fcf_ttm: Optional[float],
    shares: Optional[float],
    growth_yrs: int = 5,
    growth_rate: float = 0.12,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    exit_multiple: Optional[float] = None,
) -> Optional[float]:
    """
    Simplified FCF DCF:
    - Project FCF for growth_yrs with constant growth_rate
    - Terminal value either by Gordon Growth or exit_multiple of last-year FCF
    - Discount back and divide by shares
    """
    fcf_ttm = safe_float(fcf_ttm)
    shares = safe_float(shares)
    if fcf_ttm is None or shares is None or shares <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None

    fcf = fcf_ttm
    pv = 0.0
    for y in range(1, growth_yrs + 1):
        fcf *= (1 + growth_rate)
        pv += fcf / ((1 + discount_rate) ** y)

    if exit_multiple is not None:
        tv = fcf * exit_multiple
    else:
        tv = fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)

    pv += tv / ((1 + discount_rate) ** growth_yrs)
    return pv / shares


def reverse_dcf_implied_growth(
    price: Optional[float],
    fcf_ttm: Optional[float],
    shares: Optional[float],
    growth_yrs: int,
    discount_rate: float,
    terminal_growth: float,
    exit_multiple: Optional[float] = None,
    lo: float = -0.2,
    hi: float = 0.6,
) -> Optional[float]:
    """Solve for growth_rate s.t. DCF fair value matches current price."""
    price = safe_float(price)
    if price is None or price <= 0:
        return None

    def f(gr: float) -> Optional[float]:
        v = dcf_fair_value_per_share(
            fcf_ttm=fcf_ttm,
            shares=shares,
            growth_yrs=growth_yrs,
            growth_rate=gr,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            exit_multiple=exit_multiple,
        )
        return v

    vlo = f(lo)
    vhi = f(hi)
    if vlo is None or vhi is None:
        return None
    # If both on same side, can't solve robustly
    if (vlo - price) * (vhi - price) > 0:
        return None

    for _ in range(40):
        mid = (lo + hi) / 2
        vm = f(mid)
        if vm is None:
            return None
        if abs(vm - price) / price < 0.005:
            return mid
        if (vlo - price) * (vm - price) <= 0:
            hi = mid
            vhi = vm
        else:
            lo = mid
            vlo = vm
    return (lo + hi) / 2



# ---------------------------
# Weighted score + verdict (stock-picking)
# ---------------------------
def _score_from_thresholds(val: Optional[float], low: float, mid: float, high: float, invert: bool = False) -> Optional[float]:
    """Maps val to 0..100 with 50 at mid, using linear ramps. invert=True flips direction (lower is better)."""
    v = safe_float(val)
    if v is None:
        return None
    if not invert:
        if v <= low: return 0.0
        if v >= high: return 100.0
        if v < mid:
            return 50.0 * (v - low) / (mid - low)
        return 50.0 + 50.0 * (v - mid) / (high - mid)
    else:
        if v >= high: return 0.0
        if v <= low: return 100.0
        if v > mid:
            return 50.0 * (high - v) / (high - mid)
        return 50.0 + 50.0 * (mid - v) / (mid - low)


def compute_weighted_signal(
    fair_value: Optional[float],
    current_price: Optional[float],
    metrics: Dict[str, Metric],
    info: Dict[str, Any],
    sentiment_score_0_100: Optional[float],
    insider_pro_score_0_100: Optional[float],
    insider_net_flow_value: Optional[float],
    implied_fcf_growth: Optional[float],
    lookback_rev_growth: Optional[float],
) -> Tuple[int, float, str, str, List[str], float, Dict[str, float], List[str]]:
    """Weighted score (0–100) + verdict + bullets + reverse DCF warnings."""
    price = safe_float(current_price)
    fv = safe_float(fair_value)
    mos = None
    if price and fv:
        mos = safe_div(fv - price, price)

    analyst_mean = safe_float(metrics.get("target_mean").value if metrics.get("target_mean") else None)
    analyst_gap = None
    if price and analyst_mean:
        analyst_gap = safe_div(analyst_mean - price, price)

    mos_score = _score_from_thresholds(mos, low=-0.20, mid=0.05, high=0.30, invert=False) if mos is not None else None
    gap_score = _score_from_thresholds(analyst_gap, low=-0.15, mid=0.0, high=0.25, invert=False) if analyst_gap is not None else None
    val_parts = [x for x in [mos_score, gap_score] if x is not None]
    valuation = float(np.mean(val_parts)) if val_parts else 50.0

    curr = metrics.get("current_ratio").value if metrics.get("current_ratio") else None
    debt_assets = metrics.get("leverage").value if metrics.get("leverage") else None
    op_margin = safe_float(info.get("operatingMargins"))

    curr_s = _score_from_thresholds(curr, low=0.6, mid=1.0, high=2.0, invert=False)
    debt_s = _score_from_thresholds(debt_assets, low=0.20, mid=0.45, high=0.80, invert=True)
    opm_s = _score_from_thresholds(op_margin, low=0.00, mid=0.10, high=0.30, invert=False)
    qh_parts = [x for x in [curr_s, debt_s, opm_s] if x is not None]
    quality_health = float(np.mean(qh_parts)) if qh_parts else 50.0

    rev_g = safe_float(info.get("revenueGrowth"))
    fcf_y = metrics.get("fcf_yield").value if metrics.get("fcf_yield") else None
    rev_s = _score_from_thresholds(rev_g, low=-0.05, mid=0.08, high=0.20, invert=False)
    fcfy_s = _score_from_thresholds(fcf_y, low=0.00, mid=0.03, high=0.08, invert=False)
    gr_parts = [x for x in [rev_s, fcfy_s] if x is not None]
    growth = float(np.mean(gr_parts)) if gr_parts else 50.0

    sent = safe_float(sentiment_score_0_100)
    if sent is None:
        rec_mean = metrics.get("rec_mean").value if metrics.get("rec_mean") else None
        sent = 80 if rec_mean is not None and rec_mean <= 2.0 else 60 if rec_mean is not None and rec_mean <= 2.8 else 50.0

    ins = safe_float(insider_pro_score_0_100)
    if ins is None:
        ins = 50.0

    sent_ins = 0.55*sent + 0.45*ins

    nf = safe_float(insider_net_flow_value)
    if nf is not None and nf < 0:
        mcap = safe_float(metrics.get("market_cap").value if metrics.get("market_cap") else None)
        if mcap and mcap > 0:
            pct = abs(nf)/mcap
            penalty = clamp(pct/0.002 * 10, 0, 10)
        else:
            penalty = 4.0
        sent_ins = float(clamp(sent_ins - penalty, 0, 100))

    final = 0.40*valuation + 0.30*quality_health + 0.20*growth + 0.10*sent_ins
    final_int = int(round(clamp(final, 0, 100)))

    mos_v = safe_float(mos) if mos is not None else None
    verdict = "HOLD / WAIT"
    color = "#C9A227"
    if mos_v is not None and final_int > 80 and mos_v > 0.20:
        verdict = "STRONG BUY"; color = "#2ECC71"
    elif mos_v is not None and final_int > 70 and mos_v > 0.05:
        verdict = "BUY"; color = "#3CCB7F"
    elif (final_int < 50) or (mos_v is not None and mos_v < -0.15):
        verdict = "OVERVALUED / AVOID"; color = "#E74C3C"

    bullets = [
        f"Valuation: {valuation:.0f}/100 (MOS={fmt_pct(mos_v) if mos_v is not None else '—'}, Analyst gap={fmt_pct(analyst_gap) if analyst_gap is not None else '—'}).",
        f"Quality & Health: {quality_health:.0f}/100 (Current ratio={fmt_num(curr)}, Debt/Assets={fmt_pct(debt_assets)}, Op margin={fmt_pct(op_margin)}).",
        f"Growth: {growth:.0f}/100 (Revenue growth={fmt_pct(rev_g)}, FCF yield={fmt_pct(fcf_y)}).",
        f"Sentiment & Insiders: {sent_ins:.0f}/100 (News sentiment={sent:.0f}, Insider pro={ins:.0f}).",
    ]

    # If analysts are bullish but DCF MOS is very negative, flag the mismatch explicitly
    if mos_v is not None and analyst_gap is not None:
        if mos_v < -0.15 and analyst_gap > 0.10:
            bullets.append("⚠️ Mismatch: Analytici vidí upside, ale DCF vychází výrazně nadhodnoceně (MOS < -15%). Zkontroluj DCF parametry a jednotky FCF/shares.")

    warnings: List[str] = []
    if implied_fcf_growth is not None and lookback_rev_growth is not None:
        if implied_fcf_growth - lookback_rev_growth >= 0.10:
            warnings.append(
                f"Market expectations are too high: implied FCF growth ≈ {implied_fcf_growth*100:.1f}% vs revenue growth ≈ {lookback_rev_growth*100:.1f}%."
            )

    comps = {
        "valuation": float(valuation),
        "quality_health": float(quality_health),
        "growth": float(growth),
        "sentiment_insiders": float(sent_ins),
    }
    return final_int, (mos_v if mos_v is not None else float('nan')), verdict, color, bullets, (analyst_gap if analyst_gap is not None else float('nan')), comps, warnings



def fmt_price(x: Optional[float], currency: str = "$") -> str:
    """Format price-like numbers safely for UI."""
    try:
        if x is None:
            return "—"
        v = float(x)
        if v != v:  # NaN
            return "—"
        return f"{currency}{v:,.2f}"
    except Exception:
        return "—"


def dynamic_buy_conditions(
    fair_value: Optional[float],
    current_price: Optional[float],
    metrics: Dict[str, Metric],
    info: Dict[str, Any],
    score: int,
    mos: Optional[float],
    implied_fcf_growth: Optional[float],
) -> List[str]:
    conds: List[str] = []
    price = safe_float(current_price)
    fv = safe_float(fair_value)
    opm = safe_float(info.get("operatingMargins"))
    curr = safe_float(metrics.get("current_ratio").value if metrics.get("current_ratio") else None)
    lev = safe_float(metrics.get("leverage").value if metrics.get("leverage") else None)

    if price and fv:
        buy_under = fv / 1.05
        strong_under = fv / 1.20
        mos_v = safe_float(mos)
        if mos_v is None or mos_v < 0.05:
            conds.append(f"Cena musí klesnout pod {fmt_price(buy_under)} (MOS ≥ 5%).")
        else:
            conds.append(f"Udržet cenu pod {fmt_price(buy_under)} (MOS ≥ 5%); pod {fmt_price(strong_under)} je MOS ≥ 20%.")
    else:
        conds.append("Získat spolehlivou férovou cenu (DCF/targets) a nastavit buy threshold (MOS).")

    if opm is not None:
        if opm < 0.10:
            conds.append("Operating Margin musí zůstat nad 10 %.")
        else:
            conds.append(f"Operating Margin udržet nad 10 % (aktuálně {opm*100:.1f}%).")
    else:
        conds.append("Doplnit data o Operating Margin (bez něj je kvalita hůř čitelná).")

    if curr is not None and curr < 1.0:
        conds.append("Current Ratio musí být ≥ 1.0 (likvidita).")
    elif lev is not None and lev > 0.45:
        conds.append("Debt/Assets musí být < 45 % (nižší leverage).")
    else:
        if implied_fcf_growth is not None and implied_fcf_growth > 0.25:
            conds.append(f"Trh implikuje vysoký růst (~{implied_fcf_growth*100:.0f}% FCF). Potřebuješ potvrzení ve výsledcích/guidance.")
        else:
            conds.append("Potvrdit růst ve výsledcích (tržby/marže/FCF) a hlídat guidance.")

    return conds[:3]

# ---------------------------
# Scorecard & checklist
# ---------------------------
def build_scorecard(metrics: Dict[str, Metric], info: Dict[str, Any]) -> Tuple[int, Dict[str, int], List[str]]:
    """
    Explainable heuristic score 0-100, plus category breakdown and red flags.
    This is NOT financial advice; it's a structured helper.
    """
    score = 0
    cats = {"Valuation": 0, "Quality": 0, "Growth": 0, "Health": 0, "Risk": 0, "Sentiment": 0}
    flags: List[str] = []

    pe = metrics["pe"].value
    fcf_y = metrics["fcf_yield"].value
    curr = metrics["current_ratio"].value
    lev = metrics["leverage"].value
    beta = metrics["beta"].value
    rec_mean = metrics["rec_mean"].value
    rec_key = getattr(metrics.get("_rec_key"), "help", "—")

    # Valuation (25)
    v = 0
    if fcf_y is not None:
        if fcf_y >= 0.06: v += 12
        elif fcf_y >= 0.03: v += 7
        elif fcf_y >= 0.015: v += 3
        else: flags.append("Nízký FCF yield (firma může být drahá vzhledem k cashflow).")
    else:
        flags.append("Chybí FCF/FCF yield – valuace přes cashflow nejde ověřit.")
    if pe is not None:
        if pe <= 18: v += 10
        elif pe <= 30: v += 6
        elif pe <= 45: v += 2
        else: flags.append("Velmi vysoké P/E – trh čeká silný růst (vyšší riziko zklamání).")
    else:
        flags.append("Chybí P/E – může být záporný zisk nebo nedostupná data.")
    v = clamp(v, 0, 25) or 0
    cats["Valuation"] = int(v); score += int(v)

    # Quality (25) via margins proxies: use grossMargins/operatingMargins if available
    q = 0
    gm = safe_float(info.get("grossMargins"))
    om = safe_float(info.get("operatingMargins"))
    pm = safe_float(info.get("profitMargins"))
    if gm is not None:
        q += 8 if gm >= 0.45 else 5 if gm >= 0.30 else 2
    if om is not None:
        q += 10 if om >= 0.25 else 6 if om >= 0.15 else 2
    if pm is not None:
        q += 7 if pm >= 0.15 else 4 if pm >= 0.08 else 1
    if gm is None and om is None and pm is None:
        flags.append("Chybí marže (gross/operating/profit) – kvalitu byznysu nejde dobře posoudit.")
    q = clamp(q, 0, 25) or 0
    cats["Quality"] = int(q); score += int(q)

    # Growth (20): revenueGrowth, earningsGrowth if available
    g = 0
    rg = safe_float(info.get("revenueGrowth"))
    eg = safe_float(info.get("earningsGrowth"))
    if rg is not None:
        g += 10 if rg >= 0.15 else 6 if rg >= 0.07 else 2 if rg >= 0 else 0
        if rg < 0: flags.append("Tržby meziročně klesají.")
    if eg is not None:
        g += 10 if eg >= 0.15 else 6 if eg >= 0.07 else 2 if eg >= 0 else 0
        if eg < 0: flags.append("Zisk meziročně klesá.")
    if rg is None and eg is None:
        flags.append("Chybí růst (revenue/earnings) – growth profil nejde zrychleně vyhodnotit.")
    g = clamp(g, 0, 20) or 0
    cats["Growth"] = int(g); score += int(g)

    # Health (15)
    h = 0
    if curr is not None:
        h += 6 if curr >= 1.5 else 4 if curr >= 1.0 else 1
        if curr < 1.0: flags.append("Current ratio < 1 – potenciální tlak na likviditu.")
    else:
        flags.append("Chybí current ratio.")
    if lev is not None:
        h += 6 if lev <= 0.25 else 4 if lev <= 0.45 else 1
        if lev > 0.6: flags.append("Vysoké zadlužení vzhledem k aktivům.")
    else:
        flags.append("Chybí leverage (Debt/Assets).")
    fcf = metrics["fcf"].value
    if fcf is not None and fcf > 0:
        h += 3
    elif fcf is not None and fcf <= 0:
        flags.append("Negativní FCF – firma pálí cash (nebo investuje hodně).")
    h = clamp(h, 0, 15) or 0
    cats["Health"] = int(h); score += int(h)

    # Risk (10)
    r = 0
    if beta is not None:
        r += 5 if beta <= 1.1 else 3 if beta <= 1.5 else 1
    else:
        r += 2
    # high valuation risk: very high PE
    if pe is not None and pe > 45:
        r -= 2
    r = clamp(r, 0, 10) or 0
    cats["Risk"] = int(r); score += int(r)

    # Sentiment (5) via analyst recommendation
    s = 0
    if rec_mean is not None:
        # 1 strong buy, 2 buy, 3 hold, 4 underperform, 5 sell
        s += 5 if rec_mean <= 2.0 else 3 if rec_mean <= 2.8 else 1
    else:
        # fallback on recommendationKey text
        if isinstance(rec_key, str):
            k = rec_key.lower()
            if "buy" in k and "hold" not in k:
                s += 3
            elif "sell" in k:
                s += 1
            else:
                s += 2
    s = clamp(s, 0, 5) or 0
    cats["Sentiment"] = int(s); score += int(s)

    score = int(clamp(score, 0, 100) or 0)
    return score, cats, flags


def checklist_items(metrics: Dict[str, Metric], info: Dict[str, Any], fair_value: Optional[float]) -> List[Tuple[str, bool, str]]:
    price = metrics["price"].value
    fcf_y = metrics["fcf_yield"].value
    curr = metrics["current_ratio"].value
    lev = metrics["leverage"].value
    rg = safe_float(info.get("revenueGrowth"))
    om = safe_float(info.get("operatingMargins"))
    shares = metrics["shares"].value

    out: List[Tuple[str, bool, str]] = []

    # MOS
    mos_ok = False
    mos_note = "Férová cena není k dispozici."
    if price and fair_value:
        mos = safe_div(fair_value - price, price)
        mos_ok = (mos is not None and mos >= 0.2)
        mos_note = f"Margin of safety = {fmt_pct(mos)} (>= 20% je konzervativní polštář)."
    out.append(("Margin of safety (MOS)", mos_ok, mos_note))

    # FCF yield
    out.append(("FCF yield >= 3%", (fcf_y is not None and fcf_y >= 0.03),
                f"FCF yield = {fmt_pct(fcf_y)} (vyšší bývá lepší; pozor na cykličnost)."))

    # Liquidity
    out.append(("Current ratio >= 1.0", (curr is not None and curr >= 1.0),
                f"Current ratio = {fmt_num(curr)} (pod 1 může být varování u slabších bilancí)."))

    # Leverage
    out.append(("Debt/Assets <= 45%", (lev is not None and lev <= 0.45),
                f"Debt/Assets = {fmt_pct(lev)} (nižší = menší finanční riziko)."))

    # Growth
    out.append(("Revenue growth >= 0%", (rg is not None and rg >= 0.0),
                f"Revenue growth = {fmt_pct(rg)} (záporný růst = horší momentum byznysu)."))

    # Profitability
    out.append(("Operating margin >= 10%", (om is not None and om >= 0.10),
                f"Operating margin = {fmt_pct(om)} (stabilní marže často signalizují 'moat')."))

    # Shares present
    out.append(("Počet akcií dostupný", (shares is not None and shares > 0),
                "Potřebné pro přepočty na akcii (fair value, FCF/share)."))

    return out


# ---------------------------
# Watchlist & memo
# ---------------------------
def get_watchlist() -> Dict[str, Any]:
    ensure_data_dir()
    return load_json(WATCHLIST_PATH, {"items": {}})


def set_watchlist(obj: Dict[str, Any]) -> None:
    save_json(WATCHLIST_PATH, obj)


def get_memos() -> Dict[str, Any]:
    ensure_data_dir()
    return load_json(MEMOS_PATH, {"memos": {}})


def set_memos(obj: Dict[str, Any]) -> None:
    save_json(MEMOS_PATH, obj)


def export_memo_pdf(ticker: str, company: str, memo: Dict[str, Any], summary: Dict[str, Any]) -> Optional[bytes]:
    if not _HAS_PDF:
        return None
    from io import BytesIO
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter

    x = 0.75 * inch
    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, f"Investment Memo — {ticker} ({company})")
    y -= 0.35 * inch
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y -= 0.4 * inch

    def section(title: str, text: str):
        nonlocal y
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x, y, title)
        y -= 0.22 * inch
        c.setFont("Helvetica", 10)
        # wrap
        lines = []
        for para in (text or "").split("\n"):
            lines.extend(_wrap(para, 95))
            lines.append("")
        for line in lines[:]:
            if y < 0.8 * inch:
                c.showPage()
                y = height - 0.75 * inch
                c.setFont("Helvetica", 10)
            c.drawString(x, y, line[:120])
            y -= 0.16 * inch
        y -= 0.15 * inch

    def _wrap(s: str, width_chars: int) -> List[str]:
        words = s.split()
        out = []
        cur = ""
        for w in words:
            if len(cur) + 1 + len(w) <= width_chars:
                cur = (cur + " " + w).strip()
            else:
                out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        if not out:
            out = [""]
        return out

    # Summary table-ish
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x, y, "Key numbers")
    y -= 0.25 * inch
    c.setFont("Helvetica", 10)
    for k, v in summary.items():
        if y < 0.8 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica", 10)
        c.drawString(x, y, f"{k}: {v}")
        y -= 0.16 * inch
    y -= 0.2 * inch

    section("Thesis (why this wins)", memo.get("thesis", ""))
    section("Key drivers", memo.get("drivers", ""))
    section("Risks & what to watch", memo.get("risks", ""))
    section("Catalysts", memo.get("catalysts", ""))
    section("Buy conditions / price targets", memo.get("buy_conditions", ""))
    section("Notes", memo.get("notes", ""))

    c.save()
    return buf.getvalue()


# ---------------------------
# UI helpers
# ---------------------------
def metric_card(m: Metric) -> str:
    if m.fmt == "pct":
        return fmt_pct(m.value)
    if m.fmt == "money":
        return fmt_money(m.value)
    return fmt_num(m.value)



def show_metric_with_help(m: Metric):
    col1, col2 = st.columns([1, 2])
    with col1:
        label = f"**{m.label}**"
        if getattr(m, "source", ""):
            label += f"  ·  `{m.source}`"
        st.write(label)
        st.write(metric_card(m))
    with col2:
        st.caption(m.help)


def pick_interval(period: str) -> str:
    # sensible defaults for yfinance limits
    if period in ("1d", "5d"):
        return "5m"
    if period in ("1mo", "3mo"):
        return "1h"
    if period in ("6mo", "1y", "2y"):
        return "1d"
    if period in ("5y", "10y", "max"):
        return "1wk"
    return "1d"



def compute_insider_pro_signal(insider_df, current_price=None, market_cap=None, lookback_days=180):
    """
    Returns: (score_0_100, stats_dict, notes_list)

    score is heuristic:
      - role-weighted net flow (value) with sensible scaling
      - cluster buying/selling detection
      - separates grants/awards and sell-to-cover/10b5-1 (if tags exist)
    """
    notes = []
    stats = {
        "buy_count": 0,
        "sell_count": 0,
        "grant_count": 0,
        "other_count": 0,
        "weighted_buy_value": 0.0,
        "weighted_sell_value": 0.0,
        "weighted_net_value": 0.0,
        "cluster_buying": False,
        "cluster_selling": False,
        "unique_buyers": 0,
        "unique_sellers": 0,
        "window_days": lookback_days,
    }

    # Defensive: if no data
    if insider_df is None or getattr(insider_df, "empty", True):
        return 50, stats, ["No insider transaction data available."]

    df = insider_df.copy()

    # Normalize columns
    def _col(name_variants):
        for n in name_variants:
            if n in df.columns:
                return n
        return None

    col_insider = _col(["Insider", "insider", "Name", "name"])
    col_pos = _col(["Position", "position", "Title", "title"])
    col_text = _col(["Text", "text", "Description", "description"])
    col_trx = _col(["Transaction", "transaction", "Type", "type"])
    col_tag = _col(["Tag", "tag", "Category", "category"])
    col_value = _col(["Value", "value", "Amount", "amount"])
    col_shares = _col(["Shares", "shares", "Qty", "qty"])
    col_date = _col(["Start Date", "StartDate", "Date", "date", "Filing Date", "filingDate"])

    # Parse date and filter lookback
    if col_date:
        df[col_date] = pd.to_datetime(df[col_date], errors="coerce")
        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=int(lookback_days))
        df = df[df[col_date].notna() & (df[col_date] >= cutoff)]
        if df.empty:
            return 50, stats, [f"No insider transactions in last {lookback_days} days."]

    # Helpers
    def role_weight(pos: str) -> float:
        p = (pos or "").lower()
        if "chief executive" in p or p.startswith("ceo") or " ceo" in p:
            return 3.0
        if "chief financial" in p or p.startswith("cfo") or " cfo" in p:
            return 2.5
        if "chief operating" in p or p.startswith("coo") or " coo" in p:
            return 2.2
        if "president" in p:
            return 2.0
        if "general counsel" in p:
            return 1.8
        if "director" in p:
            return 1.5
        if "officer" in p:
            return 1.3
        return 1.0

    def classify(row) -> str:
        # Prefer Tag if exists
        tag = (row.get(col_tag) if col_tag else "") or ""
        tag_l = str(tag).lower()
        if any(k in tag_l for k in ["grant", "award", "rsu", "vesting"]):
            return "grant"
        if any(k in tag_l for k in ["sell-to-cover", "tax", "withhold"]):
            return "sell_tax"
        if "10b5" in tag_l or "10b5-1" in tag_l:
            return "sell_plan"
        if "open market buy" in tag_l:
            return "buy"
        if "open market sell" in tag_l:
            return "sell"

        trx = (row.get(col_trx) if col_trx else "") or ""
        txt = (row.get(col_text) if col_text else "") or ""
        s = f"{trx} {txt}".lower()

        if any(k in s for k in ["stock award", "award", "grant", "rsu", "restricted stock"]):
            return "grant"
        if any(k in s for k in ["sell to cover", "sell-to-cover", "tax", "withholding"]):
            return "sell_tax"
        if "10b5" in s or "10b5-1" in s:
            return "sell_plan"
        if any(k in s for k in ["purchase", "buy", "acquire"]) and not any(k in s for k in ["sell", "sale"]):
            return "buy"
        if any(k in s for k in ["sell", "sale", "dispose"]):
            return "sell"
        return "other"

    def get_value(row) -> float:
        v = row.get(col_value) if col_value else None
        try:
            if v is None:
                raise ValueError()
            if isinstance(v, str) and v.strip() == "":
                raise ValueError()
            vv = float(v)
            if vv != vv:
                raise ValueError()
            return vv
        except Exception:
            # Approximate from shares * price if available
            try:
                sh = float(row.get(col_shares) or 0.0) if col_shares else 0.0
                if current_price and sh:
                    return float(current_price) * sh
            except Exception:
                pass
            return 0.0

    # Aggregate
    buyers=set()
    sellers=set()
    weighted_buy=0.0
    weighted_sell=0.0
    buy_count=sell_count=grant_count=other_count=0

    for _, r in df.iterrows():
        cls = classify(r)
        pos = str(r.get(col_pos) or "")
        w = role_weight(pos)
        name = str(r.get(col_insider) or "").strip() if col_insider else ""

        if cls == "buy":
            buy_count += 1
            buyers.add(name or f"row{_}")
            weighted_buy += w * get_value(r)
        elif cls in ("sell", "sell_tax", "sell_plan"):
            sell_count += 1
            sellers.add(name or f"row{_}")
            weighted_sell += w * get_value(r)
        elif cls == "grant":
            grant_count += 1
        else:
            other_count += 1

    stats.update({
        "buy_count": buy_count,
        "sell_count": sell_count,
        "grant_count": grant_count,
        "other_count": other_count,
        "unique_buyers": len([b for b in buyers if b]),
        "unique_sellers": len([s for s in sellers if s]),
        "weighted_buy_value": float(weighted_buy),
        "weighted_sell_value": float(weighted_sell),
        "weighted_net_value": float(weighted_buy - weighted_sell),
    })

    # Cluster detection: many unique insiders in short time window
    # Use last 30 days within filtered df if date exists
    cluster_window_days = 30
    if col_date:
        recent_cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=cluster_window_days)
        recent = df[df[col_date] >= recent_cutoff].copy()
    else:
        recent = df.copy()

    # compute unique buyers/sellers in recent window
    if not recent.empty:
        rb=set(); rs=set()
        for _, r in recent.iterrows():
            cls = classify(r)
            nm = str(r.get(col_insider) or "").strip() if col_insider else ""
            if cls=="buy":
                rb.add(nm or f"row{_}")
            elif cls in ("sell","sell_tax","sell_plan"):
                rs.add(nm or f"row{_}")
        # thresholds (tuned for megacaps)
        if len(rb) >= 3 and buy_count >= 3:
            stats["cluster_buying"] = True
            notes.append(f"Cluster buying: {len(rb)} unique buyers in last {cluster_window_days} days.")
        if len(rs) >= 5 and sell_count >= 8:
            stats["cluster_selling"] = True
            notes.append(f"Cluster selling: {len(rs)} unique sellers in last {cluster_window_days} days (many sells are planned/tax).")

    # Score construction
    score = 50.0

    # Scale net flow vs market cap if available
    net = stats["weighted_net_value"]
    scale = None
    if market_cap and market_cap > 0:
        # 0.1% of market cap is "meaningful" for net flow
        scale = market_cap * 0.001
    else:
        # fallback scale
        scale = max(abs(net), 1.0)

    # net contribution capped
    net_contrib = 0.0
    if scale and scale > 0:
        net_contrib = max(-25.0, min(25.0, (net / scale) * 25.0))
    score += net_contrib

    if stats["cluster_buying"]:
        score += 8.0
    if stats["cluster_selling"]:
        score -= 8.0

    # If only grants and no meaningful trades, dampen to neutral
    if buy_count == 0 and sell_count == 0:
        score = 50.0
        notes.append("Only grants/awards detected; insider signal treated as neutral.")

    # Keep bounds
    score = max(0.0, min(100.0, score))

    # Additional notes
    if stats["weighted_sell_value"] > 0 and stats["weighted_buy_value"] == 0:
        notes.append("Net insider flow is negative (role-weighted), but many sells can be planned/tax-related.")
    if stats["unique_buyers"] > 0 and stats["weighted_buy_value"] > 0:
        notes.append("Open-market buying (especially by senior roles) is typically a stronger signal than selling.")

    return float(round(score, 1)), stats, notes


# ---------------------------
# Main App
# ---------------------------

def _hide_sidebar_once():
    """Mark sidebar to be hidden on next render (mobile UX)."""
    try:
        st.session_state["_hide_sidebar"] = True
    except Exception:
        pass

def main():
    # ---- session init (must happen before widgets are created) ----
    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "NVDA"
    if "ticker_input" not in st.session_state:
        st.session_state["ticker_input"] = st.session_state["ticker"]

    st.set_page_config(page_title=f"{APP_NAME} {APP_VERSION}", layout="wide", initial_sidebar_state="collapsed")

    # ---- Mobile UX: hide sidebar after submitting ticker ----
    if st.session_state.get("_hide_sidebar"):
        st.markdown(
            """<style>
section[data-testid="stSidebar"] {display: none;}
</style>""",
            unsafe_allow_html=True,
        )
        # Button to re-open menu (works on mobile)
        if st.button("☰ Show menu"):
            st.session_state["_hide_sidebar"] = False
            st.rerun()


        # ---- Mobile UX: hide sidebar after submitting ticker ----
        if st.session_state.get("_hide_sidebar"):
            st.markdown(
                """<style>
    section[data-testid="stSidebar"] {display: none;}
    </style>""",
                unsafe_allow_html=True,
            )
            # Small button to re-open menu (works on mobile)
            if st.button("☰ Show menu"):
                st.session_state["_hide_sidebar"] = False
                st.rerun()


    # --- Responsive tweaks (mobile) ---
    st.markdown(
        """
    <style>
    /* Smaller metrics + tighter spacing on phones */
    @media (max-width: 768px) {
      div[data-testid="metric-container"] { padding: 6px 8px; }
      div[data-testid="stMetricValue"] { font-size: 1.15rem; line-height: 1.2; }
      div[data-testid="stMetricLabel"] { font-size: 0.85rem; }
      div[data-testid="stMetricDelta"] { font-size: 0.80rem; }
    }
    /* Reduce top padding a bit */
    .block-container { padding-top: 1.25rem; }
    </style>
        """,
        unsafe_allow_html=True,
    )

    ensure_data_dir()

    st.markdown(f"# 📊 {APP_NAME} {APP_VERSION}")

    # Sidebar controls
    with st.sidebar:
        st.markdown("## Nastavení")
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "NVDA"), key="ticker", on_change=_hide_sidebar_once).strip().upper()
        period = st.selectbox("Time frame", options=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=4)
        interval = st.selectbox("Interval", options=["5m", "15m", "30m", "1h", "1d", "1wk"], index=["5m","15m","30m","1h","1d","1wk"].index(pick_interval(period)))
        st.markdown("---")
        show_debug = st.checkbox("Zobrazit debug info", value=False)
        st.caption("Tip: delší time frame = stabilnější obrázek; kratší = lepší pro timing.")

    if not ticker:
        st.info("Zadej ticker.")
        return

    # Fetch core data
    info = fetch_ticker_info(ticker)
    objects = fetch_ticker_objects(ticker)
    hist = fetch_history(ticker, period=period, interval=interval)

    metrics = compute_metrics(ticker, info, objects)
    # Optional: enrich missing ratios with FMP (if FMP_API_KEY is set)
    metrics, fmp_notes = enrich_metrics_with_fmp(ticker, metrics)
    company = info.get("longName") or info.get("shortName") or ticker

    # DCF inputs
    with st.sidebar:
        st.markdown("## DCF nastavení")
        g_yrs = st.slider(
            "Horizon (years)",
            3, 10, 5, 1,
            help="Kolik let dopředu modelujeme firmu detailně. 5 let je standard. 7–10 let dává smysl u firem s dlouhým růstovým runway (např. AI/tech).",
        )
        g_rate = st.slider(
            "FCF growth (base)",
            -20.0, 60.0, 12.0, 1.0,
            help=(
                "Očekávaný průměrný meziroční růst Free Cash Flow během zvoleného horizontu. "
                "Např. 12 znamená ~12% růst FCF za rok (zjednodušený scénář). "
                "U growth firem je to nejcitlivější parametr."
            ),
        ) / 100.0
        disc = st.slider(
            "Discount rate",
            6.0, 18.0, 10.0, 0.5,
            help=(
                "Diskontní sazba = požadovaná návratnost / míra rizika. "
                "Vyšší = konzervativnější (víc 'trestá' budoucnost). "
                "Mega-cap quality často 8–9%, běžně 10%, rizikovější 12%+."
            ),
        ) / 100.0
        term_g = st.slider(
            "Terminal growth",
            0.0, 6.0, 3.0, 0.5,
            help=(
                "Dlouhodobý růst po skončení horizontu (""na věčnost"" v modelu). "
                "Typicky 2–3%. Vyšší hodnoty jsou agresivní."
            ),
        ) / 100.0
        use_exit_mult = st.checkbox(
            "Use exit multiple",
            value=False,
            help="Místo terminal growth použije násobek (např. 20× FCF) pro výpočet terminální hodnoty. Často stabilnější než terminal growth u growth firem.",
        )
        exit_mult = None
        if use_exit_mult:
            exit_mult = st.slider(
                "Exit multiple (FCF)",
                8, 40, 20, 1,
                help="Kolikanásobek ročního FCF v posledním roce horizontu. Vyšší násobek = vyšší férová cena. Orientačně: stabilní firmy 12–20, růstové 18–30 (záleží na režimu trhu).",
            )

        with st.expander("Co znamenají tyto parametry?", expanded=False):
            st.write("**FCF growth**: průměrný meziroční růst free cash flow v horizontu (např. 12%/rok).")
            st.write("**Discount rate**: požadovaná návratnost (vyšší = konzervativnější).")
            st.write("**Terminal growth / Exit multiple**: způsob, jak ocenit firmu po horizontu (dlouhodobý růst vs násobek).")
            st.caption("DCF je citlivý na předpoklady. Proto je nejlepší brát výsledek jako scénář (bear/base/bull), ne jako jediné číslo.")

    fair_value = dcf_fair_value_per_share(
        fcf_ttm=metrics["fcf"].value,
        shares=metrics["shares"].value,
        growth_yrs=g_yrs,
        growth_rate=g_rate,
        discount_rate=disc,
        terminal_growth=term_g,
        exit_multiple=exit_mult,
    )

    implied_growth = reverse_dcf_implied_growth(
        price=metrics["price"].value,
        fcf_ttm=metrics["fcf"].value,
        shares=metrics["shares"].value,
        growth_yrs=g_yrs,
        discount_rate=disc,
        terminal_growth=term_g,
        exit_multiple=exit_mult,
    )

    # Top header cards (multi fair value)
    # Priority for "main" fair value:
    # 1) Analyst target mean/median (Yahoo)
    # 2) Internal DCF fair value (fallback)
    analyst_mean = metrics.get("target_mean").value if "target_mean" in metrics else None
    analyst_median = metrics.get("target_median").value if "target_median" in metrics else None

    main_fair = analyst_mean or analyst_median or fair_value
    main_src = "Analyst (Yahoo mean)" if analyst_mean else ("Analyst (Yahoo median)" if analyst_median else "DCF (internal)")

    # Optional manual reference band for NVDA (editable in code)
    ref_band_low = None
    ref_band_high = None
    if (ticker or "").upper().strip() == "NVDA":
        ref_band_low = 140.0
        ref_band_high = 350.0

    
    # Header metrics (price + multiple fair values + ATH)
    analyst_low = metrics.get("target_low").value if metrics.get("target_low") else None
    analyst_high = metrics.get("target_high").value if metrics.get("target_high") else None

    ath = get_all_time_high(ticker) if ticker else None

    # MOS based on DCF fair value (if available)
    mos_dcf = None
    if metrics["price"].value and fair_value:
        mos_dcf = safe_div(fair_value - metrics["price"].value, metrics["price"].value)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Firma (YF)", metrics["company"].value or ticker)
    with c2:
        st.metric("Cena (YF)", fmt_money(metrics["price"].value))
    with c3:
        # Analyst fair value shown separately to avoid confusion with DCF MOS
        a_fv = analyst_mean or analyst_median
        delta_a = safe_div(a_fv - metrics["price"].value, metrics["price"].value) if (metrics["price"].value and a_fv) else None
        st.metric("🎯 Férová cena (Analytici · YF)", fmt_money(a_fv) if a_fv else "—", delta=fmt_pct(delta_a) if delta_a is not None else None)
        # Show the broadest band available
        band = None
        if analyst_low and analyst_high:
            band = f"{fmt_money(analyst_low)} – {fmt_money(analyst_high)}"
        elif analyst_median and analyst_mean:
            lo = min(analyst_mean, analyst_median); hi = max(analyst_mean, analyst_median)
            band = f"{fmt_money(lo)} – {fmt_money(hi)}"
        elif ref_band_low and ref_band_high:
            band = f"{fmt_money(ref_band_low)} – {fmt_money(ref_band_high)}"
        st.caption(f"Pásmo: {band if band else '—'}")
    with c4:
        delta_d = safe_div(fair_value - metrics["price"].value, metrics["price"].value) if (metrics["price"].value and fair_value) else None
        st.metric("🧮 Férová cena (DCF)", fmt_money(fair_value) if fair_value else "—", delta=fmt_pct(delta_d) if delta_d is not None else None)
        st.caption(f"MOS (DCF): {fmt_pct(mos_dcf) if mos_dcf is not None else '—'}")
        buy_under = (fair_value * 0.95) if fair_value else None
        strong_buy_under = (fair_value * 0.80) if fair_value else None
        st.caption(f"Buy-under (MOS ≥ 5%): {fmt_money(buy_under) if buy_under else '—'} · Strong (MOS ≥ 20%): {fmt_money(strong_buy_under) if strong_buy_under else '—'}")
    with c5:
        st.metric("🏔️ ATH", fmt_money(ath) if ath else "—")
        ath_gap = None
        if ath and metrics.get("price") and metrics["price"].value:
            ath_gap = safe_div(ath - metrics["price"].value, metrics["price"].value)
        st.caption("All‑time high (Yahoo, max denní High).")
        st.caption(f"Do ATH: {fmt_pct(ath_gap) if ath_gap is not None else '—'}")


    with st.expander("Férové ceny – detaily a vysvětlivky", expanded=False):
        st.markdown("**Analyst targets (Yahoo)** – cílové ceny analytiků; citlivé na sentiment a očekávání.")
        st.write(
            f"Mean: {fmt_money(analyst_mean) if analyst_mean else '—'} | "
            f"Median: {fmt_money(analyst_median) if analyst_median else '—'}"
        )

        st.markdown("**DCF (internal)** – jednoduchý FCF DCF model (parametry nastavuješ v aplikaci).")
        st.write(f"DCF férová cena: {fmt_money(fair_value) if fair_value else '—'}")

        st.markdown("**Mean reversion (orientačně)** – násobky jako teploměr očekávání, ne přesná férovka.")
        pe = metrics.get("pe").value if "pe" in metrics else None
        st.write(f"P/E (TTM): {pe:.1f}×" if pe is not None else "P/E (TTM): —")

        if (ticker or "").upper().strip() == "NVDA":
            st.markdown("**Reference pro NVDA (ručně zadané)**")
            st.write("Analyst consensus ~ $260–265 | Bull ~ $350 | Bear ~ $140")
            st.write("Morningstar FV ~ $240 | AlphaSpread (DCF) ~ $106")
            st.caption("Pozn.: Tyto reference jsou ručně zadané; aktualizuj je dle potřeby.")

    # Leverage card (kept as quick risk indicator)
    # guardrails: show "—" if nonsensical
    lev = metrics["leverage"].value
    lev_display = fmt_pct(lev) if lev is not None and 0 <= lev <= 5 else "—"
    # We'll show it under the header as a small line instead of a 5th card
    st.caption(f"Leverage (Debt/Assets): {lev_display} — Dluh/aktiva. Pokud Yahoo vrátí prázdná aktiva, hodnota se schová.")

    if show_debug:
        st.sidebar.markdown("### Debug")
        st.sidebar.write("yfinance:", yf.__version__)
        st.sidebar.write("ticker:", ticker)
        st.sidebar.write("period/interval:", period, interval)
        st.sidebar.write("info keys:", len(info.keys()))
        st.sidebar.write("hist rows:", len(hist) if isinstance(hist, pd.DataFrame) else None)
        try:
            fcf_dbg = derive_fcf_ttm(objects)
        except Exception:
            fcf_dbg = None
        st.sidebar.write("FCF (derived):", fcf_dbg)
        st.sidebar.write("sharesOutstanding:", info.get("sharesOutstanding"))
        st.sidebar.write("marketCap:", info.get("marketCap"))
        st.sidebar.write("currentPrice:", info.get("currentPrice") or info.get("regularMarketPrice"))

    tabs = st.tabs([
        "📈 Graf & TA",
        "📊 Fundamenty",
        "📰 Novinky & Sentiment",
        "🧑‍💼 Insider Trading",
        "🧩 Peers",
        "✅ Dashboard (Scorecard)",
        "🧾 Investment Memo & Watchlist",
    ])
    # Cross-tab shared signals (defaults; overwritten if data available)
    sentiment_score_num = 50.0
    insider_pro_score = 50.0
    insider_net_flow_value = None


    # ---------------- Tab 1: Chart & TA ----------------
    with tabs[0]:
        st.subheader("Cena a technický kontext")
        if hist.empty:
            st.warning("Nepodařilo se stáhnout historická data (Yahoo může blokovat nebo ticker neexistuje).")
        else:
            # basic chart
            df = hist.copy()
            if "Close" not in df.columns:
                st.warning(f"Chybí sloupec Close v historických datech. Dostupné sloupce: {list(df.columns)}")
                st.dataframe(df.head(10))
            else:
                df = df.dropna(subset=["Close"]) 
                df["SMA20"] = df["Close"].rolling(20).mean()
                df["SMA50"] = df["Close"].rolling(50).mean()
                st.line_chart(df.set_index("Datetime")[["Close", "SMA20", "SMA50"]])

                # Simple TA notes
                last = df["Close"].iloc[-1]
                sma20 = df["SMA20"].iloc[-1]
                sma50 = df["SMA50"].iloc[-1]
                colA, colB, colC = st.columns(3)
                with colA:
                    st.metric("Close", fmt_money(last))
                with colB:
                    st.metric("SMA20", fmt_money(sma20), delta=fmt_pct(safe_div(last - sma20, sma20)) if sma20 else None)
                with colC:
                    st.metric("SMA50", fmt_money(sma50), delta=fmt_pct(safe_div(last - sma50, sma50)) if sma50 else None)
                st.caption("SMA: jednoduché klouzavé průměry. Nad SMA = trend spíše pozitivní, pod = negativní (zjednodušeně).")
    # ---------------- Tab 2: Fundamentals ----------------
    with tabs[1]:
        st.subheader("Základní fundamenty (rychlý přehled)")
        left, right = st.columns([1, 1])

        with left:
            st.markdown("### Valuace")
            for k in ["market_cap", "enterprise_value", "pe", "forward_pe", "pb", "fcf", "fcf_yield"]:
                show_metric_with_help(metrics[k])

        with right:
            st.markdown("### Zdraví & vlastníci")
            for k in ["cash", "debt", "current_ratio", "leverage", "held_insiders", "held_institutions"]:
                show_metric_with_help(metrics[k])

        st.markdown("### Analytici (Yahoo)")
        rec_key = metrics["_rec_key"].help
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Recommendation", rec_key if rec_key else "—")
        with a2:
            st.metric("Recommendation mean", fmt_num(metrics["rec_mean"].value))
        with a3:
            # prefer mean target, else median
            targ = metrics["target_mean"].value or metrics["target_median"].value
            st.metric("Cílová cena", fmt_money(targ), delta=fmt_pct(safe_div(targ - metrics["price"].value, metrics["price"].value)) if targ and metrics["price"].value else None)

        st.caption("Pozn.: Yahoo data nejsou vždy kompletní. U některých firem mohou chybět marže/FCF apod.")

    # ---------------- Tab 3: News ----------------
    with tabs[2]:
        st.subheader("Poslední zprávy (Yahoo)")
        news = fetch_news(ticker, limit=12)
        company_name = ""
        try:
            company_name = (metrics.get("company").value if isinstance(metrics, dict) and "company" in metrics else "") or ""
        except Exception:
            company_name = ""

        if not news:
            st.warning("Yahoo Finance nevrátilo žádné čitelné titulky zpráv. (Někdy blokuje/vrací prázdná data.)")
            relevant_news, other_news = [], []
        else:
            relevant_news, other_news = split_relevant_news(news, ticker=ticker, company=company_name)

            st.caption(f"Relevance filtr: {len(relevant_news)} relevantních, {len(other_news)} ostatních (market/šum).")
            show_noise = st.checkbox("Zobrazit i nerelevantní/market zprávy", value=False)

            display_news = list(relevant_news) + (list(other_news) if show_noise else [])
            for i, n in enumerate(display_news, start=1):
                title = n.get("title") or ""
                pub = n.get("publisher") or "Neznámý zdroj"
                url = n.get("url")
                when = n.get("published")
                rel = n.get("rel_label") or ""
                badge = "✅" if rel == "RELEVANT" else ("🟨" if rel == "MAYBE" else "⚪")
                with st.expander(f"{badge} {i}. {title}", expanded=False):
                    st.write(f"**Zdroj:** {pub}  ·  **Relevance:** {rel} (score {n.get('rel_score', 0)})")
                    if when:
                        st.write(f"Publikováno: {when}")
                    if url:
                        st.link_button("Otevřít článek", url)
                    else:
                        st.caption("Odkaz není k dispozici (Yahoo někdy neposílá URL).")
        st.markdown("---")
        st.subheader("AI / Fallback sentiment")
        st.caption("Standardně se používá rychlá heuristika z titulků. Volitelně můžeš spustit AI analýzu (Gemini) tlačítkem – šetří kvóty a brání 429.")

        fb_score, fb_label, fb_conf, fb_reasons, fb_highlights = headline_sentiment_explain((relevant_news or news), max_items=10)

        ai_enabled = _gemini_available()
        colA, colB = st.columns([2, 1])
        with colA:
            use_ai = st.toggle("Použít AI sentiment (Gemini)", value=False, disabled=not ai_enabled, help="AI se nespouští automaticky – jen po kliknutí na tlačítko, aby se nevyčerpaly kvóty.")
        with colB:
            run_ai = st.button("Spustit AI analýzu", disabled=not (use_ai and ai_enabled))

        # Session cache for AI result (per ticker)
        cache_key = f"ai_sentiment::{ticker}::{hash(tuple((it.get('title') or '') for it in (news or [])[:10]))}"
        if "ai_sentiment_cache" not in st.session_state:
            st.session_state["ai_sentiment_cache"] = {}

        ai_score = None
        ai_label = None
        ai_conf = None
        ai_bullets: List[str] = []

        if use_ai and run_ai:
            score, label, conf, bullets = gemini_sentiment_from_headlines(news, max_items=10)
            st.session_state["ai_sentiment_cache"][cache_key] = (score, label, conf, bullets)

        if use_ai and cache_key in st.session_state["ai_sentiment_cache"]:
            ai_score, ai_label, ai_conf, ai_bullets = st.session_state["ai_sentiment_cache"][cache_key]

        if use_ai and ai_score is not None:
            st.metric("Sentiment (AI)", ai_label)
            st.caption(f"Skóre: {ai_score}/100 • Confidence: {int((ai_conf or 0)*100)}% • Model: {GEMINI_MODEL}")
            if ai_bullets:
                st.markdown("**Důvody (AI):**")
                for b in ai_bullets:
                    st.markdown(f"- {b}")
            sentiment_score_num = float(ai_score)
        else:
            st.metric("Sentiment (fallback)", fb_label)
            st.caption(f"Skóre: {fb_score}/100 • Confidence: {int(fb_conf*100)}% • Heuristika z titulků (negace + váha podle stáří).")
            with st.expander("Proč to vyšlo takhle (fallback)", expanded=False):
                st.markdown("**Důvody:**")
                for r in (fb_reasons or []):
                    st.write(f"- {r}")
                pos = (fb_highlights or {}).get("positive") or []
                neg = (fb_highlights or {}).get("negative") or []
                if pos:
                    st.markdown("**Top pozitivní titulky:**")
                    for t in pos:
                        st.write(f"- {t}")
                if neg:
                    st.markdown("**Top negativní titulky:**")
                    for t in neg:
                        st.write(f"- {t}")
                if not pos and not neg:
                    st.caption("Žádné výrazně pozitivní/negativní titulky v rámci heuristiky.")
            sentiment_score_num = float(fb_score)


    # ---------------- Tab 4: Insiders ----------------
    with tabs[3]:
        st.subheader("Insider Trading — transakce vedení")
        df_raw = objects.get("insider_transactions")
        df = normalize_insiders_df(df_raw)
        if df.empty:
            st.info("Insider transakce nejsou dostupné (Yahoo to často nemá pro všechny tickery).")
        else:
            st.caption("Pozn.: Část prodejů bývá automatická (10b5-1 plán / sell-to-cover kvůli daním při vestingu). Proto oddělujeme typy.")

            # Filters
            colf1, colf2, colf3 = st.columns([1, 1, 2])
            with colf1:
                excl_grants = st.checkbox("Ignorovat granty/RSU", value=True, help="Grant/Award nejsou reálný nákup (jen přidělení/vesting).")
            with colf2:
                excl_tax = st.checkbox("Ignorovat sell-to-cover", value=True, help="Sell-to-cover jsou prodeje kvůli daním/withholdingu, často automatické.")
            with colf3:
                st.write("")
                st.caption("Tip: Pro signál sleduj hlavně 'Open market' BUY/Sell mimo granty a daně.")

            df_view = df.copy()
            if excl_grants:
                df_view = df_view[df_view["Type"] != "Grant/Award"]
            if excl_tax:
                df_view = df_view[~((df_view["Type"] == "Sell") & (df_view.get("Tag", "") == "Sell-to-cover (tax)"))]

            # summary
            buy_df = df_view[df_view["Type"] == "Buy"]
            sell_df = df_view[df_view["Type"] == "Sell"]
            grant_df = df[df["Type"] == "Grant/Award"]
            sell_tax_df = df[(df["Type"] == "Sell") & (df.get("Tag", "") == "Sell-to-cover (tax)")]
            sell_10b5_df = df[(df["Type"] == "Sell") & (df.get("Tag", "") == "10b5-1 planned")]

            def sum_col(d: pd.DataFrame, col: str) -> float:
                if col not in d.columns or d.empty:
                    return 0.0
                return float(pd.to_numeric(d[col], errors="coerce").fillna(0).sum())

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Nákupy (count)", len(buy_df))
                st.caption("Největší váhu mají open-market nákupy (mimo granty).")
            with c2:
                st.metric("Prodeje (count)", len(sell_df))
                st.caption("Část prodejů může být 10b5-1 plán nebo sell-to-cover.")
            with c3:
                st.metric("Granty/Awardy", len(grant_df))
                st.caption("RSU/award často nejsou 'nákup' – jen přidělení/vesting.")
            with c4:
                net_value = sum_col(buy_df, "Value") - sum_col(sell_df, "Value")

                # Expose for dashboard
                insider_net_flow_value = net_value

                # Pro insider signal (cluster + role-weighted net flow)
                insider_pro_score, insider_pro_stats, insider_pro_notes = compute_insider_pro_signal(
                    df_view,
                    market_cap=metrics.get("market_cap").value if metrics.get("market_cap") else None,
                    lookback_days=90,
                )
                st.metric("Net flow (Value)", fmt_money(net_value))
                st.caption("Buy value − Sell value (kde Yahoo value poskytuje).")


            st.markdown("#### Insider PRO signál (cluster + role-weighted net flow)")
            st.progress(int(clamp(insider_pro_score,0,100)))
            st.write(f"**Insider PRO skóre:** {insider_pro_score:.0f}/100")
            with st.expander("Detaily výpočtu (pro)", expanded=False):
                st.json(insider_pro_stats)
                for n in insider_pro_notes:
                    st.write("• " + n)

            st.markdown("#### Kontext pro prodeje")
            c5, c6, c7 = st.columns(3)
            with c5:
                st.metric("Sell-to-cover (count)", len(sell_tax_df))
            with c6:
                st.metric("10b5-1 plán (count)", len(sell_10b5_df))
            with c7:
                other_sells = max(len(df[df["Type"] == "Sell"]) - len(sell_tax_df) - len(sell_10b5_df), 0)
                st.metric("Ostatní prodeje (count)", other_sells)

            # Breakdown by insider
            st.markdown("### Kdo kupuje/prodává")
            if "Insider" in df.columns:
                by = df.groupby(["Insider", "Type"]).size().unstack(fill_value=0)
                # Ensure stable columns
                for col in ["Buy", "Sell", "Grant/Award", "Other", "Unknown"]:
                    if col not in by.columns:
                        by[col] = 0
                by = by[[c for c in ["Buy", "Sell", "Grant/Award", "Other", "Unknown"] if c in by.columns]]
                st.dataframe(by.sort_values(["Buy", "Sell"], ascending=False), width='stretch')

            st.markdown("### Rozpad podle typu (Tag)")
            if "Tag" in df.columns:
                tags = df[df["Type"].isin(["Buy", "Sell", "Grant/Award"])].copy()
                tag_counts = tags.groupby(["Type", "Tag"]).size().reset_index(name="Count")
                st.dataframe(tag_counts.sort_values(["Type", "Count"], ascending=[True, False]), width='stretch')

            st.markdown("### Detailní seznam")
            show_cols = [c for c in ["Start Date", "Insider", "Position", "Type", "Tag", "Shares", "Value", "Text"] if c in df.columns]
            st.dataframe(df.sort_values("Start Date", ascending=False)[show_cols], width='stretch')

    # ---------------- Tab 5: Peers ----------------
    with tabs[4]:
        st.subheader("Peers & relativní srovnání")
        st.caption("Automatické peers je bez specializovaného zdroje nepřesné; zde můžeš zadat vlastní seznam peer tickerů.")
        default_peers = ""
        # Quick heuristics for popular tickers
        heuristic = {
            "NVDA": "AMD, INTC, AVGO, TSM, QCOM",
            "AAPL": "MSFT, GOOGL, AMZN, META",
            "MSFT": "AAPL, GOOGL, AMZN, META",
            "GOOGL": "META, MSFT, AMZN, AAPL",
            "AMZN": "MSFT, GOOGL, AAPL, META",
            "TSLA": "GM, F, RIVN, LCID",
        }
        if ticker in heuristic:
            default_peers = heuristic[ticker]
        peers_str = st.text_input("Peers (comma separated)", value=default_peers)
        peers = [p.strip().upper() for p in peers_str.split(",") if p.strip()]
        peers = [p for p in peers if p != ticker][:10]
        if not peers:
            st.info("Zadej aspoň 1 peer ticker (např. AMD, INTC…).")
        else:
            rows = []
            all_tickers = [ticker] + peers
            for tkr in all_tickers:
                inf = fetch_ticker_info(tkr)
                objs = fetch_ticker_objects(tkr)
                m = compute_metrics(tkr, inf, objs)
                rows.append({
                    "Ticker": tkr,
                    "Price": m["price"].value,
                    "MarketCap": m["market_cap"].value,
                    "P/E": m["pe"].value,
                    "Fwd P/E": m["forward_pe"].value,
                    "FCF Yield": m["fcf_yield"].value,
                    "Op Margin": safe_float(inf.get("operatingMargins")),
                    "Rev Growth": safe_float(inf.get("revenueGrowth")),
                })
            dfp = pd.DataFrame(rows)
            # nicer formatting in UI
            st.dataframe(dfp, width='stretch')
            st.caption("Srovnání je orientační. Některé metriky mohou být prázdné kvůli Yahoo datům.")

    # ---------------- Tab 6: Dashboard ----------------
    with tabs[5]:
        st.subheader("Dashboard (stock-picking signál)")

        lookback_rev_growth = safe_float(info.get("revenueGrowth"))
        final_score, mos_val, verdict, color, bullets, analyst_gap, comps, warns = compute_weighted_signal(
            fair_value=fair_value,
            current_price=metrics.get("price").value if metrics.get("price") else None,
            metrics=metrics,
            info=info,
            sentiment_score_0_100=sentiment_score_num,
            insider_pro_score_0_100=insider_pro_score,
            insider_net_flow_value=insider_net_flow_value,
            implied_fcf_growth=implied_growth,
            lookback_rev_growth=lookback_rev_growth,
        )

        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            st.metric("Finální skóre (0–100)", final_score)
            mos_show = "—" if not math.isfinite(mos_val) else fmt_pct(mos_val)
            st.metric("MOS (DCF)", mos_show)
        with c2:
            ag = None if not math.isfinite(analyst_gap) else analyst_gap
            st.metric("Gap vs Analyst mean", fmt_pct(ag))
            st.caption("Rozdíl mezi aktuální cenou a průměrnou cílovou cenou analytiků (Yahoo).")
        with c3:
            st.markdown(
                f"""<div style="padding:14px;border-radius:12px;border:1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.03)">
                <div style="font-size:12px;opacity:0.8">Verdikt</div>
                <div style="font-size:28px;font-weight:800;color:{color}">{verdict}</div>
                </div>""",
                unsafe_allow_html=True
            )

        st.markdown("### Proč to vyšlo takhle")
        for b in bullets:
            st.write("• " + b)

        if warns:
            st.markdown("### ⚠️ Reverse DCF validace")
            for w in warns:
                st.warning(w)

        st.markdown("### Dynamický checklist (3 podmínky pro nákup)")
        conds = dynamic_buy_conditions(
            fair_value=fair_value,
            current_price=metrics.get("price").value if metrics.get("price") else None,
            metrics=metrics,
            info=info,
            score=final_score,
            mos=(None if not math.isfinite(mos_val) else mos_val),
            implied_fcf_growth=implied_growth,
        )
        for c in conds:
            st.write("✅ " + c)

        st.markdown("### Rozpad váženého skóre")
        st.json({k: round(v, 1) for k, v in comps.items()})

        st.markdown("### Reverse DCF (co trh implikuje)")
        if implied_growth is None:
            st.info("Implied growth se nepodařilo spočítat (chybí data nebo cena neleží mezi řešeními).")
        else:
            st.metric("Implied FCF growth (DCF)", (f"{implied_growth*100:.1f}%" if isinstance(implied_growth,(int,float)) else "—"))
            st.caption("Jaký roční růst FCF by musel nastat, aby DCF vyšel na aktuální cenu (při tvých DCF nastaveních).")


# ---------------- Tab 7: Memo & Watchlist ----------------
    with tabs[6]:
        st.subheader("Investment memo (one-pager) + watchlist")
        memos = get_memos()
        watch = get_watchlist()

        memo = memos["memos"].get(ticker, {})
        wl = watch["items"].get(ticker, {})

        # Auto-draft snippets
        price_now = metrics.get("price").value if metrics.get("price") else None
        analyst_mean_target = metrics.get("target_mean").value if metrics.get("target_mean") else None
        if not analyst_mean_target:
            analyst_mean_target = metrics.get("target_median").value if metrics.get("target_median") else None
        mos_local = None
        try:
            if fair_value and price_now and float(price_now) != 0:
                mos_local = (float(fair_value) / float(price_now)) - 1.0
        except Exception:
            mos_local = None
        mos_str = f"{mos_local*100:.1f}%" if mos_local is not None else "—"
        auto_thesis = (
            f"{company} ({ticker}) — rychlé shrnutí.\n"
            f"• Sektor/odvětví: {info.get('sector','—')} / {info.get('industry','—')}\n"
            f"• Cena: {fmt_money(price_now)} • Verdikt: {verdict}\n"
            f"• Férovka (DCF): {fmt_money(fair_value) if fair_value else '—'} (MOS {mos_str})\n"
            f"• Analytici (mean target): {fmt_money(analyst_mean_target) if analyst_mean_target else '—'}"
        )
        auto_drivers = (
            "- Růst tržeb a monetizace (produkty, cloud, AI, pricing)\n"
            "- Marže (operating/gross) a provozní páka\n"
            "- Free Cash Flow a kapitálová alokace (buyback/dividendy)\n"
            "- Konkurenční výhoda (moat) + kvalita managementu"
        )
        auto_risks = (
            "- Valuace a očekávání trhu (Reverse DCF / implied růst)\n"
            "- Konkurence/regulace a technologické riziko\n"
            "- Cyklus poptávky / makro / FX\n"
            "- Riziko marží (náklady, capex)"
        )
        auto_buy = (
            f"- Buy zone: pod {fmt_money(fair_value*0.95) if fair_value else '—'} (MOS ≥ 5%)\n"
            f"- Strong buy: pod {fmt_money(fair_value*0.80) if fair_value else '—'} (MOS ≥ 20%)\n"
            f"- Verdikt: {verdict}\n"
            + ((f"- Reverse DCF implied FCF růst: {(implied_growth*100):.1f}%\n") if isinstance(implied_growth,(int,float)) else "- Reverse DCF implied FCF růst: —\n")
        )

        st.markdown("### Memo")
        thesis = st.text_area("Teze (proč to vyhraje)", value=memo.get("thesis") or auto_thesis, height=90)
        drivers = st.text_area("Key drivers (co musí platit, aby teze vyšla)", value=memo.get("drivers") or auto_drivers, height=90)
        risks = st.text_area("Rizika & co sledovat", value=memo.get("risks") or auto_risks, height=90)
        catalysts = st.text_area("Catalysts (co může pohnout cenou)", value=memo.get("catalysts") or "", height=70)
        buy_conditions = st.text_area("Buy podmínky / targety", value=memo.get("buy_conditions") or auto_buy, height=80)
        notes = st.text_area("Poznámky", value=memo.get("notes") or "", height=80)

        col_save, col_pdf = st.columns([1, 1])
        with col_save:
            if st.button("💾 Uložit memo", width='stretch'):
                memos["memos"][ticker] = {
                    "thesis": thesis,
                    "drivers": drivers,
                    "risks": risks,
                    "catalysts": catalysts,
                    "buy_conditions": buy_conditions,
                    "notes": notes,
                    "updated_at": dt.datetime.now().isoformat(),
                }
                set_memos(memos)
                st.success("Uloženo.")

        with col_pdf:
            if _HAS_PDF:
                if st.button("📄 Export PDF (memo)", width='stretch'):
                    summary = {
                        "Price": metric_card(metrics["price"]),
                        "DCF fair value": fmt_money(fair_value) if fair_value else "—",
                        "FCF yield": fmt_pct(metrics["fcf_yield"].value),
                        "P/E": fmt_num(metrics["pe"].value),
                        "Revenue growth": fmt_pct(safe_float(info.get("revenueGrowth"))),
                        "Operating margin": fmt_pct(safe_float(info.get("operatingMargins"))),
                        "Score": str(build_scorecard(metrics, info)[0]),
                    }
                    pdf_bytes = export_memo_pdf(
                        ticker=ticker,
                        company=company,
                        memo={
                            "thesis": thesis,
                            "drivers": drivers,
                            "risks": risks,
                            "catalysts": catalysts,
                            "buy_conditions": buy_conditions,
                            "notes": notes,
                        },
                        summary=summary,
                    )
                    if pdf_bytes:
                        st.download_button(
                            "⬇️ Stáhnout PDF",
                            data=pdf_bytes,
                            file_name=f"memo_{ticker}.pdf",
                            mime="application/pdf",
                            width='stretch'
                        )
                    else:
                        st.error("PDF export není dostupný (chybí reportlab).")
            else:
                st.info("PDF export není dostupný (nainstaluj reportlab).")

        st.markdown("---")
        st.markdown("### Watchlist")
        target_buy = st.number_input("Moje cílová nákupní cena", value=float(wl.get("target_buy", 0.0)) if wl else 0.0, step=1.0)
        add = st.button("⭐ Přidat/aktualizovat ve watchlistu", width='stretch')
        remove = st.button("🗑️ Odebrat z watchlistu", width='stretch')

        if add:
            watch["items"][ticker] = {
                "target_buy": target_buy,
                "added_at": wl.get("added_at") or dt.datetime.now().isoformat(),
                "updated_at": dt.datetime.now().isoformat(),
            }
            set_watchlist(watch)
            st.success("Watchlist aktualizován.")

        if remove:
            if ticker in watch["items"]:
                watch["items"].pop(ticker, None)
                set_watchlist(watch)
                st.success("Odebráno z watchlistu.")

        st.markdown("#### Moje položky")
        items = watch.get("items", {})
        if not items:
            st.info("Watchlist je prázdný.")
        else:
            rows = []
            for tkr, item in items.items():
                inf = fetch_ticker_info(tkr)
                price_now = safe_float(inf.get("currentPrice") or inf.get("regularMarketPrice"))
                tgt = safe_float(item.get("target_buy"))
                hit = (price_now is not None and tgt is not None and tgt > 0 and price_now <= tgt)
                rows.append({
                    "Ticker": tkr,
                    "Price": price_now,
                    "Target buy": tgt,
                    "Hit?": "✅" if hit else "",
                    "Updated": item.get("updated_at", ""),
                })
            st.dataframe(pd.DataFrame(rows), width='stretch')
            st.caption("Jednoduchý alert: pokud cena <= target buy, zobrazí se ✅.")

    st.markdown("---")
    st.caption("Data: Yahoo Finance přes yfinance. Některé metriky mohou chybět / být opožděné. Toto není investiční doporučení.")


if __name__ == "__main__":
    main()
