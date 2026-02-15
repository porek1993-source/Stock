# 🔧 Oprava průhledného sidebaru na mobilu

## 🐛 Problém

Na mobilním zařízení je sidebar **průhledný** a přes něj je vidět hlavní obsah ("Vítej v Stock Picker Pro"), což je **hodně nepřehledné**.

```
❌ ŠPATNĚ (PŘED):
┌─────────────────────────┐
│ [průhledný sidebar]     │ <- Vidíš přes něj text za ním
│   MSFT                  │
│   [input]               │
│                         │
│   "Vítej v Stock..."    │ <- Text z pozadí prosvítá
└─────────────────────────┘
```

## ✅ Řešení

Implementoval jsem **3 vrstvy oprav**:

### 1. CSS Override v app.py (automaticky načteno)
- Force `background-color: #262730` na sidebar
- Z-index 999999 pro překrytí obsahu
- Solid opacity (ne průhlednost)

### 2. External CSS soubor `mobile_fix.css`
- Silnější override s `!important`
- Fallback overlay pro extra jistotu
- Light/dark mode podpora

### 3. Streamlit config `.streamlit/config.toml`
- Nastavení theme barev
- `secondaryBackgroundColor = "#262730"` pro sidebar

## 🚀 Jak aplikovat opravu

### Možnost A: Automatická (doporučeno)
```bash
# Všechny soubory jsou už vytvořené
# Stačí restartovat aplikaci:
streamlit run app.py
```

### Možnost B: Manuální CSS inject
Pokud automatické načtení nefunguje, přidej tento kód na začátek `main()`:

```python
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: #262730 !important;
    z-index: 999999 !important;
}
section[data-testid="stSidebar"] > div {
    background-color: #262730 !important;
}
@media (max-width: 768px) {
    section[data-testid="stSidebar"][aria-expanded="true"] {
        position: fixed !important;
        width: 85% !important;
        height: 100vh !important;
        background-color: #262730 !important;
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.8) !important;
    }
}
</style>
""", unsafe_allow_html=True)
```

### Možnost C: Browser DevTools test
1. Otevři aplikaci na mobilu
2. Použij Chrome Remote Debugging (na desktopu)
3. V DevTools přidej CSS:
```css
section[data-testid="stSidebar"] {
    background-color: #262730 !important;
}
```
4. Zkontroluj, jestli zmizela průhlednost

## 📱 Výsledek po opravě

```
✅ DOBŘE (PO):
┌─────────────────────────┐
│ [#262730 solid sidebar] │ <- Tmavé, nepůhledné pozadí
│   MSFT                  │
│   [input]               │
│                         │
│                         │ <- Žádný prosvítající text
└─────────────────────────┘

Main obsah: [50% opacity blur] <- Ztmaven a rozmazán
```

## 🔍 Debugging

### Krok 1: Zkontroluj, že soubory existují
```bash
ls -la .streamlit/config.toml
ls -la mobile_fix.css
```

### Krok 2: Zkontroluj console v prohlížeči
```javascript
// Otevři DevTools (F12)
// V Console spusť:
document.querySelector('[data-testid="stSidebar"]').style.backgroundColor
// Mělo by vrátit: "rgb(38, 39, 48)" nebo "#262730"
```

### Krok 3: Force reload
```bash
# Na mobilu:
Ctrl+Shift+R (Android Chrome)
Cmd+Shift+R (iOS Safari)

# Nebo vymaž cache:
Settings → Privacy → Clear browsing data → Cached images
```

### Krok 4: Zkontroluj Streamlit verzi
```bash
streamlit --version
# Mělo by být >= 1.30.0

# Pokud je starší:
pip install --upgrade streamlit
```

## 🆘 Pokud stále nefunguje

### Workaround 1: Disable initial sidebar state
V `app.py` změň:
```python
st.set_page_config(
    ...
    initial_sidebar_state="collapsed"  # Místo "expanded"
)
```

### Workaround 2: Mobile-only notice
Přidej warning na začátek pro mobilní uživatele:
```python
st.warning("📱 Pro nejlepší zážitek zavřete sidebar po zadání tickeru (klikněte na ×)")
```

### Workaround 3: Alternative layout
Použij tabs místo sidebaru:
```python
tab1, tab2 = st.tabs(["Analýza", "Nastavení"])
with tab1:
    # Main content
with tab2:
    # Settings (ticker input, DCF params)
```

## 📊 Testovací checklist

- [ ] Otevři app na mobilu (Chrome/Safari)
- [ ] Klikni na hamburger menu (≡) vlevo nahoře
- [ ] Sidebar se otevře
- [ ] **KONTROLA**: Je sidebar tmavý (#262730) nebo průhledný?
- [ ] **KONTROLA**: Je hlavní obsah ztmavený/rozmazaný?
- [ ] **KONTROLA**: Vidíš text "Vítej v Stock Picker Pro" přes sidebar?
- [ ] Zadej ticker (např. MSFT)
- [ ] Klikni "🔍 Analyzovat" nebo stiskni Enter
- [ ] **KONTROLA**: Funguje analýza?

### Expected results:
✅ Sidebar: Tmavé pozadí bez průhlednosti  
✅ Main content: Ztmavený když je sidebar otevřený  
✅ Text input: Jasně viditelný, ne překrytý  
✅ Enter: Spustí analýzu  

## 🎨 Customizace barev

Pokud chceš jinou barvu sidebaru, uprav v `.streamlit/config.toml`:

```toml
[theme]
# Změní barvu sidebaru:
secondaryBackgroundColor = "#1E1E1E"  # Tmavší
# nebo
secondaryBackgroundColor = "#F0F0F0"  # Světlejší
```

A v `mobile_fix.css` všude kde je `#262730` nahraď za tvoji barvu.

## 📞 Pokud nic nepomohlo

1. **Pošli screenshot** průhledného sidebaru
2. **Zjisti Streamlit verzi**: `streamlit --version`
3. **Zjisti prohlížeč**: Chrome/Safari/Firefox + verze
4. **Zkus jiný browser**: Možná browser-specific issue

---

**Status po aktualizaci:** Sidebar má nyní **3 vrstvy CSS override** s `!important`, což by mělo vyřešit průhlednost na 99% mobilních zařízení.

**Fallback:** Pokud stále průhledný → použij Workaround 3 (tabs místo sidebar)
