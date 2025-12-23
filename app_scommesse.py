import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
import re
from datetime import datetime, timedelta
from groq import Groq
from duckduckgo_search import DDGS

# ==============================================================================
# ⚙️ CONFIGURAZIONE UTENTE
# ==============================================================================
GROQ_API_KEY = "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" 
DEFAULT_BUDGET = 100.0

# ==============================================================================
# STILE GRAFICO "CLEAN"
# ==============================================================================
st.set_page_config(page_title="AI Betting BOSS", page_icon="🦍", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #111; border-radius: 8px; 
        color: #888; flex: 1; border: 1px solid #333; font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff88 !important; color: black !important; border: none;
    }
    
    /* Box Risultato Finale */
    .final-tip-box {
        background: linear-gradient(45deg, #004d00, #006600);
        padding: 15px; border-radius: 10px; text-align: center;
        border: 1px solid #00ff88; margin-bottom: 10px;
        box-shadow: 0 0 10px rgba(0, 255, 136, 0.2);
    }
    .final-tip-label { font-size: 12px; color: #aaffcc; text-transform: uppercase; letter-spacing: 2px; }
    .final-tip-value { font-size: 28px; font-weight: 900; color: #fff; margin: 5px 0; }
    
    /* Box Soldi */
    .money-box {
        background-color: #1a1a1a; border: 1px solid #333; border-radius: 10px;
        padding: 15px; text-align: center; height: 100%;
    }
    .money-val { font-size: 24px; color: #00ff88; font-weight: bold; }
    
    /* Box News */
    .news-report {
        font-family: 'Courier New', monospace; font-size: 13px; color: #ddd;
        background-color: #222; padding: 15px; border-radius: 8px; border-left: 3px solid #ffaa00;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTORE DATI & MATEMATICA
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Champ", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
]

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        sc = df.groupby('HomeTeam')[['FTHG','FTAG']].mean()
        st = df.groupby('AwayTeam')[['FTAG','FTHG']].mean()
        fc = df.groupby('HomeTeam')[['FTHG','FTAG']].apply(lambda x: x.tail(5).mean())
        ft = df.groupby('AwayTeam')[['FTAG','FTHG']].apply(lambda x: x.tail(5).mean())
        
        sc.columns, st.columns = ['H_GF_S','H_GS_S'], ['A_GF_S','A_GS_S']
        fc.columns, ft.columns = ['H_GF_F','H_GS_F'], ['A_GF_F','A_GS_F']
        
        tot = pd.concat([sc,st,fc,ft], axis=1)
        PS, PF = 0.65, 0.35
        tot['Att_H'] = ((tot['H_GF_S']*PS + tot['H_GF_F']*PF) / avg_h)
        tot['Dif_H'] = ((tot['H_GS_S']*PS + tot['H_GS_F']*PF) / avg_a)
        tot['Att_A'] = ((tot['A_GF_S']*PS + tot['A_GF_F']*PF) / avg_a)
        tot['Dif_A'] = ((tot['A_GS_S']*PS + tot['A_GS_F']*PF) / avg_h)
        return tot, avg_h, avg_a
    except: return None, None, None

def analyze_math(h, a, stats, ah, aa):
    try:
        if h not in stats.index or a not in stats.index: return None
        lh = stats.at[h,'Att_H'] * stats.at[a,'Dif_A'] * ah
        la = stats.at[a,'Att_A'] * stats.at[h,'Dif_H'] * aa
        
        ph, pd, pa = 0, 0, 0
        for i in range(6):
            for j in range(6):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i>j: ph+=p
                elif i==j: pd+=p
                else: pa+=p
        
        p_o25 = 1 - (poisson.pmf(0, lh+la) + poisson.pmf(1, lh+la) + poisson.pmf(2, lh+la))
        p_gg = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
        
        options = [
            ("PUNTA 1", ph, 1/ph if ph>0 else 0), 
            ("PUNTA 2", pa, 1/pa if pa>0 else 0), 
            ("RISCHIO X", pd, 1/pd if pd>0 else 0),
            ("OVER 2.5", p_o25, 1/p_o25 if p_o25>0 else 0), 
            ("UNDER 2.5", 1-p_o25, 1/(1-p_o25) if 1-p_o25>0 else 0), 
            ("GOAL", p_gg, 1/p_gg if p_gg>0 else 0)
        ]
        
        safe = [o for o in options if o[1] > 0.50]
        if safe:
            safe.sort(key=lambda x: x[1], reverse=True)
            best = safe[0]
            tip, prob, qmin = best[0], best[1], best[2]*1.05
        else:
            tip, prob, qmin = "NO BET", 0, 0

        return {
            "c": h, "o": a, "Tip": tip, "ProbWin": prob, "Quota": qmin,
            "p1": ph, "px": pd, "p2": pa
        }
    except: return None

# ==============================================================================
# 🧠 IL CERVELLO DECISIONALE (AI + WEB)
# ==============================================================================
def search_news(team1, team2):
    """Cerca su DuckDuckGo e riassume i titoli"""
    q = f"{team1} vs {team2} team news injuries lineups prediction"
    try:
        res = DDGS().text(q, max_results=3)
        return "\n".join([f"- {r['body']}" for r in res])[:1200]
    except: return "Nessuna news live trovata."

def ai_final_decision(match_data, math_stake, news_text):
    """L'IA prende Math + News e decide il verdetto finale"""
    if not GROQ_API_KEY or "INCOLLA" in GROQ_API_KEY: return "ERR_KEY", 0, "Manca API Key"
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Sei il Boss delle Scommesse. Hai due input:
    1. MATEMATICA: Dice {match_data['Tip']} (Quota {match_data['Quota']:.2f}). Puntata Math: €{math_stake}.
    2. NEWS LIVE: {news_text}
    
    COMPITO:
    Analizza le news (infortuni, assenze AFCON, turnover).
    Se le news sono brutte per la scommessa matematica, CAMBIA il pronostico o abbassa la puntata.
    Se le news confermano, mantieni.
    
    OUTPUT RIGIDO (Non scrivere altro):
    TIP: [Il tuo pronostico finale, es: OVER 2.5 o UNDER 2.5 o PUNTA 1]
    STAKE: [La nuova puntata in Euro, es: 3.50]
    REASON: [Spiegazione breve con nomi giocatori e emoji. Max 2 righe]
    """
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = resp.choices[0].message.content
        
        # Parsing manuale della risposta
        final_tip = "N/A"
        final_stake = math_stake
        reason = "Analisi non disponibile."
        
        for line in content.split('\n'):
            if "TIP:" in line: final_tip = line.replace("TIP:", "").strip()
            if "STAKE:" in line: 
                try:
                    s_str = line.replace("STAKE:", "").replace("€", "").strip()
                    final_stake = float(re.findall(r"\d+\.\d+", s_str)[0])
                except: pass
            if "REASON:" in line: reason = line.replace("REASON:", "").strip()
            
        return final_tip, final_stake, reason
    except Exception as e:
        return match_data['Tip'], math_stake, f"Errore AI: {e}"

def calculate_stake(prob, quota, bankroll):
    try:
        if quota <= 1: return 0
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        stake = bankroll * (f * 0.20) # Molto prudente
        return round(max(0, stake), 2)
    except: return 0

# ==============================================================================
# UI PRINCIPALE
# ==============================================================================
st.title("🦍 AI Betting BOSS v7")

with st.expander("💰 Bankroll (Budget)", expanded=False):
    bankroll = st.number_input("Cassa (€):", value=DEFAULT_BUDGET, step=10.0)

tab_radar, tab_cart = st.tabs(["RADAR VELOCE", "CARRELLO INTELLIGENTE"])

# --- TAB 1: RADAR (Solo Matematica Veloce) ---
with tab_radar:
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI 📅", use_container_width=True): t_scan = 0
    if c2.button("DOMANI 📆", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Scansione matematica del {target_d}...")
        
        for db in DATABASE:
            df_cal = get_data(db['fixture'])
            if df_cal is not None:
                cd = 'Date' if 'Date' in df_cal.columns else 'Match Date'
                df_cal[cd] = pd.to_datetime(df_cal[cd], errors='coerce')
                matches = df_cal[df_cal[cd].dt.strftime('%Y-%m-%d') == target_d]
                
                if not matches.empty:
                    df_h = get_data(db['history'])
                    if df_h is not None:
                        stats, ah, aa = process_stats(df_h)
                        if stats is not None:
                            for _, r in matches.iterrows():
                                c = r.get('Home Team', r.get('HomeTeam','')).strip()
                                o = r.get('Away Team', r.get('AwayTeam','')).strip()
                                m = {"Man Utd":"Man United", "Utd":"United"}
                                c, o = m.get(c,c), m.get(o,o)
                                
                                res = analyze_math(c, o, stats, ah, aa)
                                if res and res['ProbWin'] > 0.55:
                                    stake = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                                    with st.container(border=True):
                                        st.markdown(f"**{c} vs {o}**")
                                        c_t, c_s = st.columns([2,1])
                                        c_t.info(f"Math: {res['Tip']}")
                                        c_s.markdown(f"**€{stake}**")

# --- TAB 2: CARRELLO (AI + NEWS REALI) ---
with tab_cart:
    names = [d['nome'] for d in DATABASE]
    sel = st.selectbox("Campionato", names)
    
    if st.session_state['loaded_league'] != sel:
        with st.spinner("Loading..."):
            db = next(d for d in DATABASE if d['nome'] == sel)
            df = get_data(db['history'])
            if df is not None:
                stats, ah, aa = process_stats(df)
                st.session_state.update({'cur_stats': stats, 'cur_ah': ah, 'cur_aa': aa, 
                                       'cur_teams': sorted(stats.index.tolist()), 'loaded_league': sel})

    if 'cur_teams' in st.session_state:
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", st.session_state['cur_teams'])
        a = c2.selectbox("Ospite", st.session_state['cur_teams'], index=1)
        
        if st.button("➕ AGGIUNGI", use_container_width=True):
            if h != a:
                st.session_state['cart'].append({
                    'c': h, 'o': a, 'lega': sel,
                    'stats': st.session_state['cur_stats'],
                    'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']
                })

    st.divider()
    
    if st.session_state['cart']:
        st.subheader(f"🛒 Carrello ({len(st.session_state['cart'])})")
        
        # Lista Carrello
        for i, item in enumerate(st.session_state['cart']):
            c_txt, c_del = st.columns([5,1])
            c_txt.text(f"{item['c']} vs {item['o']}")
            if c_del.button("❌", key=f"del_{i}"):
                st.session_state['cart'].pop(i)
                st.rerun()

        # BOTTONE MAGICO
        if st.button("🧠 ELABORA VERDETTO FINALE (MATH + NEWS)", type="primary", use_container_width=True):
            
            bar = st.progress(0)
            for idx, item in enumerate(st.session_state['cart']):
                bar.progress((idx+1)/len(st.session_state['cart']))
                
                # 1. Matematica Base
                res_math = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                
                if res_math:
                    math_stake = calculate_stake(res_math['ProbWin'], res_math['Quota'], bankroll)
                    
                    # 2. Ricerca News
                    news = search_news(item['c'], item['o'])
                    
                    # 3. Decisione AI
                    final_tip, final_stake, reason = ai_final_decision(res_math, math_stake, news)
                    
                    # DISPLAY RISULTATO
                    st.markdown("---")
                    st.subheader(f"{item['c']} - {item['o']}")
                    st.caption(f"📍 {item['lega']}")
                    
                    # BOX VERDE E BOX SOLDI
                    c_tip, c_cash = st.columns([3, 1])
                    with c_tip:
                        st.markdown(f"""
                        <div class="final-tip-box">
                            <div class="final-tip-label">VERDETTO IA</div>
                            <div class="final-tip-value">{final_tip}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with c_cash:
                        st.markdown(f"""
                        <div class="money-box">
                            <div style="font-size:10px; color:#888;">PUNTA</div>
                            <div class="money-val">€{final_stake}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # EXPANDER CON SPIEGAZIONE
                    with st.expander("🕵️ Perché questa scelta? (Clicca per leggere)"):
                        st.markdown(f"""
                        <div class="news-report">
                        <b>🔍 ANALISI INTELLIGENCE:</b><br>
                        {reason}<br><br>
                        <i>(Base Matematica era: {res_math['Tip']} @ {res_math['Quota']:.2f})</i>
                        </div>
                        """, unsafe_allow_html=True)
            
            bar.empty()
            
        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
