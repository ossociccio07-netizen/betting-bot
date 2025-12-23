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
# 1. INCOLLA QUI LA TUA API KEY DI GROQ
GROQ_API_KEY = "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" 

# 2. BUDGET INIZIALE
DEFAULT_BUDGET = 100.0

# ==============================================================================
# STILE GRAFICO "TOTAL CONTROL"
# ==============================================================================
st.set_page_config(page_title="AI Betting ULTIMATE", page_icon="🧠", layout="centered")

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
        background-color: #00d26a !important; color: black !important; border: none;
    }
    
    /* Box Verdetto Finale */
    .verdict-box {
        background: linear-gradient(135deg, #1e1e1e 0%, #0d0d0d 100%);
        border: 2px solid #00d26a; border-radius: 12px; padding: 20px;
        text-align: center; margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 210, 106, 0.2);
    }
    .verdict-title { font-size: 14px; color: #00d26a; letter-spacing: 2px; text-transform: uppercase; font-weight: bold; }
    .verdict-main { font-size: 32px; color: #fff; font-weight: 900; margin: 10px 0; }
    .verdict-stake { background-color: #00d26a; color: #000; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 18px; display: inline-block; }
    
    /* Box Statistiche */
    .stats-container {
        background-color: #222; border-radius: 10px; padding: 15px; margin-top: 10px; border: 1px solid #333;
    }
    .stat-row { margin-bottom: 10px; }
    .stat-label { font-size: 12px; color: #aaa; margin-bottom: 2px; }
    
    /* Box News */
    .news-box {
        font-size: 13px; background-color: #2d2d2d; padding: 15px; border-radius: 8px;
        border-left: 4px solid #ffaa00; margin-top: 15px; color: #ddd;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTORE DATI
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
        
        # Over/Under
        p_o25 = 1 - (poisson.pmf(0, lh+la) + poisson.pmf(1, lh+la) + poisson.pmf(2, lh+la))
        p_u25 = 1 - p_o25
        
        # BTTS (Goal)
        p_gg = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
        
        # Scelta Migliore (Math)
        options = [
            ("PUNTA 1", ph, 1/ph if ph>0 else 0), 
            ("PUNTA 2", pa, 1/pa if pa>0 else 0), 
            ("RISCHIO X", pd, 1/pd if pd>0 else 0),
            ("OVER 2.5", p_o25, 1/p_o25 if p_o25>0 else 0), 
            ("UNDER 2.5", p_u25, 1/p_u25 if p_u25>0 else 0),
        ]
        
        # Filtro Safe Math
        safe = [o for o in options if o[1] > 0.50 or (o[0]=="RISCHIO X" and o[1]>0.32)]
        if safe:
            safe.sort(key=lambda x: x[1], reverse=True)
            best = safe[0]
            tip, prob, qmin = best[0], best[1], best[2]*1.05
        else:
            tip, prob, qmin = "NO BET", 0, 0

        return {
            "c": h, "o": a, 
            "Math_Tip": tip, "Math_Prob": prob, "Math_Quota": qmin,
            "p1": ph, "px": pd, "p2": pa,
            "p_o25": p_o25, "p_u25": p_u25, "p_gg": p_gg
        }
    except: return None

# ==============================================================================
# RICERCA WEB & AI
# ==============================================================================
def search_news_sherlock(team1, team2):
    q = f"{team1} {team2} injuries suspensions team news lineup prediction December 2025"
    try:
        res = DDGS().text(q, max_results=4)
        return "\n".join([f"- {r['body']}" for r in res])[:2500] if res else "Nessuna news live."
    except: return "Errore ricerca news."

def ai_final_decision(match_data, math_stake, news_text):
    if not GROQ_API_KEY or "INCOLLA" in GROQ_API_KEY: return match_data['Math_Tip'], math_stake, "Manca API Key."
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Sei un Betting Manager. Analizza questi dati per {match_data['c']} vs {match_data['o']}.

    1. DATI MATEMATICI (Poisson):
    - Miglior scelta matematica: {match_data['Math_Tip']} (Prob: {match_data['Math_Prob']*100:.1f}%)
    - Prob 1X2: Casa {match_data['p1']*100:.0f}%, X {match_data['px']*100:.0f}%, Ospite {match_data['p2']*100:.0f}%
    - Prob Gol: Over 2.5 {match_data['p_o25']*100:.0f}%, Under 2.5 {match_data['p_u25']*100:.0f}%
    
    2. NEWS LIVE (WEB):
    {news_text}
    
    COMPITO:
    Incrocia Matematica e News. 
    - Se le news riportano infortuni pesanti (Top Player), DEVI cambiare il pronostico o ridurre drasticamente lo stake.
    - Se la matematica dice "PUNTA 1" ma mancano i bomber di casa, cambia in "UNDER 2.5" o "NO BET".
    
    OUTPUT JSON FORMAT (Solo testo):
    VERDETTO: [Esito Finale, es: OVER 2.5]
    STAKE: [Euro, es: 5.50]
    ANALISI: [Spiegazione logica. Cita i giocatori assenti se ci sono.]
    """
    
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = resp.choices[0].message.content
        
        final_tip = match_data['Math_Tip']
        final_stake = math_stake
        reason = "Analisi standard."
        
        for line in content.split('\n'):
            if "VERDETTO:" in line: final_tip = line.replace("VERDETTO:", "").strip()
            if "STAKE:" in line: 
                try:
                    s_str = line.replace("STAKE:", "").replace("€", "").strip()
                    final_stake = float(re.findall(r"\d+\.\d+", s_str)[0])
                except: pass
            if "ANALISI:" in line: reason = line.replace("ANALISI:", "").strip()
            
        return final_tip, final_stake, reason
    except Exception as e:
        return match_data['Math_Tip'], math_stake, f"Errore AI: {e}"

def calculate_stake(prob, quota, bankroll):
    try:
        if quota <= 1: return 0
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        stake = bankroll * (f * 0.20)
        return round(max(0, stake), 2)
    except: return 0

# ==============================================================================
# UI PRINCIPALE
# ==============================================================================
st.title("🧠 AI Betting ULTIMATE")

with st.expander("💰 Bankroll Manager", expanded=False):
    bankroll = st.number_input("Cassa Totale (€):", value=DEFAULT_BUDGET, step=10.0)

tab_radar, tab_cart = st.tabs(["RADAR VELOCE", "CARRELLO PROFONDO"])

# --- RADAR ---
with tab_radar:
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI 📅", use_container_width=True): t_scan = 0
    if c2.button("DOMANI 📆", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Scan {target_d} (Solo Matematica)...")
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
                                if res and res['Math_Prob'] > 0.55:
                                    with st.container(border=True):
                                        st.markdown(f"**{c} vs {o}**")
                                        st.caption(f"Math: {res['Math_Tip']} ({res['Math_Prob']*100:.0f}%)")

# --- CARRELLO ---
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
        for i, item in enumerate(st.session_state['cart']):
            c_txt, c_del = st.columns([5,1])
            c_txt.text(f"{item['c']} vs {item['o']}")
            if c_del.button("❌", key=f"del_{i}"):
                st.session_state['cart'].pop(i)
                st.rerun()

        if st.button("🧠 ANALIZZA TUTTO (MATH + WEB + AI)", type="primary", use_container_width=True):
            
            bar = st.progress(0)
            for idx, item in enumerate(st.session_state['cart']):
                bar.progress((idx+1)/len(st.session_state['cart']))
                
                # 1. MATH
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    math_stake = calculate_stake(res['Math_Prob'], res['Math_Quota'], bankroll)
                    
                    # 2. WEB SEARCH
                    news = search_news_sherlock(item['c'], item['o'])
                    
                    # 3. AI JUDGE
                    final_tip, final_stake, reason = ai_final_decision(res, math_stake, news)
                    
                    # 4. VISUALIZZAZIONE
                    st.markdown("---")
                    st.markdown(f"### {item['c']} - {item['o']}")
                    st.caption(f"📍 {item['lega']}")
                    
                    # VERDETTO FINALE
                    st.markdown(f"""
                    <div class="verdict-box">
                        <div class="verdict-title">VERDETTO FINALE INTELLIGENTE</div>
                        <div class="verdict-main">{final_tip}</div>
                        <div class="verdict-stake">Punta: €{final_stake}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # BOX ESPANDIBILE: DATI GREZZI + RAGIONAMENTO
                    with st.expander("📊 Vedi Dati Matematici & Analisi News"):
                        
                        st.markdown("#### 1. Matematica Pura (Prima delle News)")
                        st.info(f"Il modello Poisson consigliava: **{res['Math_Tip']}**")
                        
                        # Barre Probabilità
                        st.markdown("**Probabilità 1X2:**")
                        c1, c2, c3 = st.columns(3)
                        c1.progress(res['p1'], f"1: {res['p1']*100:.0f}%")
                        c2.progress(res['px'], f"X: {res['px']*100:.0f}%")
                        c3.progress(res['p2'], f"2: {res['p2']*100:.0f}%")
                        
                        st.markdown("**Probabilità Gol:**")
                        c4, c5 = st.columns(2)
                        c4.progress(res['p_o25'], f"Over 2.5: {res['p_o25']*100:.0f}%")
                        c5.progress(res['p_u25'], f"Under 2.5: {res['p_u25']*100:.0f}%")
                        
                        st.markdown("---")
                        st.markdown("#### 2. Intelligence Report (AI)")
                        st.markdown(f"""
                        <div class="news-box">
                        <b>🕵️‍♂️ ANALISI INVESTIGATIVA:</b><br>{reason}<br><br>
                        <b>🌐 FONTI NEWS:</b><br>
                        <i style="font-size:11px">{news[:500]}...</i>
                        </div>
                        """, unsafe_allow_html=True)
            
            bar.empty()
            
        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
