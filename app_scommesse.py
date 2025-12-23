import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta
from groq import Groq

# ==============================================================================
# ⚙️ CONFIGURAZIONE UTENTE (MODIFICA QUI!)
# ==============================================================================

# 1. INCOLLA QUI LA TUA API KEY DI GROQ (tra le virgolette)
GROQ_API_KEY = "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" 

# 2. BUDGET INIZIALE DI DEFAULT
DEFAULT_BUDGET = 100.0

# ==============================================================================
# CONFIGURAZIONE GRAFICA
# ==============================================================================
st.set_page_config(page_title="AI Betting GOD", page_icon="🤖", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1a1a1a; border-radius: 8px; 
        color: #bbb; flex: 1; border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d26a !important; color: black !important; font-weight: bold; border: none;
    }
    
    .stake-box {
        background-color: #000; border: 1px solid #00d26a; border-radius: 8px;
        padding: 8px; text-align: center;
    }
    .stake-title { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .stake-value { font-size: 18px; font-weight: bold; color: #00d26a; }
    
    .ai-text {
        font-size: 14px; line-height: 1.5; color: #e0e0e0;
        background-color: #262730; padding: 15px; border-radius: 10px;
        border-left: 3px solid #00d26a;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTORE DATI
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "I2", "nome": "🇮🇹 Serie B", "history": "https://www.football-data.co.uk/mmz4281/2526/I2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-b-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Champ", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
    {"id": "P1", "nome": "🇵🇹 Port", "history": "https://www.football-data.co.uk/mmz4281/2526/P1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/portugal-primeira-liga-2025.csv"},
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

def calculate_stake(prob, quota, bankroll):
    try:
        if quota <= 1: return 0
        b = quota - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        stake_pct = (f * 0.25) 
        if stake_pct < 0: return 0
        stake = bankroll * stake_pct
        return round(stake, 2)
    except: return 0

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
        
        # Filtro Safe
        safe = [o for o in options if o[1] > 0.52 or (o[0]=="RISCHIO X" and o[1]>0.32)]
        if safe:
            safe.sort(key=lambda x: x[1], reverse=True)
            best = safe[0]
            tip, prob, qmin = best[0], best[1], best[2]*1.05
        else:
            tip, prob, qmin = "NO BET", 0, 0

        return {
            "c": h, "o": a, "p1": ph, "px": pd, "p2": pa, 
            "p_o25": p_o25, "p_gg": p_gg, "Tip": tip, "ProbWin": prob, "Quota": qmin,
            "Att_H": stats.at[h,'Att_H'], "Dif_H": stats.at[h,'Dif_H'],
            "Att_A": stats.at[a,'Att_A'], "Dif_A": stats.at[a,'Dif_A']
        }
    except: return None

# ==============================================================================
# AGENTE AI (GROQ) - MODELLO AGGIORNATO
# ==============================================================================
def ask_ai_agent(match_data, stake):
    if not GROQ_API_KEY or "INCOLLA" in GROQ_API_KEY: 
        return "⚠️ Manca la API KEY nel codice."
    
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""
    Sei un betting analyst esperto.
    Match: {match_data['c']} vs {match_data['o']}.
    Il modello Poisson dice: {match_data['Tip']} (Prob: {match_data['ProbWin']*100:.1f}%).
    Dati Stats: Attacco Casa {match_data['Att_H']:.2f}, Difesa Ospite {match_data['Dif_A']:.2f}.
    L'algoritmo di Money Management suggerisce di puntare: €{stake}.
    
    SPIEGA IN 2 FRASI PERCHÉ:
    1. Perché statisticamente è uscito questo risultato?
    2. Perché la puntata è alta/bassa? (Se alta = alta sicurezza, se bassa = rischio).
    Sii colloquiale e usa emoji.
    """
    
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # <--- MODELLO AGGIORNATO QUI
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=150, top_p=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Errore AI: {str(e)}"

# ==============================================================================
# UI PRINCIPALE
# ==============================================================================
st.title("🤖 AI Betting GOD")

with st.expander("💰 Il tuo Budget (Clicca per modificare)", expanded=False):
    bankroll = st.number_input("Cassa Attuale (€):", value=DEFAULT_BUDGET, step=10.0)

tab_auto, tab_manual = st.tabs(["📡 RADAR", "🛠️ SCHEDINA"])

# --- TAB AUTO ---
with tab_auto:
    # BOTTONI RIPRISTINATI
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI 📅", use_container_width=True, type="primary"): t_scan = 0
    if c2.button("DOMANI 📆", use_container_width=True): t_scan = 1

    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi del {target_d}...")
        
        results_found = False
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
                                    results_found = True
                                    stake = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                                    
                                    with st.container(border=True):
                                        st.subheader(f"{res['c']} - {res['o']}")
                                        st.caption(db['nome'])
                                        
                                        # ROW 1: CONSIGLIO E SOLDI
                                        c1, c2 = st.columns([3, 1])
                                        with c1:
                                            if "PUNTA" in res['Tip']: st.success(f"**{res['Tip']}**")
                                            elif "OVER" in res['Tip']: st.info(f"**{res['Tip']}**")
                                            else: st.warning(res['Tip'])
                                            st.caption(f"Quota Min: {res['Quota']:.2f}")
                                        with c2:
                                            st.markdown(f"""<div class="stake-box"><div class="stake-title">PUNTA</div><div class="stake-value">€{stake}</div></div>""", unsafe_allow_html=True)

                                        # ROW 2: BARRE
                                        st.progress(res['p1'], f"1: {res['p1']*100:.0f}%")
                                        st.progress(res['p_o25'], f"Over 2.5: {res['p_o25']*100:.0f}%")
                                        
                                        # ROW 3: POPUP AI (EXPANDER)
                                        with st.expander("❓ Perché questa scelta? (Analisi AI)"):
                                            with st.spinner("L'AI sta scrivendo..."):
                                                ai_msg = ask_ai_agent(res, stake)
                                                st.markdown(f"<div class='ai-text'>{ai_msg}</div>", unsafe_allow_html=True)
        
        if not results_found:
             st.warning(f"Nessuna 'Value Bet' ad alta probabilità trovata per {target_d}.")


# --- TAB MANUALE ---
with tab_manual:
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
        
        if st.button("🔮 ANALIZZA MATCH", type="primary", use_container_width=True):
            res = analyze_math(h, a, st.session_state['cur_stats'], st.session_state['cur_ah'], st.session_state['cur_aa'])
            if res:
                stake = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                
                with st.container(border=True):
                    st.markdown(f"### {h} vs {a}")
                    
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.info(f"Consiglio: **{res['Tip']}**")
                        st.caption(f"Probabilità: {res['ProbWin']*100:.1f}%")
                    with c2:
                        st.markdown(f"""<div class="stake-box"><div class="stake-title">PUNTA</div><div class="stake-value">€{stake}</div></div>""", unsafe_allow_html=True)
                    
                    st.divider()
                    st.progress(res['p1'], f"Casa: {res['p1']*100:.0f}%")
                    st.progress(res['p_o25'], f"Over 2.5: {res['p_o25']*100:.0f}%")
                    st.progress(res['p_gg'], f"Goal: {res['p_gg']*100:.0f}%")
                    
                    # POPUP AI
                    with st.expander("❓ Clicca per l'analisi dell'Esperto AI"):
                        with st.spinner("Consultazione archivio..."):
                             ai_msg = ask_ai_agent(res, stake)
                             st.markdown(f"<div class='ai-text'>{ai_msg}</div>", unsafe_allow_html=True)
