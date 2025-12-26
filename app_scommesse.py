import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io

# ==============================================================================
# 1. CONFIGURAZIONE & DESIGN (LAYOUT DARK NEON)
# ==============================================================================
st.set_page_config(page_title="BETTING MASTER AI", page_icon="💎", layout="wide")

# CSS PERSONALIZZATO
st.markdown("""
<style>
    /* Sfondo e colori base */
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Intestazioni */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; font-weight: 800; }
    h1 { color: #00e676; text-transform: uppercase; letter-spacing: 2px; }
    h3 { color: #00b0ff; }
    
    /* Card delle Partite */
    .match-card {
        background-color: #1a1c24;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Box Statistiche dentro la card */
    .stat-box {
        text-align: center;
        background: #262730;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #444;
        width: 30%;
    }
    .stat-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .stat-value { font-size: 18px; font-weight: bold; color: #fff; }
    
    .stat-green { color: #00e676 !important; text-shadow: 0 0 10px rgba(0,230,118,0.3); }
    .stat-blue { color: #2979ff !important; }
    .stat-yellow { color: #ffeb3b !important; }
    
    /* Pulsanti */
    .stButton>button {
        background: linear-gradient(45deg, #00e676, #00c853);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px #00e676;
        color: black;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }
    
    /* Flex container per le stats */
    .flex-stats {
        display: flex;
        justify-content: space-around;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATABASE COMPLETO EUROPA (Link 2024/2025)
# ==============================================================================
LEAGUES = {
    "🇮🇹 Serie A": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "🇬🇧 Premier League": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇪🇸 La Liga": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "🇩🇪 Bundesliga": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "🇫🇷 Ligue 1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "🇳🇱 Eredivisie (Olanda)": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "🇵🇹 Primeira Liga (Portogallo)": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
    "🇧🇪 Jupiler League (Belgio)": "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
    "🏴󠁧󠁢󠁳󠁣󠁴󠁿 Premiership (Scozia)": "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
    "🇹🇷 Super Lig (Turchia)": "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
    "🇬🇷 Super League (Grecia)": "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
}

# ==============================================================================
# 3. LOGICA MATEMATICA (POISSON ENGINE)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            return df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
    except: return None
    return None

def calculate_stats(df, home, away):
    try:
        avg_h = df['FTHG'].mean()
        avg_a = df['FTAG'].mean()
        
        # Statistiche Casa
        h_att = df[df['HomeTeam'] == home]['FTHG'].mean() / avg_h
        h_def = df[df['HomeTeam'] == home]['FTAG'].mean() / avg_a
        
        # Statistiche Ospite
        a_att = df[df['AwayTeam'] == away]['FTAG'].mean() / avg_a
        a_def = df[df['AwayTeam'] == away]['FTHG'].mean() / avg_h
        
        if any(pd.isna([h_att, h_def, a_att, a_def])): return None

        # Lambda (Expected Goals)
        xg_h = h_att * a_def * avg_h
        xg_a = a_att * h_def * avg_a
        
        return xg_h, xg_a
    except: return None

def predict_match(xg_h, xg_a):
    # Matrice Poisson
    max_goals = 6
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i][j] = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a)
    
    # Probabilità
    p_1 = np.sum(np.tril(matrix, -1))
    p_x = np.sum(np.diag(matrix))
    p_2 = np.sum(np.triu(matrix, 1))
    
    p_over25 = 0
    p_gg = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > 2.5: p_over25 += matrix[i][j]
            if i > 0 and j > 0: p_gg += matrix[i][j]

    return {
        "1": p_1, "X": p_x, "2": p_2,
        "OVER 2.5": p_over25, "UNDER 2.5": 1 - p_over25,
        "GOAL": p_gg, "NO GOAL": 1 - p_gg
    }

def calculate_confidence_stake(prob, bankroll):
    # Se probabilità < 50%, non scommettere
    if prob <= 0.50: return 0.0
    # Scaliamo lo stake in base alla sicurezza
    # 50% -> 0€
    # 60% -> 4% del budget
    # 80% -> 12% del budget
    # 100% -> 20% del budget
    factor = (prob - 0.50) * 2 
    stake = bankroll * (factor * 0.20)
    return round(stake, 2)

# ==============================================================================
# 4. INTERFACCIA UTENTE
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ IMPOSTAZIONI")
    budget = st.number_input("IL TUO BUDGET (€)", value=100.0, step=10.0)
    st.divider()
    st.info("💡 Scegli il campionato, seleziona le squadre e aggiungi al carrello. Il sistema calcolerà tutto.")

# --- MAIN ---
st.title("💎 BETTING MASTER AI")
st.markdown("### Algoritmo Predittivo Poisson • Manual Mode")

# Session State
if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}

# SELEZIONE
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 1. Scegli Match")
    with st.container(border=True):
        league_sel = st.selectbox("Campionato", list(LEAGUES.keys()))
        
        # Caricamento Dati
        if league_sel not in st.session_state['data_cache']:
            with st.spinner("Scarico dati..."):
                df = load_data(LEAGUES[league_sel])
                if df is not None:
                    st.session_state['data_cache'][league_sel] = df
        
        current_df = st.session_state['data_cache'].get(league_sel)
        
        if current_df is not None:
            teams = sorted(current_df['HomeTeam'].unique())
            home_team = st.selectbox("Squadra Casa", teams)
            away_team = st.selectbox("Squadra Ospite", teams, index=1)
            
            st.write("") # Spazio
            
            if st.button("➕ AGGIUNGI AL CARRELLO", type="primary"):
                if home_team != away_team:
                    st.session_state['cart'].append({
                        "League": league_sel,
                        "Home": home_team,
                        "Away": away_team,
                        "DF": current_df
                    })
                    st.success("Match aggiunto!")
                else:
                    st.error("Scegli squadre diverse!")
        else:
            st.error("Errore scaricamento dati per questo campionato.")

# CARRELLO & RISULTATI
with col2:
    st.markdown(f"#### 2. Schedina ({len(st.session_state['cart'])} match)")
    
    if not st.session_state['cart']:
        st.info("Il carrello è vuoto. Aggiungi delle partite dalla colonna di sinistra.")
    else:
        if st.button("🗑️ SVUOTA TUTTO"):
            st.session_state['cart'] = []
            st.rerun()
            
        st.write("") # Spazio
        
        # LOOP SUL CARRELLO
        for i, item in enumerate(st.session_state['cart']):
            # Calcoli
            xg = calculate_stats(item['DF'], item['Home'], item['Away'])
            
            if xg:
                xg_h, xg_a = xg
                probs = predict_match(xg_h, xg_a)
                
                # Determinazione Pronostico Migliore
                best_tip = max(probs, key=probs.get)
                best_prob = probs[best_tip]
                
                # Calcolo Stake Automatico
                stake = calculate_confidence_stake(best_prob, budget)
                
                # --- RENDER CARD ---
                # Usiamo st.markdown con HTML pulito per la card
                card_html = f"""
                <div class="match-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <span style="color:#888; font-size:12px;">{item['League']}</span>
                        <span style="color:#00e676; font-weight:bold; font-family:monospace;">xG: {xg_h:.2f} - {xg_a:.2f}</span>
                    </div>
                    <h3 style="margin:0; text-align:center; color:#fff; font-size:22px;">{item['Home']} <span style="color:#666">vs</span> {item['Away']}</h3>
                    <hr style="border-color:#333; margin:15px 0;">
                    
                    <div class="flex-stats">
                        <div class="stat-box">
                            <div class="stat-label">STRATEGIA</div>
                            <div class="stat-value stat-green">{best_tip}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">SICUREZZA</div>
                            <div class="stat-value stat-blue">{best_prob*100:.1f}%</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">PUNTA</div>
                            <div class="stat-value stat-yellow">€{stake:.2f}</div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                # Dettagli Espandibili
                with st.expander(f"📊 Analisi Dettagliata: {item['Home']} vs {item['Away']}"):
                    c1, c2, c3 = st.columns(3)
                    
                    c1.write("**Esito Finale (1X2)**")
                    c1.progress(probs['1'], f"1: {probs['1']*100:.1f}%")
                    c1.progress(probs['X'], f"X: {probs['X']*100:.1f}%")
                    c1.progress(probs['2'], f"2: {probs['2']*100:.1f}%")
                    
                    c2.write("**Under / Over**")
                    c2.progress(probs['OVER 2.5'], f"Over 2.5: {probs['OVER 2.5']*100:.1f}%")
                    c2.progress(probs['UNDER 2.5'], f"Under 2.5: {probs['UNDER 2.5']*100:.1f}%")
                    
                    c3.write("**Goal / NoGoal**")
                    c3.progress(probs['GOAL'], f"Goal: {probs['GOAL']*100:.1f}%")
                    c3.progress(probs['NO GOAL'], f"No Goal: {probs['NO GOAL']*100:.1f}%")
            
            else:
                st.error(f"Dati insufficienti per calcolare {item['Home']} vs {item['Away']}")
