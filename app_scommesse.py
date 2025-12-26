import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io

# ==============================================================================
# 1. CONFIGURAZIONE & DESIGN (LAYOUT FIGO)
# ==============================================================================
st.set_page_config(page_title="BETTING MASTER AI", page_icon="💎", layout="wide")

# CSS PERSONALIZZATO PER UN LOOK "DARK NEON"
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
    
    /* Metriche personalizzate */
    .stat-box {
        text-align: center;
        background: #262730;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #444;
    }
    .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
    .stat-value { font-size: 20px; font-weight: bold; color: #fff; }
    .stat-green { color: #00e676 !important; }
    .stat-blue { color: #2979ff !important; }
    .stat-yellow { color: #ffeb3b !important; }
    
    /* Pulsanti */
    .stButton>button {
        background: linear-gradient(45deg, #00e676, #00c853);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        width: 100%;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px #00e676;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111; border-right: 1px solid #222; }
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
    "🇬🇧 Championship (UK 2)": "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
    "🇮🇹 Serie B": "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
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
    
    p_over25 = 1 - np.sum(matrix[0:3, 0:3]) + matrix[0,2] + matrix[1,1] + matrix[2,0] # Approssimazione corretta O2.5
    # Correzione Over 2.5 precisa: somma prob dove i+j > 2.5
    p_over25 = 0
    p_gg = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > 2.5: p_over25 += matrix[i][j]
            if i > 0 and j > 0: p_gg += matrix[i][j]

    return {
        "1": p_1, "X": p_x, "2": p_2,
        "Over2.5": p_over25, "Under2.5": 1 - p_over25,
        "GG": p_gg, "NG": 1 - p_gg
    }

def kelly_criterion(prob, odds, bankroll, fraction=0.1):
    if odds <= 1: return 0
    b = odds - 1
    q = 1 - prob
    f = (b * prob - q) / b
    return max(0, bankroll * (f * fraction))

# ==============================================================================
# 4. INTERFACCIA UTENTE
# ==============================================================================

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ SETTINGS")
    budget = st.number_input("IL TUO BUDGET (€)", value=100.0, step=10.0)
    st.divider()
    st.info("💡 Scegli il campionato, seleziona le squadre e aggiungi al carrello. Poi analizza tutto insieme.")

# --- MAIN ---
st.title("💎 BETTING MASTER AI")
st.markdown("### Algoritmo Predittivo Poisson • Multi-League")

# Session State
if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}

# SELEZIONE
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### 1. Scegli Match")
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
        
        odds_val = st.number_input("Quota Bookmaker (Opzionale per Stake)", value=1.0, step=0.01, min_value=1.0)
        
        if st.button("➕ AGGIUNGI AL CARRELLO", type="primary"):
            if home_team != away_team:
                st.session_state['cart'].append({
                    "League": league_sel,
                    "Home": home_team,
                    "Away": away_team,
                    "Odds": odds_val,
                    "DF": current_df # Salviamo il riferimento al DF
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
        # Tasto Reset e Analizza
        c_act1, c_act2 = st.columns(2)
        if c_act1.button("🗑️ SVUOTA TUTTO"):
            st.session_state['cart'] = []
            st.rerun()
            
        if len(st.session_state['cart']) > 0:
            st.divider()
            
            # LOOP SUL CARRELLO
            for i, item in enumerate(st.session_state['cart']):
                # Calcoli
                xg = calculate_stats(item['DF'], item['Home'], item['Away'])
                
                if xg:
                    xg_h, xg_a = xg
                    probs = predict_match(xg_h, xg_a)
                    
                    # Determinazione Pronostico Migliore
                    options = {
                        "1": probs['1'], "X": probs['X'], "2": probs['2'],
                        "OVER 2.5": probs['Over2.5'], "UNDER 2.5": probs['Under2.5'],
                        "GOAL": probs['GG'], "NO GOAL": probs['NG']
                    }
                    best_tip = max(options, key=options.get)
                    best_prob = options[best_tip]
                    
                    # Calcolo Stake
                    stake = kelly_criterion(best_prob, item['Odds'], budget) if item['Odds'] > 1 else 0
                    
                    # --- RENDER CARD (HTML/CSS) ---
                    st.markdown(f"""
                    <div class="match-card">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                            <span style="color:#888; font-size:12px;">{item['League']}</span>
                            <span style="color:#00e676; font-weight:bold;">xG: {xg_h:.2f} - {xg_a:.2f}</span>
                        </div>
                        <h3 style="margin:0; text-align:center; color:#fff;">{item['Home']} <span style="color:#666">vs</span> {item['Away']}</h3>
                        <hr style="border-color:#333; margin:15px 0;">
                        
                        <div style="display:flex; justify-content:space-around;">
                            <div class="stat-box">
                                <div class="stat-label">CONSIGLIO</div>
                                <div class="stat-value stat-green">{best_tip}</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">SICUREZZA</div>
                                <div class="stat-value stat-blue">{best_prob*100:.1f}%</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-label">STAKE</div>
                                <div class="stat-value stat-yellow">€{stake:.2f}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Dettagli Espandibili
                    with st.expander(f"📊 Dettagli Probabilità: {item['Home']} vs {item['Away']}"):
                        c1, c2, c3 = st.columns(3)
                        c1.progress(probs['1'], f"1: {probs['1']*100:.1f}%")
                        c1.progress(probs['X'], f"X: {probs['X']*100:.1f}%")
                        c1.progress(probs['2'], f"2: {probs['2']*100:.1f}%")
                        
                        c2.progress(probs['Over2.5'], f"Over 2.5: {probs['Over2.5']*100:.1f}%")
                        c2.progress(probs['Under2.5'], f"Under 2.5: {probs['Under2.5']*100:.1f}%")
                        
                        c3.progress(probs['GG'], f"Goal: {probs['GG']*100:.1f}%")
                        c3.progress(probs['NG'], f"No Goal: {probs['NG']*100:.1f}%")
                
                else:
                    st.error(f"Dati insufficienti per calcolare {item['Home']} vs {item['Away']}")

# Footer
st.markdown("<div style='text-align:center; color:#444; margin-top:50px;'>Betting Master AI v5.0 • Powered by Poisson Distribution</div>", unsafe_allow_html=True)
