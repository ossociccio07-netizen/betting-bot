import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io

# ==============================================================================
# 1. CONFIGURAZIONE & DESIGN (DARK NEON FIX)
# ==============================================================================
st.set_page_config(page_title="BETTING MASTER AI", page_icon="💎", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    /* Intestazioni */
    h1 { color: #00e676; font-family: sans-serif; text-transform: uppercase; }
    h3 { color: #00b0ff; }
    
    /* CARD STYLES */
    .match-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .card-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 10px;
        font-size: 12px;
        color: #888;
    }
    
    .card-title {
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin: 10px 0;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin-top: 15px;
    }
    
    .stat-box {
        text-align: center;
        background-color: #2b2b2b;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 10px;
        width: 30%;
    }
    
    .stat-label {
        font-size: 10px;
        color: #aaa;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    
    .stat-val {
        font-size: 18px;
        font-weight: 900;
    }
    
    /* Colori Neon */
    .neon-green { color: #00e676; text-shadow: 0 0 8px rgba(0,230,118,0.4); }
    .neon-blue { color: #2979ff; }
    .neon-yellow { color: #ffea00; }
    
    /* Pulsanti */
    .stButton>button {
        background: linear-gradient(90deg, #00c853, #64dd17);
        color: black;
        border: none;
        font-weight: bold;
        border-radius: 5px;
    }
    
    [data-testid="stSidebar"] { background-color: #111; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATABASE (LINK 2024/2025)
# ==============================================================================
LEAGUES = {
    "Serie A (Italia)": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "Premier League (UK)": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "La Liga (Spagna)": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "Bundesliga (Germania)": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "Ligue 1 (Francia)": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "Eredivisie (Olanda)": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "Primeira Liga (Portogallo)": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
    "Jupiler League (Belgio)": "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
    "Premiership (Scozia)": "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
    "Super Lig (Turchia)": "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
    "Super League (Grecia)": "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
}

# ==============================================================================
# 3. ENGINE (POISSON)
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
        h_att = df[df['HomeTeam'] == home]['FTHG'].mean() / avg_h
        h_def = df[df['HomeTeam'] == home]['FTAG'].mean() / avg_a
        a_att = df[df['AwayTeam'] == away]['FTAG'].mean() / avg_a
        a_def = df[df['AwayTeam'] == away]['FTHG'].mean() / avg_h
        
        if any(pd.isna([h_att, h_def, a_att, a_def])): return None
        return (h_att * a_def * avg_h), (a_att * h_def * avg_a)
    except: return None

def predict_match(xg_h, xg_a):
    max_goals = 6
    matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            matrix[i][j] = poisson.pmf(i, xg_h) * poisson.pmf(j, xg_a)
    
    p_1 = np.sum(np.tril(matrix, -1))
    p_x = np.sum(np.diag(matrix))
    p_2 = np.sum(np.triu(matrix, 1))
    
    p_over = 0
    p_gg = 0
    for i in range(max_goals):
        for j in range(max_goals):
            if i + j > 2.5: p_over += matrix[i][j]
            if i > 0 and j > 0: p_gg += matrix[i][j]

    return {"1": p_1, "X": p_x, "2": p_2, "OVER": p_over, "UNDER": 1-p_over, "GG": p_gg, "NG": 1-p_gg}

def get_stake(prob, bankroll):
    if prob <= 0.50: return 0.0
    factor = (prob - 0.50) * 2 
    return round(bankroll * (factor * 0.20), 2)

# ==============================================================================
# 4. APP
# ==============================================================================
with st.sidebar:
    st.title("IMPOSTAZIONI")
    budget = st.number_input("BUDGET (€)", value=100.0, step=10.0)
    st.divider()
    st.write("Seleziona campionato e squadre.")

st.title("💎 BETTING MASTER AI")

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}

c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("1. Seleziona Match")
    with st.container(border=True):
        lg = st.selectbox("Campionato", list(LEAGUES.keys()))
        
        if lg not in st.session_state['data_cache']:
            with st.spinner("Caricamento..."):
                df = load_data(LEAGUES[lg])
                if df is not None: st.session_state['data_cache'][lg] = df
        
        curr_df = st.session_state['data_cache'].get(lg)
        
        if curr_df is not None:
            tms = sorted(curr_df['HomeTeam'].unique())
            ht = st.selectbox("Casa", tms)
            at = st.selectbox("Ospite", tms, index=1)
            
            if st.button("AGGIUNGI", type="primary"):
                if ht != at:
                    st.session_state['cart'].append({"L": lg, "H": ht, "A": at, "DF": curr_df})
                    st.success("Aggiunto!")
                else: st.error("Squadre uguali!")
        else: st.error("Errore Dati.")

with c2:
    st.subheader(f"2. Schedina ({len(st.session_state['cart'])})")
    
    if st.session_state['cart']:
        if st.button("SVUOTA"):
            st.session_state['cart'] = []
            st.rerun()
        
        for item in st.session_state['cart']:
            xg = calculate_stats(item['DF'], item['H'], item['A'])
            if xg:
                xh, xa = xg
                probs = predict_match(xh, xa)
                
                # Strategia Migliore
                opts = {"1": probs['1'], "X": probs['X'], "2": probs['2'], "OVER 2.5": probs['OVER'], "GOAL": probs['GG']}
                best_t = max(opts, key=opts.get)
                best_p = opts[best_t]
                stake = get_stake(best_p, budget)
                
                # --- HTML CARD (SENZA INDENTAZIONE PERICOLOSA) ---
                html = f"""
<div class="match-card">
    <div class="card-header">
        <span>{item['L']}</span>
        <span style="color:#00e676">xG: {xh:.2f} - {xa:.2f}</span>
    </div>
    <div class="card-title">{item['H']} <span style="color:#666">vs</span> {item['A']}</div>
    <hr style="border-color:#333; margin:10px 0;">
    <div class="stats-container">
        <div class="stat-box">
            <div class="stat-label">STRATEGIA</div>
            <div class="stat-val neon-green">{best_t}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">SICUREZZA</div>
            <div class="stat-val neon-blue">{best_p*100:.1f}%</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">PUNTA</div>
            <div class="stat-val neon-yellow">€{stake}</div>
        </div>
    </div>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)
                
                with st.expander("📊 Dettagli Probabilità"):
                    k1, k2, k3 = st.columns(3)
                    k1.write("1X2")
                    k1.progress(probs['1'], f"1: {probs['1']*100:.0f}%")
                    k1.progress(probs['X'], f"X: {probs['X']*100:.0f}%")
                    k1.progress(probs['2'], f"2: {probs['2']*100:.0f}%")
                    
                    k2.write("Goals")
                    k2.progress(probs['OVER'], f"Over 2.5: {probs['OVER']*100:.0f}%")
                    k2.progress(probs['UNDER'], f"Under 2.5: {probs['UNDER']*100:.0f}%")
                    
                    k3.write("GG/NG")
                    k3.progress(probs['GG'], f"Goal: {probs['GG']*100:.0f}%")
                    k3.progress(probs['NG'], f"NoGoal: {probs['NG']*100:.0f}%")
            else:
                st.error("Dati insufficienti.")
    else:
        st.info("Aggiungi partite dal menu a sinistra.")
