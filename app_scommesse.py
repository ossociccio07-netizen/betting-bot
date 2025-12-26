import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io

# ==============================================================================
# 1. CONFIGURAZIONE & STILE (NEON DARK MODE)
# ==============================================================================
st.set_page_config(page_title="BETTING PRO 1X2", page_icon="⚽", layout="wide")

st.markdown("""
<style>
    /* Sfondo Nero Totale */
    .stApp { background-color: #000000; color: #e0e0e0; }
    
    /* Nascondi Menu Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* TITOLI */
    h1 { color: #00ff00 !important; font-family: sans-serif; text-transform: uppercase; letter-spacing: 2px; }
    h3 { color: #00bfff !important; }
    
    /* CARD PARTITA */
    .match-card {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 15px rgba(0,0,0,0.5);
    }
    
    /* HEADER DELLA CARD */
    .card-header {
        display: flex; justify-content: space-between; color: #666; font-size: 12px; margin-bottom: 10px;
    }
    
    /* TITOLO MATCH */
    .match-title {
        text-align: center; font-size: 24px; font-weight: 900; color: white; margin: 10px 0;
    }
    
    /* STATS CONTAINER */
    .stats-row {
        display: flex; justify-content: space-around; align-items: center; margin-top: 15px;
    }
    
    /* BOX DEI NUMERI */
    .stat-box {
        text-align: center; background-color: #1a1a1a; border: 1px solid #444;
        border-radius: 8px; padding: 10px; width: 30%;
    }
    
    .stat-label { font-size: 10px; color: #888; letter-spacing: 1px; }
    .stat-val { font-size: 20px; font-weight: bold; }
    
    /* COLORI SPECIFICI */
    .green-neon { color: #00ff00; text-shadow: 0 0 10px rgba(0,255,0,0.4); }
    .blue-neon { color: #00bfff; }
    .white-val { color: #ffffff; }
    
    /* BOTTONI */
    .stButton>button {
        background-color: #222; color: #00ff00; border: 1px solid #00ff00;
        border-radius: 5px; font-weight: bold; width: 100%;
    }
    .stButton>button:hover {
        background-color: #00ff00; color: black;
    }
    
    /* BARRE PROBABILITA */
    .stProgress > div > div > div > div { background-color: #00bfff; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATABASE (Link corretti 2024/2025)
# ==============================================================================
LEAGUES = {
    "🇮🇹 Serie A": "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
    "🇬🇧 Premier League": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
    "🇪🇸 La Liga": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
    "🇩🇪 Bundesliga": "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
    "🇫🇷 Ligue 1": "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
    "🇳🇱 Eredivisie": "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
    "🇵🇹 Primeira Liga": "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
}

# ==============================================================================
# 3. MOTORE MATEMATICO
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
    f = (prob - 0.50) * 2 
    return round(bankroll * (f * 0.20), 2)

# ==============================================================================
# 4. INTERFACCIA
# ==============================================================================
with st.sidebar:
    st.title("IMPOSTAZIONI")
    budget = st.number_input("Budget (€)", value=100.0, step=10.0)
    st.info("Seleziona Campionato e Squadre per l'analisi.")

st.title("⚽ BETTING PRO 1X2")

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'data_cache' not in st.session_state: st.session_state['data_cache'] = {}

# LAYOUT A DUE COLONNE
c_sel, c_res = st.columns([1, 2])

with c_sel:
    st.subheader("1. Crea Schedina")
    with st.container(border=True):
        lg = st.selectbox("Campionato", list(LEAGUES.keys()))
        
        if lg not in st.session_state['data_cache']:
            with st.spinner("Scarico dati..."):
                df = load_data(LEAGUES[lg])
                if df is not None: st.session_state['data_cache'][lg] = df
        
        curr_df = st.session_state['data_cache'].get(lg)
        
        if curr_df is not None:
            tms = sorted(curr_df['HomeTeam'].unique())
            ht = st.selectbox("Casa", tms)
            at = st.selectbox("Ospite", tms, index=1)
            
            if st.button("AGGIUNGI AL CARRELLO", type="primary"):
                if ht != at:
                    st.session_state['cart'].append({"L": lg, "H": ht, "A": at, "DF": curr_df})
                    st.success("Aggiunto!")
                else: st.error("Squadre uguali!")
        else: st.error("Errore Link.")

with c_res:
    st.subheader(f"2. Analisi ({len(st.session_state['cart'])})")
    
    if st.session_state['cart']:
        if st.button("SVUOTA TUTTO"):
            st.session_state['cart'] = []
            st.rerun()
        
        for item in st.session_state['cart']:
            xg = calculate_stats(item['DF'], item['H'], item['A'])
            if xg:
                xh, xa = xg
                probs = predict_match(xh, xa)
                
                # Strategia
                opts = {"1": probs['1'], "X": probs['X'], "2": probs['2'], 
                        "OVER 2.5": probs['OVER'], "UNDER 2.5": probs['UNDER'], 
                        "GOAL": probs['GG'], "NO GOAL": probs['NG']}
                
                best_t = max(opts, key=opts.get)
                best_p = opts[best_t]
                stake = get_stake(best_p, budget)
                
                # Favorito 1X2 per visualizzazione
                probs_1x2 = {"1": probs['1'], "X": probs['X'], "2": probs['2']}
                fav_1x2 = max(probs_1x2, key=probs_1x2.get)
                fav_1x2_prob = probs_1x2[fav_1x2]
                
                # HTML CARD PULITO
                html = f"""
<div class="match-card">
    <div class="card-header">
        <span>{item['L']}</span>
        <span style="color:#00ff00">xG: {xh:.2f} - {xa:.2f}</span>
    </div>
    <div class="match-title">{item['H']} <span style="color:#666">vs</span> {item['A']}</div>
    <hr style="border-color:#333; margin:10px 0;">
    
    <div class="stats-row">
        <div class="stat-box">
            <div class="stat-label">STRATEGIA</div>
            <div class="stat-val green-neon">{best_t}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">1X2</div>
            <div class="stat-val blue-neon">{fav_1x2} ({fav_1x2_prob*100:.0f}%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">PUNTA</div>
            <div class="stat-val white-val">€{stake}</div>
        </div>
    </div>
</div>
"""
                st.markdown(html, unsafe_allow_html=True)
                
                # Dettagli
                with st.expander(f"Dettagli {item['H']} - {item['A']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.write("Esito 1X2")
                    c1.progress(probs['1'], f"1: {probs['1']*100:.0f}%")
                    c1.progress(probs['X'], f"X: {probs['X']*100:.0f}%")
                    c1.progress(probs['2'], f"2: {probs['2']*100:.0f}%")
                    
                    c2.write("Under/Over 2.5")
                    c2.progress(probs['OVER'], f"Over: {probs['OVER']*100:.0f}%")
                    c2.progress(probs['UNDER'], f"Under: {probs['UNDER']*100:.0f}%")
                    
                    c3.write("Goal/NoGoal")
                    c3.progress(probs['GG'], f"Goal: {probs['GG']*100:.0f}%")
                    c3.progress(probs['NG'], f"NoGoal: {probs['NG']*100:.0f}%")
            else:
                st.error("Dati insufficienti.")
    else:
        st.info("Aggiungi partite dal menu a sinistra.")
