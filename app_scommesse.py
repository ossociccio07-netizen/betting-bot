import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta
import difflib

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
st.set_page_config(page_title="BETTING PRO: HEAVY DUTY", page_icon="☣️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1, h2, h3 { color: #00ff00 !important; font-family: 'Courier New', monospace; }
    .stButton>button { background-color: #111; color: #00ff00; border: 1px solid #00ff00; }
    
    div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #333; background-color: #111; padding: 15px; border-radius: 5px;
    }
    [data-testid="stMetricValue"] { font-family: monospace; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATI & NORMALIZZAZIONE
# ==============================================================================
TEAM_ALIAS = {
    "Man Utd": "Man United", "Manchester Utd": "Man United", "Manchester United": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Newcastle": "Newcastle United", "West Ham": "West Ham United",
    "Wolves": "Wolverhampton", "Brighton": "Brighton & Hove Albion",
    "Nott'm Forest": "Nottm Forest", "Nottingham Forest": "Nottm Forest",
    "Luton": "Luton Town", "Sheffield Utd": "Sheffield United",
    "Inter": "Internazionale", "Milan": "AC Milan", "Roma": "AS Roma",
    "Verona": "Hellas Verona", "Monza": "AC Monza",
    "Ath Madrid": "Athletico Madrid", "Atletico Madrid": "Athletico Madrid",
    "Betis": "Real Betis", "Sociedad": "Real Sociedad",
    "Bayern Munich": "Bayern Munchen", "Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen", "Leipzig": "RB Leipzig"
}

def normalize_name(name):
    name = str(name).strip()
    return TEAM_ALIAS.get(name, name)

# LINK STAGIONE 2024-2025
LEAGUES = {
    "🇬🇧 Premier League": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2425/E0.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2024/england-premier-league-2024.csv"]
    },
    "🇮🇹 Serie A": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2425/I1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2024/italy-serie-a-2024.csv"]
    },
    "🇪🇸 La Liga": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2425/SP1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2024/spain-la-liga-2024.csv"]
    },
    "🇩🇪 Bundesliga": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2425/D1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2024/germany-bundesliga-2024.csv"]
    },
    "🇫🇷 Ligue 1": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2425/F1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2024/france-ligue-1-2024.csv"]
    }
}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_robust_data(url_list):
    for url in url_list:
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty: return df, url
        except: continue
    return None, None

# ==============================================================================
# CALCOLO
# ==============================================================================
def calculate_poisson(h_team, a_team, df_hist):
    try:
        df = df_hist[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        avg_h_goal = df['FTHG'].mean()
        avg_a_goal = df['FTAG'].mean()
        
        stats_h = df[df['HomeTeam'] == h_team]['FTHG'].mean()
        conc_h = df[df['HomeTeam'] == h_team]['FTAG'].mean()
        stats_a = df[df['AwayTeam'] == a_team]['FTAG'].mean()
        conc_a = df[df['AwayTeam'] == a_team]['FTHG'].mean()
        
        if pd.isna(stats_h) or pd.isna(stats_a): return None

        att_h = stats_h / avg_h_goal
        def_h = conc_h / avg_a_goal
        att_a = stats_a / avg_a_goal
        def_a = conc_a / avg_h_goal
        
        lambda_h = att_h * def_a * avg_h_goal
        lambda_a = att_a * def_h * avg_a_goal
        
        ph, pd_draw, pa = 0, 0, 0
        p_over25 = 0
        
        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                if i > j: ph += prob
                elif i == j: pd_draw += prob
                else: pa += prob
                if (i+j) > 2.5: p_over25 += prob
                
        best_prob = max(ph, pa)
        prediction = "1" if ph > pa else "2"
        
        return {
            "Tip": prediction, "Prob": best_prob, "FairOdd": 1/best_prob if best_prob>0 else 0,
            "Over25": p_over25
        }
    except: return None

# ==============================================================================
# UI
# ==============================================================================
st.title("☣️ BETTING PRO: SCANNER")
budget = st.number_input("Budget (€)", value=100.0)
st.divider()

col_scan, col_manual = st.columns([2, 1])

with col_scan:
    st.subheader("📡 RADAR MATCH")
    scan_date = st.date_input("Cerca partite dal:", datetime.now())
    
    if st.button("🚀 AVVIA SCANSIONE", type="primary"):
        log = st.expander("Log Operazioni", expanded=False)
        res_container = st.container()
        found = 0
        
        with log:
            for league, urls in LEAGUES.items():
                st.write(f"Analisi {league}...")
                
                # 1. Storico
                df_hist, _ = fetch_robust_data(urls['hist'])
                if df_hist is None: continue
                
                df_hist['HomeTeam'] = df_hist['HomeTeam'].apply(normalize_name)
                df_hist['AwayTeam'] = df_hist['AwayTeam'].apply(normalize_name)
                valid_teams = df_hist['HomeTeam'].unique().tolist()
                
                # 2. Calendario
                df_fix, _ = fetch_robust_data(urls['fix'])
                if df_fix is None:
                    # Fallback
                    df_fix = df_hist[df_hist['FTHG'].isna()].copy()
                
                if not df_fix.empty:
                    d_col = next((c for c in df_fix.columns if 'Date' in c or 'Time' in c), None)
                    if d_col:
                        df_fix['DT'] = pd.to_datetime(df_fix[d_col], dayfirst=True, errors='coerce').dt.date
                        # Prendiamo 5 partite future
                        matches = df_fix[df_fix['DT'] >= scan_date].sort_values('DT').head(5)
                        
                        for _, m in matches.iterrows():
                            raw_h = m.get('Home Team', m.get('HomeTeam', '')).strip()
                            raw_a = m.get('Away Team', m.get('AwayTeam', '')).strip()
                            m_date = m['DT']
                            
                            rh = normalize_name(raw_h)
                            ra = normalize_name(raw_a)
                            
                            # Fuzzy Match
                            if rh not in valid_teams:
                                x = difflib.get_close_matches(rh, valid_teams, n=1, cutoff=0.6)
                                if x: rh = x[0]
                            if ra not in valid_teams:
                                x = difflib.get_close_matches(ra, valid_teams, n=1, cutoff=0.6)
                                if x: ra = x[0]
                                
                            res = calculate_poisson(rh, ra, df_hist)
                            
                            if res and res['Prob'] > 0.50:
                                found += 1
                                with res_container:
                                    with st.container():
                                        st.markdown(f"#### {rh} vs {ra}")
                                        st.caption(f"{league} | {m_date}")
                                        c1, c2, c3, c4 = st.columns(4)
                                        c1.metric("PUNTA", res['Tip'], f"{res['Prob']*100:.0f}%")
                                        c2.metric("QUOTA", f"{res['FairOdd']:.2f}")
                                        c3.metric("OVER 2.5", f"{res['Over25']*100:.0f}%")
                                        stake = max(0, (res['Prob']-(1-res['Prob']))*budget*0.1)
                                        c4.metric("STAKE", f"€{stake:.2f}")
                                        st.divider()

        if found == 0: st.warning("Nessuna occasione trovata.")

with col_manual:
    st.subheader("🛠️ MANUALE")
    sl = st.selectbox("League", list(LEAGUES.keys()))
    if st.button("CARICA SQUADRE"):
        df, _ = fetch_robust_data(LEAGUES[sl]['hist'])
        if df is not None:
            df['HomeTeam'] = df['HomeTeam'].apply(normalize_name)
            st.session_state['tm'] = sorted(df['HomeTeam'].unique())
            st.session_state['df'] = df
            st.rerun()
            
    if 'tm' in st.session_state:
        h = st.selectbox("Casa", st.session_state['tm'])
        a = st.selectbox("Ospite", st.session_state['tm'], index=1)
        if st.button("CALCOLA"):
            r = calculate_poisson(h, a, st.session_state['df'])
            if r:
                st.success(f"PUNTA {r['Tip']} ({r['Prob']*100:.1f}%)")
                st.progress(r['Prob'])
