import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta
import difflib

# ==============================================================================
# 0. CONFIGURAZIONE & STILE (NERO & VERDE)
# ==============================================================================
st.set_page_config(page_title="BETTING PRO: HEAVY DUTY", page_icon="☣️", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    h1, h2, h3 { color: #00ff00 !important; font-family: 'Courier New', monospace; }
    .stButton>button { background-color: #111; color: #00ff00; border: 1px solid #00ff00; }
    .stButton>button:hover { background-color: #00ff00; color: #000; }
    
    /* Box Risultati */
    div[data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid #333; background-color: #111; padding: 15px; border-radius: 5px;
    }
    
    /* Metriche */
    [data-testid="stMetricLabel"] { color: #888; }
    [data-testid="stMetricValue"] { font-family: monospace; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. IL GRANDE TRADUTTORE (NORMALIZZAZIONE NOMI)
# ==============================================================================
TEAM_ALIAS = {
    # PREMIER LEAGUE
    "Man Utd": "Man United", "Manchester Utd": "Man United", "Manchester United": "Man United", "Man. Utd": "Man United",
    "Man City": "Man City", "Manchester City": "Man City", "Man. City": "Man City",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "Newcastle": "Newcastle United", "Newcastle Utd": "Newcastle United",
    "West Ham": "West Ham United", "West Ham Utd": "West Ham United",
    "Wolves": "Wolverhampton", "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton": "Brighton & Hove Albion", "Brighton & Hove": "Brighton & Hove Albion",
    "Nott'm Forest": "Nottm Forest", "Nottingham Forest": "Nottm Forest", "Nottingham": "Nottm Forest",
    "Luton": "Luton Town", "Sheffield Utd": "Sheffield United", "Leicester": "Leicester City",
    "Leeds": "Leeds United", "Ipswich": "Ipswich Town",
    
    # SERIE A
    "Inter": "Internazionale", "Internazionale Milano": "Internazionale",
    "Milan": "AC Milan", "A.C. Milan": "AC Milan",
    "Roma": "AS Roma", "A.S. Roma": "AS Roma",
    "Verona": "Hellas Verona", "H. Verona": "Hellas Verona",
    "Monza": "AC Monza",
    
    # LIGA
    "Ath Madrid": "Athletico Madrid", "Atletico Madrid": "Athletico Madrid",
    "Real Madrid": "Real Madrid",
    "Barcelona": "Barcelona", "FC Barcelona": "Barcelona",
    "Betis": "Real Betis", "Sevilla": "Sevilla FC",
    "Sociedad": "Real Sociedad",
    
    # BUNDESLIGA
    "Bayern Munich": "Bayern Munchen", "Bayern": "Bayern Munchen",
    "Dortmund": "Borussia Dortmund", "B. Dortmund": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen", "B. Leverkusen": "Bayer Leverkusen",
    "Leipzig": "RB Leipzig", "R.B. Leipzig": "RB Leipzig",
    "Mainz": "Mainz 05", "Frankfurt": "Eintracht Frankfurt"
}

def normalize_name(name):
    """Pulisce e standardizza il nome della squadra"""
    name = str(name).strip()
    return TEAM_ALIAS.get(name, name)

# ==============================================================================
# 2. MOTORE DI RICERCA DATI (MULTI-SOURCE)
# ==============================================================================
# Link aggiornati alla STAGIONE REALE 2024/2025
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
    """Prova una lista di URL finché non ne trova uno funzionante"""
    for url in url_list:
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            if r.status_code == 200:
                df = pd.read_csv(io.StringIO(r.text))
                if not df.empty:
                    return df, url 
        except:
            continue
    return None, None

# ==============================================================================
# 3. MOTORE MATEMATICO (POISSON)
# ==============================================================================
def calculate_poisson(h_team, a_team, df_hist):
    try:
        # 1. Pulisci dati
        df = df_hist[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        
        # 2. Calcola Medie Campionato
        avg_home_goal = df['FTHG'].mean()
        avg_away_goal = df['FTAG'].mean()
        
        # 3. Calcola Forza Squadre
        stats_h = df[df['HomeTeam'] == h_team]['FTHG'].mean()
        conc_h = df[df['HomeTeam'] == h_team]['FTAG'].mean()
        stats_a = df[df['AwayTeam'] == a_team]['FTAG'].mean()
        conc_a = df[df['AwayTeam'] == a_team]['FTHG'].mean()
        
        if pd.isna(stats_h) or pd.isna(stats_a): return None 

        # 4. Parametri Lambda
        att_h = stats_h / avg_home_goal
        def_h = conc_h / avg_away_goal
        att_a = stats_a / avg_away_goal
        def_a = conc_a / avg_home_goal
        
        lambda_h = att_h * def_a * avg_home_goal
        lambda_a = att_a * def_h * avg_away_goal
        
        # 5. Distribuzione Poisson
        ph, pd, pa = 0, 0, 0
        p_over25 = 0
        p_goal = 0
        
        for i in range(6):
            for j in range(6):
                prob = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
                if i > j: ph += prob
                elif i == j: pd += prob
                else: pa += prob
                
                if (i+j) > 2.5: p_over25 += prob
                if i > 0 and j > 0: p_goal += prob
                
        # 6. Risultato
        best_prob = max(ph, pa)
        prediction = "1" if ph > pa else "2"
        
        return {
            "1": ph, "X": pd, "2": pa,
            "Tip": prediction, "Prob": best_prob, "FairOdd": 1/best_prob,
            "Over25": p_over25, "Goal": p_goal
        }
    except Exception as e:
        return None

# ==============================================================================
# 4. INTERFACCIA UTENTE (UI)
# ==============================================================================
st.title("☣️ BETTING PRO: SCANNER")
st.caption("Sistema di analisi Multi-Source per trovare partite ovunque.")

budget = st.number_input("Budget (€)", value=100.0)
st.divider()

col_scan, col_manual = st.columns([2, 1])

# --- SEZIONE SCANNER AUTOMATICO ---
with col_scan:
    st.subheader("📡 RADAR MATCH")
    
    # Selettore Data (Default: Oggi)
    scan_date = st.date_input("Cerca partite dal:", datetime.now())
    
    if st.button("🚀 AVVIA SCANSIONE PROFONDA", type="primary"):
        log_container = st.expander("📜 Log Operazioni", expanded=True)
        results_container = st.container()
        
        found_matches = 0
        
        with log_container:
            st.write(f"Inizio scansione database dal {scan_date}...")
            
            for league_name, urls in LEAGUES.items():
                st.write(f"**{league_name}**...")
                
                # 1. Scarica Storico
                df_hist, used_url = fetch_robust_data(urls['hist'])
                if df_hist is None:
                    st.error(f"❌ Fallito download Storico")
                    continue
                
                # Normalizza i nomi nello storico
                df_hist['HomeTeam'] = df_hist['HomeTeam'].apply(normalize_name)
                df_hist['AwayTeam'] = df_hist['AwayTeam'].apply(normalize_name)
                valid_teams = df_hist['HomeTeam'].unique().tolist()

                # 2. Scarica Calendario
                df_fix, used_url_fix = fetch_robust_data(urls['fix'])
                if df_fix is None:
                    st.warning(f"⚠️ Calendario offline. Provo storico...")
                    df_fix = df_hist[df_hist['FTHG'].isna()].copy()

                # 3. Filtra per Data
                if not df_fix.empty:
                    date_col = next((c for c in df_fix.columns if 'Date' in c or 'Time' in c), None)
                    if date_col:
                        df_fix['DT_CLEAN'] = pd.to_datetime(df_fix[date_col], dayfirst=True, errors='coerce').dt.date
                        # PRENDIAMO LE PROSSIME 10 PARTITE
                        matches = df_fix[df_fix['DT_CLEAN'] >= scan_date].sort_values('DT_CLEAN').head(10)
                        
                        if matches.empty:
                            st.info(f"Nessuna partita futura trovata.")
                        else:
                            st.success(f"Trovate {len(matches)} partite future!")
                            
                            for _, match in matches.iterrows():
                                raw_h = match.get('Home Team', match.get('HomeTeam', 'Unknown'))
                                raw_a = match.get('Away Team', match.get('AwayTeam', 'Unknown'))
                                match_d = match['DT_CLEAN']
                                
                                real_h = normalize_name(raw_h)
                                real_a = normalize_name(raw_a)
                                
                                # Fuzzy Match
                                if real_h not in valid_teams:
                                    matches_f = difflib.get_close_matches(real_h, valid_teams, n=1, cutoff=0.6)
                                    if matches_f: real_h = matches_f[0]
                                if real_a not in valid_teams:
                                    matches_f = difflib.get_close_matches(real_a, valid_teams, n=1, cutoff=0.6)
                                    if matches_f: real_a = matches_f[0]

                                # Calcola
                                res = calculate_poisson(real_h, real_a, df_hist)
                                
                                if res:
                                    # FILTRO SICUREZZA 50%
                                    if res['Prob'] > 0.50:
                                        found_matches += 1
                                        with results_container:
                                            with st.container():
                                                st.markdown(f"#### ⚽ {real_h} vs {real_a}")
                                                st.caption(f"{league_name} | {match_d}")
                                                
                                                c1, c2, c3, c4 = st.columns(4)
                                                c1.metric("PUNTA", res['Tip'], f"{res['Prob']*100:.0f}%")
                                                c2.metric("Q. REALE
