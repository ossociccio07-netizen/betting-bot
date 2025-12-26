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
# Questo dizionario unifica i nomi diversi usati dai vari siti di dati
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
    return TEAM_ALIAS.get(name, name) # Se c'è nel dizionario usa quello, sennò lascia originale

# ==============================================================================
# 2. MOTORE DI RICERCA DATI (MULTI-SOURCE)
# ==============================================================================
# Definiamo più URL per ogni campionato. Se uno fallisce, proviamo l'altro.
# Usiamo i link per la stagione 2025/2026 (dato che oggi è Dic 2025)
LEAGUES = {
    "🇬🇧 Premier League": {
        "hist": [
            "https://www.football-data.co.uk/mmz4281/2526/E0.csv", # Principale
            "https://www.football-data.co.uk/mmz4281/2425/E0.csv"  # Fallback stagione passata
        ],
        "fix": [
            "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv", # Fonte A
            "https://fixturedownload.com/download/csv/2026/england-premier-league-2026.csv", # Fonte B (Futuro)
        ]
    },
    "🇮🇹 Serie A": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2526/I1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"]
    },
    "🇪🇸 La Liga": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2526/SP1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"]
    },
    "🇩🇪 Bundesliga": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2526/D1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"]
    },
    "🇫🇷 Ligue 1": {
        "hist": ["https://www.football-data.co.uk/mmz4281/2526/F1.csv"],
        "fix": ["https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"]
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
                    return df, url # Ritorna i dati e l'URL vincente
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
        
        # 3. Calcola Forza Squadre (Ultimi 5 match pesano di più)
        # Semplificazione per robustezza: usiamo media totale
        stats_h = df[df['HomeTeam'] == h_team]['FTHG'].mean()
        conc_h = df[df['HomeTeam'] == h_team]['FTAG'].mean()
        stats_a = df[df['AwayTeam'] == a_team]['FTAG'].mean()
        conc_a = df[df['AwayTeam'] == a_team]['FTHG'].mean()
        
        if pd.isna(stats_h) or pd.isna(stats_a): return None # Dati insufficienti

        # 4. Parametri Lambda (Attacco * Difesa / Media)
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
        
        # Fair Odds (Quote Reali)
        odd_1 = 1/ph if ph>0 else 0
        odd_2 = 1/pa if pa>0 else 0
        
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

# Input Utente
budget = st.number_input("Budget (€)", value=100.0)
st.divider()

col_scan, col_manual = st.columns([2, 1])

# --- SEZIONE SCANNER AUTOMATICO ---
with col_scan:
    st.subheader("📡 RADAR MATCH")
    
    # Selettore Data (Default: Oggi)
    scan_date = st.date_input("Cerca partite dal:", datetime.now())
    
    if st.button("🚀 AVVIA SCANSIONE PROFONDA", type="primary"):
        log_container = st.expander("📜 Log Operazioni (Click per vedere i dettagli)", expanded=True)
        results_container = st.container()
        
        found_matches = 0
        
        with log_container:
            st.write(f"Inizio scansione database per data >= {scan_date}...")
            
            for league_name, urls in LEAGUES.items():
                st.write(f"--- **Analisi {league_name}** ---")
                
                # 1. Scarica Storico (History)
                df_hist, used_url = fetch_robust_data(urls['hist'])
                if df_hist is None:
                    st.error(f"❌ Fallito download Storico per {league_name}")
                    continue
                else:
                    st.write(f"✅ Storico scaricato.")
                    # Normalizza i nomi nello storico per facilitare il match
                    df_hist['HomeTeam'] = df_hist['HomeTeam'].apply(normalize_name)
                    df_hist['AwayTeam'] = df_hist['AwayTeam'].apply(normalize_name)
                    valid_teams = df_hist['HomeTeam'].unique().tolist()

                # 2. Scarica Calendario (Fixtures)
                df_fix, used_url_fix = fetch_robust_data(urls['fix'])
                if df_fix is None:
                    st.warning(f"⚠️ Calendario non disponibile per {league_name}. Provo a dedurre partite dallo storico...")
                    # FALLBACK: A volte lo storico ha righe con risultato vuoto (partite future)
                    df_fix = df_hist[df_hist['FTHG'].isna()].copy()
                else:
                    st.write(f"✅ Calendario scaricato.")

                # 3. Filtra per Data
                if not df_fix.empty:
                    # Cerca colonna data
                    date_col = next((c for c in df_fix.columns if 'Date' in c or 'Time' in c), None)
                    if date_col:
                        df_fix['DT_CLEAN'] = pd.to_datetime(df_fix[date_col], dayfirst=True, errors='coerce').dt.date
                        # PRENDIAMO TUTTE LE PARTITE FUTURE (Massimo 5 per campionato per non intasare)
                        matches = df_fix[df_fix['DT_CLEAN'] >= scan_date].sort_values('DT_CLEAN').head(5)
                        
                        if matches.empty:
                            st.info(f"Nessuna partita futura trovata in {league_name}.")
                        else:
                            st.success(f"Trovate {len(matches)} partite future!")
                            
                            # 4. ANALISI MATCH
                            for _, match in matches.iterrows():
                                # Estrai nomi
                                raw_h = match.get('Home Team', match.get('HomeTeam', 'Unknown'))
                                raw_a = match.get('Away Team', match.get('AwayTeam', 'Unknown'))
                                match_d = match['DT_CLEAN']
                                
                                # Normalizza
                                real_h = normalize_name(raw_h)
                                real_a = normalize_name(raw_a)
                                
                                # Fuzzy Match se la normalizzazione diretta fallisce
                                if real_h not in valid_teams:
                                    matches_f = difflib.get_close_matches(real_h, valid_teams, n=1, cutoff=0.6)
                                    if matches_f: real_h = matches_f[0]
                                if real_a not in valid_teams:
                                    matches_f = difflib.get_close_matches(real_a, valid_teams, n=1, cutoff=0.6)
                                    if matches_f: real_a = matches_f[0]

                                # Calcola
                                res = calculate_poisson(real_h, real_a, df_hist)
                                
                                if res:
                                    # FILTRO SICUREZZA: Mostra solo se prob > 50%
                                    if res['Prob'] > 0.50:
                                        found_matches += 1
                                        with results_container:
                                            with st.container():
                                                st.markdown(f"#### ⚽ {real_h} vs {real_a}")
                                                st.caption(f"{league_name} | {match_d}")
                                                
                                                c1, c2, c3, c4 = st.columns(4)
                                                c1.metric("PUNTA", res['Tip'], f"{res['Prob']*100:.0f}%")
                                                c2.metric("QUOTA REALE", f"{res['FairOdd']:.2f}")
                                                c3.metric("OVER 2.5", f"{res['Over25']*100:.0f}%")
                                                
                                                stake = (res['Prob'] - (1-res['Prob'])) * budget * 0.1
                                                stake = max(0, stake)
                                                c4.metric("STAKE", f"€{stake:.2f}")
                                                st.divider()

        if found_matches == 0:
            st.warning("Nessuna partita trovata con alta probabilità (>50%) o problemi di dati.")

# --- SEZIONE MANUALE (FALLBACK) ---
with col_manual:
    st.subheader("🛠️ MANUALE")
    sel_league = st.selectbox("Campionato", list(LEAGUES.keys()))
    
    # Carica dati al volo per il manuale
    df_man, _ = fetch_robust_data(LEAGUES[sel_league]['hist'])
    
    if df_man is not None:
        df_man['HomeTeam'] = df_man['HomeTeam'].apply(normalize_name)
        teams = sorted(df_man['HomeTeam'].unique())
        
        h_man = st.selectbox("Casa", teams)
        a_man = st.selectbox("Ospite", teams, index=1)
        
        if st.button("ANALIZZA"):
            res_man = calculate_poisson(h_man, a_man, df_man)
            if res_man:
                st.success(f"Consiglio: PUNTA {res_man['Tip']}")
                st.write(f"Sicurezza: {res_man['Prob']*100:.1f}%")
                st.progress(res_man['Prob'])
    else:
        st.error("Dati non disponibili.")import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta
import difflib

# ==============================================================================
# CONFIGURAZIONE PAGINA
# ==============================================================================
st.set_page_config(page_title="BETTING PRO 24/25", page_icon="⚽", layout="centered")
DEFAULT_BUDGET = 100.0

st.markdown("""
<style>
    .stApp { background-color: #000000; }
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #000; padding: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; background-color: #1a1a1a; border: 1px solid #333;
        border-radius: 8px; color: #888; font-weight: bold; font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d26a !important; color: #000 !important; border: none;
    }

    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] {
        color: #00ff00 !important; text-shadow: 0 0 10px rgba(0,255,0,0.4);
    }
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] {
        color: #00bfff !important;
    }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] {
        color: #ffffff !important; background-color: #222; border-radius: 5px; padding: 0 5px;
    }
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #111; border: 1px solid #222; border-radius: 12px; padding: 15px;
    }
    .stProgress > div > div > div > div { background-color: #00bfff; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE (LINK CORRETTI STAGIONE 2024/2025)
# ==============================================================================
DATABASE = [
    {
        "id": "E0", "nome": "🇬🇧 Premier League", 
        "history": "https://www.football-data.co.uk/mmz4281/2425/E0.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2024/england-premier-league-2024.csv"
    },
    {
        "id": "I1", "nome": "🇮🇹 Serie A", 
        "history": "https://www.football-data.co.uk/mmz4281/2425/I1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2024/italy-serie-a-2024.csv"
    },
    {
        "id": "SP1", "nome": "🇪🇸 La Liga", 
        "history": "https://www.football-data.co.uk/mmz4281/2425/SP1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2024/spain-la-liga-2024.csv"
    },
    {
        "id": "D1", "nome": "🇩🇪 Bundesliga", 
        "history": "https://www.football-data.co.uk/mmz4281/2425/D1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2024/germany-bundesliga-2024.csv"
    },
    {
        "id": "F1", "nome": "🇫🇷 Ligue 1", 
        "history": "https://www.football-data.co.uk/mmz4281/2425/F1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2024/france-ligue-1-2024.csv"
    }
]

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

# ==============================================================================
# FUNZIONI
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

def smart_match_name(name, known_teams):
    matches = difflib.get_close_matches(name, known_teams, n=1, cutoff=0.5)
    return matches[0] if matches else name

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        PS, PF = 0.60, 0.40
        sc = df.groupby('HomeTeam')[['FTHG','FTAG']].mean()
        st = df.groupby('AwayTeam')[['FTAG','FTHG']].mean()
        fc = df.groupby('HomeTeam')[['FTHG','FTAG']].apply(lambda x: x.tail(5).mean())
        ft = df.groupby('AwayTeam')[['FTAG','FTHG']].apply(lambda x: x.tail(5).mean())
        
        sc.columns, st.columns = ['H_GF_S','H_GS_S'], ['A_GF_S','A_GS_S']
        fc.columns, ft.columns = ['H_GF_F','H_GS_F'], ['A_GF_F','A_GS_F']
        
        tot = pd.concat([sc,st,fc,ft], axis=1)
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
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph if ph>0 else 0},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa if pa>0 else 0},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd if pd>0 else 0},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25 if p_o25>0 else 0},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg if p_gg>0 else 0}
        ]
        
        probs_1x2 = {"1": ph, "X": pd, "2": pa}
        fav_sign = max(probs_1x2, key=probs_1x2.get)
        
        # FILTRO SOGLIA: Mostra se la probabilità è > 50%
        valid = [o for o in options if o['Prob'] > 0.50]
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0}

        return {
            "c": h, "o": a, "Best": best, 
            "Fav_1X2": {"Label": "CASA" if fav_sign=="1" else "OSPITE" if fav_sign=="2" else "PARI", "Prob": probs_1x2[fav_sign]},
            "Probs": probs_1x2, "xG_H": lh, "xG_A": la
        }
    except: return None

def calculate_stake(prob, quota, bankroll):
    try:
        if quota <= 1: return 0
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        return round(max(0, bankroll * (f * 0.20)), 2)
    except: return 0

# ==============================================================================
# UI
# ==============================================================================
st.title("⚽ BETTING PRO 24/25")
bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)
tab_radar, tab_cart = st.tabs(["RADAR", "SCHEDINA"])

# --- TAB RADAR ---
with tab_radar:
    st.write("### 🔎 Scanner Prossime Partite")
    
    # PULSANTE UNICO
    if st.button("CERCA PROSSIME 10 PARTITE", type="primary", use_container_width=True):
        today = datetime.now().date()
        st.info(f"Scansiono i file della stagione 2024/25 a partire da oggi ({today})...")
        
        found_any = False
        
        for db in DATABASE:
            df_cal = get_data(db['fixture'])
            if df_cal is not None:
                col_date = next((c for c in df_cal.columns if 'Date' in c or 'Time' in c), None)
                if col_date:
                    df_cal['DT_CLEAN'] = pd.to_datetime(df_cal[col_date], dayfirst=True, errors='coerce').dt.date
                    
                    # FILTRO: Partite da OGGI in poi
                    matches = df_cal[df_cal['DT_CLEAN'] >= today].sort_values('DT_CLEAN').head(10)
                    
                    if not matches.empty:
                        df_h = get_data(db['history'])
                        if df_h is not None:
                            stats, ah, aa = process_stats(df_h)
                            if stats is not None:
                                teams_list = stats.index.tolist()
                                st.toast(f"{db['nome']}: OK! ({len(matches)} match)", icon="✅")
                                
                                for _, row in matches.iterrows():
                                    raw_h = row.get('Home Team', row.get('HomeTeam','')).strip()
                                    raw_a = row.get('Away Team', row.get('AwayTeam','')).strip()
                                    match_d = row['DT_CLEAN']
                                    
                                    real_h = smart_match_name(raw_h, teams_list)
                                    real_a = smart_match_name(raw_a, teams_list)
                                    
                                    res = analyze_math(real_h, real_a, stats, ah, aa)
                                    
                                    if res and res['Best']['Prob'] > 0.50:
                                        found_any = True
                                        best = res['Best']
                                        fav = res['Fav_1X2']
                                        
                                        with st.container(border=True):
                                            st.markdown(f"**{real_h} vs {real_a}** <small style='color:#888'>({match_d})</small>", unsafe_allow_html=True)
                                            k1, k2, k3 = st.columns(3)
                                            k1.metric("TOP", best['Tip'], f"{best['Prob']*100:.0f}%")
                                            k2.metric("1X2", fav['Label'], f"{fav['Prob']*100:.0f}%")
                                            k3.metric("Q", f"{best['Q']:.2f}")

        if not found_any:
            st.warning("Nessuna partita trovata. Verifica che i campionati siano attivi.")

# --- TAB SCHEDINA ---
with tab_cart:
    names = [d['nome'] for d in DATABASE]
    sel = st.selectbox("Scegli Campionato", names)
    
    if st.session_state['loaded_league'] != sel:
        with st.spinner("Caricamento dati..."):
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
        
        if st.button("ANALIZZA MATCH", type="primary", use_container_width=True):
            if h != a:
                res = analyze_math(h, a, st.session_state['cur_stats'], st.session_state['cur_ah'], st.session_state['cur_aa'])
                if res:
                    best = res['Best']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                    
                    st.divider()
                    st.markdown(f"### 📊 {h} vs {a}")
                    c_top, c_1x2, c_soldi = st.columns(3)
                    c_top.metric("STRATEGIA", best['Tip'], f"{best['Prob']*100:.1f}%")
                    c_1x2.metric("FAVORITO", res['Fav_1X2']['Label'], f"{res['Fav_1X2']['Prob']*100:.0f}%")
                    c_soldi.metric("STAKE", f"€{stake}", f"Q: {best['Q']:.2f}")
                    
                    st.caption("Probabilità Esatte:")
                    p = res['Probs']
                    st.progress(p['1'], f"1: {p['1']*100:.0f}%")
                    st.progress(p['X'], f"X: {p['X']*100:.0f}%")
                    st.progress(p['2'], f"2: {p['2']*100:.0f}%")
