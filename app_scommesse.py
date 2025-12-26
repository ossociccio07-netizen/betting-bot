import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE PAGINA
# ==============================================================================
DEFAULT_BUDGET = 100.0
st.set_page_config(page_title="BETTING PRO 2.0", page_icon="⚽", layout="centered")

# Stile CSS per nascondere menu e migliorare la grafica
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
    
    /* Metriche personalizzate */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] { color: #00ff00 !important; } /* Verde */
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] { color: #00bfff !important; } /* Blu */
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] { color: #ffffff !important; } /* Bianco */
    
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #111; border: 1px solid #222; border-radius: 12px; padding: 15px;
    }
    .stProgress > div > div > div > div { background-color: #00bfff; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE & MAPPA NOMI (Cruciale per far coincidere i dati)
# ==============================================================================
DATABASE = [
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
]

# Dizionario per correggere i nomi diversi tra Calendario e Statistiche
NAME_MAPPING = {
    "Man Utd": "Man United", "Manchester United": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Nott'm Forest": "Nottm Forest", "Nottingham Forest": "Nottm Forest",
    "Sheffield Utd": "Sheffield United",
    "Luton": "Luton Town",
    "Wolves": "Wolverhampton", "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton": "Brighton & Hove Albion",
    "Spurs": "Tottenham", "Tottenham Hotspur": "Tottenham",
    "West Ham": "West Ham United",
    "Newcastle": "Newcastle United",
    "Inter": "Internazionale", "Milan": "AC Milan",
    "Monza": "Monza"
}

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

# ==============================================================================
# FUNZIONI DATI
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200: 
            return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

def normalize_name(name, known_teams):
    """Cerca di far combaciare il nome del calendario con quello delle statistiche"""
    name = name.strip()
    # 1. Controllo dizionario manuale
    if name in NAME_MAPPING:
        return NAME_MAPPING[name]
    # 2. Controllo diretto
    if name in known_teams:
        return name
    # 3. Tentativo "contiene" (es. "Inter Milan" -> "Internazionale" no, ma "Luton" -> "Luton Town" si)
    for t in known_teams:
        if name in t or t in name:
            return t
    return name

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        # Pesi: 60% Stagione intera, 40% Ultime 5 partite
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
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg}
        ]
        
        # Logica Favorito
        probs_1x2 = {"1": ph, "X": pd, "2": pa}
        fav_sign = max(probs_1x2, key=probs_1x2.get)
        
        # Filtro Migliore Opzione
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
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        return round(max(0, bankroll * (f * 0.20)), 2)
    except: return 0

# ==============================================================================
# UI
# ==============================================================================
st.title("⚽ BETTING PRO 2.0")
bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)
tab_radar, tab_cart = st.tabs(["RADAR AUTO", "SCHEDINA"])

# --- TAB RADAR ---
with tab_radar:
    st.write("### 🔎 Scanner Partite")
    sel_date = st.date_input("Data Partite:", datetime.now())
    
    if st.button("SCANSIONA TUTTO", type="primary", use_container_width=True):
        target_str = sel_date.strftime('%Y-%m-%d')
        st.caption(f"Cerco partite del {target_str}...")
        
        global_found = False
        
        for db in DATABASE:
            # 1. Scarica Calendario
            df_cal = get_data(db['fixture'])
            if df_cal is not None:
                # Normalizza date
                col_date = 'Date' if 'Date' in df_cal.columns else 'Match Date'
                df_cal[col_date] = pd.to_datetime(df_cal[col_date], errors='coerce')
                
                # Filtra per data scelta
                day_matches = df_cal[df_cal[col_date].dt.strftime('%Y-%m-%d') == target_str]
                
                if not day_matches.empty:
                    # 2. Se ci sono partite, scarica le Statistiche (Storia)
                    df_h = get_data(db['history'])
                    if df_h is not None:
                        stats, ah, aa = process_stats(df_h)
                        if stats is not None:
                            teams_list = stats.index.tolist()
                            
                            st.toast(f"Trovate {len(day_matches)} partite in {db['nome']}...", icon="ℹ️")
                            
                            for _, row in day_matches.iterrows():
                                # Prendi i nomi grezzi dal calendario
                                raw_h = row.get('Home Team', row.get('HomeTeam','')).strip()
                                raw_a = row.get('Away Team', row.get('AwayTeam','')).strip()
                                
                                # 3. MAPPING INTELLIGENTE (Fondamentale!)
                                real_h = normalize_name(raw_h, teams_list)
                                real_a = normalize_name(raw_a, teams_list)
                                
                                # 4. Analisi
                                res = analyze_math(real_h, real_a, stats, ah, aa)
                                
                                # 5. Filtro (Basta il 50%)
                                if res and res['Best']['Prob'] > 0.50:
                                    global_found = True
                                    best = res['Best']
                                    with st.container(border=True):
                                        st.markdown(f"**{real_h} vs {real_a}**")
                                        st.caption(f"{db['nome']}")
                                        k1, k2, k3 = st.columns(3)
                                        k1.metric("TOP", best['Tip'], f"{best['Prob']*100:.0f}%")
                                        k2.metric("1X2", res['Fav_1X2']['Label'], f"{res['Fav_1X2']['Prob']*100:.0f}%")
                                        k3.metric("QUOTA", f"{best['Q']:.2f}")

        if not global_found:
            st.warning("Nessuna partita trovata. Controlla che ci siano partite OGGI in Premier/Serie A, oppure cambia data.")

# --- TAB CARRELLO ---
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
            if h != a: st.session_state['cart'].append({'c': h, 'o': a, 'stats': st.session_state['cur_stats'], 'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']})

    st.divider()
    
    if st.session_state['cart']:
        for i, item in enumerate(st.session_state['cart']):
            c1, c2 = st.columns([5,1])
            c1.text(f"{item['c']} vs {item['o']}")
            if c2.button("🗑️", key=f"del_{i}"):
                st.session_state['cart'].pop(i)
                st.rerun()

        if st.button("🚀 CALCOLA ANALISI", type="primary", use_container_width=True):
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    best = res['Best']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                    with st.container(border=True):
                        st.markdown(f"#### {item['c']} vs {item['o']}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("STRATEGIA", best['Tip'], f"{best['Prob']*100:.1f}%")
                        c2.metric("FAVORITO", res['Fav_1X2']['Label'], f"{res['Fav_1X2']['Prob']*100:.0f}%")
                        c3.metric("PUNTA", f"€{stake}", f"Q: {best['Q']:.2f}")
                        
                        st.caption("Probabilità 1X2:")
                        p = res['Probs']
                        b1, b2, b3 = st.columns(3)
                        b1.progress(p['1'], f"1: {p['1']*100:.0f}%")
                        b2.progress(p['X'], f"X: {p['X']*100:.0f}%")
                        b3.progress(p['2'], f"2: {p['2']*100:.0f}%")

        if st.button("Svuota tutto"):
            st.session_state['cart'] = []
            st.rerun()
