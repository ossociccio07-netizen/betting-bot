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
st.set_page_config(page_title="BETTING PRO ULTIMATE", page_icon="⚽", layout="centered")

# CSS ORIGINALE (BELLO)
st.markdown("""
<style>
    .stApp { background-color: #000000; }
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* TAB STYLE */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #000; padding: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; background-color: #1a1a1a; border: 1px solid #333;
        border-radius: 8px; color: #888; font-weight: bold; font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d26a !important; color: #000 !important; border: none;
    }

    /* METRICHE */
    [data-testid="stMetricLabel"] {
        font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        font-size: 24px; font-weight: 900; color: #fff;
    }
    
    /* Colori Specifici Colonne */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] {
        color: #00ff00 !important; text-shadow: 0 0 10px rgba(0,255,0,0.4);
    }
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] {
        color: #00bfff !important;
    }
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] {
        color: #ffffff !important; background-color: #222; border-radius: 5px; padding: 0 5px;
    }

    /* CONTAINER */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #111; border: 1px solid #222; border-radius: 12px; padding: 15px;
    }
    
    /* BARRE PROBABILITA */
    .stProgress > div > div > div > div {
        background-color: #00bfff;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE & MAPPA NOMI (AGGIUNTO DIZIONARIO PER I NOMI)
# ==============================================================================
DATABASE = [
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
]

# Questo serve perché i file hanno nomi diversi per le stesse squadre
NAME_MAPPING = {
    "Man Utd": "Man United", "Manchester United": "Man United",
    "Man City": "Man City", "Manchester City": "Man City",
    "Nott'm Forest": "Nottm Forest", "Nottingham Forest": "Nottm Forest",
    "Sheffield Utd": "Sheffield United", "Luton": "Luton Town",
    "Wolves": "Wolverhampton", "Wolverhampton Wanderers": "Wolverhampton",
    "Brighton": "Brighton & Hove Albion", "Spurs": "Tottenham", 
    "Tottenham Hotspur": "Tottenham", "West Ham": "West Ham United",
    "Newcastle": "Newcastle United", "Inter": "Internazionale", "Milan": "AC Milan"
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
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

# FUNZIONE AGGIUNTA PER NORMALIZZARE I NOMI
def normalize_name(name, known_teams):
    name = str(name).strip()
    if name in NAME_MAPPING: return NAME_MAPPING[name]
    if name in known_teams: return name
    for t in known_teams:
        if name in t or t in name: return t
    return name

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        # PESI (60% Stagione / 40% Ultime 5)
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
        p_u25 = 1 - p_o25
        p_gg = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
        
        options = [
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph if ph>0 else 0},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa if pa>0 else 0},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd if pd>0 else 0},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25 if p_o25>0 else 0},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25 if p_u25>0 else 0},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg if p_gg>0 else 0}
        ]
        
        probs_1x2 = {"1": ph, "X": pd, "2": pa}
        fav_sign = max(probs_1x2, key=probs_1x2.get)
        fav_prob = probs_1x2[fav_sign]
        
        if fav_sign == "1": fav_label = "CASA"
        elif fav_sign == "2": fav_label = "OSPITE"
        else: fav_label = "PAREGGIO"

        # SOGLIA IMPOSTATA AL 50%
        valid = [o for o in options if o['Prob'] > 0.50]
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0}

        return {
            "c": h, "o": a, 
            "Best": best, 
            "Fav_1X2": {"Label": fav_label, "Prob": fav_prob, "Sign": fav_sign},
            "Probs": {"1": ph, "X": pd, "2": pa},
            "All": options,
            "xG_H": lh, "xG_A": la
        }
    except: return None

def calculate_stake(prob, quota, bankroll):
    try:
        if quota <= 1: return 0
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        stake = bankroll * (f * 0.20)
        return round(max(0, stake), 2)
    except: return 0

# ==============================================================================
# UI
# ==============================================================================
st.title("⚽ BETTING PRO ULTIMATE")

bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)

tab_radar, tab_cart = st.tabs(["RADAR AUTO", "SCHEDINA"])

# --- TAB RADAR (MODIFICA "PROSSIME PARTITE") ---
with tab_radar:
    st.write("### 🔎 Scanner Prossime Partite")
    st.caption("Analizza le prossime partite in programma (non solo oggi).")
    
    # Non chiediamo più la data esatta, partiamo da OGGI
    if st.button("CERCA PROSSIME OCCASIONI", type="primary", use_container_width=True):
        today = datetime.now().date()
        st.info(f"Cerco le partite a partire dal {today}...")
        
        global_found = False
        
        for db in DATABASE:
            # Scarica Calendario
            df_cal = get_data(db['fixture'])
            
            if df_cal is not None:
                # Cerca colonna data
                col_date = next((c for c in df_cal.columns if 'Date' in c or 'Time' in c), None)
                
                if col_date:
                    # Converti date
                    df_cal['DT_CLEAN'] = pd.to_datetime(df_cal[col_date], dayfirst=True, errors='coerce').dt.date
                    
                    # LOGICA NUOVA: Prendi le partite da OGGI in poi (massimo 15)
                    # Così se oggi è vuoto, ti fa vedere domani!
                    matches = df_cal[df_cal['DT_CLEAN'] >= today].sort_values('DT_CLEAN').head(15)
                    
                    if not matches.empty:
                        # Scarica Statistiche
                        df_h = get_data(db['history'])
                        if df_h is not None:
                            stats, ah, aa = process_stats(df_h)
                            if stats is not None:
                                teams_list = stats.index.tolist()
                                
                                # Messaggio di debug utile
                                first_match_date = matches.iloc[0]['DT_CLEAN']
                                st.toast(f"{db['nome']}: Trovate partite dal {first_match_date}", icon="📅")
                                
                                for _, row in matches.iterrows():
                                    raw_h = row.get('Home Team', row.get('HomeTeam','')).strip()
                                    raw_a = row.get('Away Team', row.get('AwayTeam','')).strip()
                                    match_date = row['DT_CLEAN']
                                    
                                    real_h = normalize_name(raw_h, teams_list)
                                    real_a = normalize_name(raw_a, teams_list)
                                    
                                    res = analyze_math(real_h, real_a, stats, ah, aa)
                                    
                                    # Filtro 50%
                                    if res and res['Best']['Prob'] > 0.50:
                                        global_found = True
                                        best = res['Best']
                                        fav = res['Fav_1X2']
                                        
                                        with st.container(border=True):
                                            # Intestazione con DATA della partita
                                            st.markdown(f"**{real_h} vs {real_a}** <span style='font-size:12px; color:#888'>({match_date})</span>", unsafe_allow_html=True)
                                            st.caption(f"{db['nome']}")
                                            
                                            k1, k2, k3 = st.columns(3)
                                            k1.metric("STRATEGIA", best['Tip'], f"{best['Prob']*100:.0f}%")
                                            k2.metric("FAVORITO", fav['Label'], f"{fav['Prob']*100:.0f}%")
                                            k3.metric("QUOTA", f"{best['Q']:.2f}")

        if not global_found:
            st.error("Nessuna partita trovata nei file. Possibili cause:")
            st.markdown("""
            1. I file online dei calendari non sono ancora aggiornati.
            2. Oggi non ci sono partite nei campionati selezionati.
            3. Il sito sorgente dei dati è momentaneamente irraggiungibile.
            """)

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

        if st.button("🚀 CALCOLA ANALISI COMPLETA", type="primary", use_container_width=True):
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    best = res['Best']
                    fav = res['Fav_1X2']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                    
                    with st.container(border=True):
                        st.markdown(f"#### {item['c']} <span style='color:#666'>vs</span> {item['o']}", unsafe_allow_html=True)
                        c_top, c_1x2, c_soldi = st.columns(3)
                        
                        c_top.metric("STRATEGIA", best['Tip'], f"{best['Prob']*100:.1f}%")
                        c_1x2.metric("FAVORITO 1X2", fav['Label'], f"{fav['Prob']*100:.1f}%")
                        c_soldi.metric("PUNTARE", f"€{stake}", f"Quota {best['Q']:.2f}")
                        
                        st.divider()
                        st.caption("Probabilità Esito Finale (1X2):")
                        p = res['Probs']
                        b1, b2, b3 = st.columns(3)
                        b1.progress(p['1'], f"1: {p['1']*100:.0f}%")
                        b2.progress(p['X'], f"X: {p['X']*100:.0f}%")
                        b3.progress(p['2'], f"2: {p['2']*100:.0f}%")
                        
                        st.markdown(f"<div style='text-align:center; font-size:11px; color:#666; margin-top:10px;'>xG Attesi: {res['xG_H']:.2f} - {res['xG_A']:.2f}</div>", unsafe_allow_html=True)

        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
