import streamlit as st
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
st.set_page_config(page_title="BETTING PRO 2025-26", page_icon="⚽", layout="centered")
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
# DATABASE AGGIORNATO ALLA STAGIONE 2025/2026
# ==============================================================================
DATABASE = [
    {
        "id": "E0", 
        "nome": "🇬🇧 Premier League", 
        # History: 2526 = Stagione 25/26
        "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", 
        # Fixture: 2026 = File che finisce nel 2026 (quindi contiene Dic 2025)
        "fixture": "https://fixturedownload.com/download/csv/2026/england-premier-league-2026.csv"
    },
    {
        "id": "I1", 
        "nome": "🇮🇹 Serie A", 
        "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2026/italy-serie-a-2026.csv"
    },
    {
        "id": "SP1", 
        "nome": "🇪🇸 La Liga", 
        "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2026/spain-la-liga-2026.csv"
    },
    {
        "id": "D1", 
        "nome": "🇩🇪 Bundesliga", 
        "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2026/germany-bundesliga-2026.csv"
    },
    {
        "id": "F1", 
        "nome": "🇫🇷 Ligue 1", 
        "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", 
        "fixture": "https://fixturedownload.com/download/csv/2026/france-ligue-1-2026.csv"
    }
]

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

# ==============================================================================
# FUNZIONI DATI & LOGICA
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

def smart_match_name(name, known_teams):
    # Cerca il nome più simile (es. Man City -> Manchester City)
    matches = difflib.get_close_matches(name, known_teams, n=1, cutoff=0.5)
    return matches[0] if matches else name

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        # Pesi
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
st.title("⚽ BETTING PRO 2025/26")
bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)
tab_radar, tab_cart = st.tabs(["RADAR", "SCHEDINA"])

# --- TAB RADAR ---
with tab_radar:
    st.write("### 🔎 Scanner Prossime Partite")
    
    # Bottone unico che cerca da OGGI in poi
    if st.button("CERCA PARTITE (DA OGGI)", type="primary", use_container_width=True):
        today = datetime.now().date()
        st.info(f"Cerco partite nel database 2025/2026 a partire dal {today}...")
        
        found_any = False
        
        for db in DATABASE:
            df_cal = get_data(db['fixture'])
            if df_cal is not None:
                # Cerca colonna data (Fixtures usa 'Date' o 'Match Date')
                col_date = next((c for c in df_cal.columns if 'Date' in c or 'Time' in c), None)
                
                if col_date:
                    # Normalizza data
                    df_cal['DT_CLEAN'] = pd.to_datetime(df_cal[col_date], dayfirst=True, errors='coerce').dt.date
                    
                    # FILTRO: Prendi le partite da OGGI in poi (max 10 per non intasare)
                    matches = df_cal[df_cal['DT_CLEAN'] >= today].sort_values('DT_CLEAN').head(10)
                    
                    if not matches.empty:
                        # Scarica Statistiche (Storico)
                        df_h = get_data(db['history'])
                        if df_h is not None:
                            stats, ah, aa = process_stats(df_h)
                            if stats is not None:
                                teams_list = stats.index.tolist()
                                st.success(f"{db['nome']}: Analisi {len(matches)} match...")
                                
                                for _, row in matches.iterrows():
                                    raw_h = row.get('Home Team', row.get('HomeTeam','')).strip()
                                    raw_a = row.get('Away Team', row.get('AwayTeam','')).strip()
                                    match_d = row['DT_CLEAN']
                                    
                                    # Correzione Nomi Automatica
                                    real_h = smart_match_name(raw_h, teams_list)
                                    real_a = smart_match_name(raw_a, teams_list)
                                    
                                    # Analisi
                                    res = analyze_math(real_h, real_a, stats, ah, aa)
                                    
                                    # Filtro 50%
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
            st.warning("Nessuna partita trovata. Verifica che i campionati non siano in pausa invernale.")

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
