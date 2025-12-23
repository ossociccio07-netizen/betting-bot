import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE & STILE NEON
# ==============================================================================
DEFAULT_BUDGET = 100.0

st.set_page_config(page_title="BETTING PRO", page_icon="⚽", layout="centered")

st.markdown("""
<style>
    /* Sfondo App */
    .stApp { background-color: #000000; }
    
    /* Nascondi elementi inutili */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* STILE TAB (Bottoni in alto) */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #000; padding: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; background-color: #1a1a1a; border: 1px solid #333;
        border-radius: 8px; color: #888; font-weight: bold; font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff00 !important; color: #000 !important; border: none;
    }

    /* TRUCCO PER I METRIC (NUMERI GRANDI) */
    [data-testid="stMetricLabel"] {
        font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px; font-weight: 900; color: #fff;
    }
    /* Colora il valore del primo metric (Consiglio) in Verde Neon */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] {
        color: #00ff00 !important; text-shadow: 0 0 10px rgba(0,255,0,0.4);
    }
    /* Colora il valore del secondo metric (Soldi) in Bianco */
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    /* BORDI DEI CONTENITORI */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        background-color: #111; border: 1px solid #222; border-radius: 12px; padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# LOGICA (Invariata)
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
]

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

@st.cache_data(ttl=3600, show_spinner=False)
def get_data(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if r.status_code == 200: return pd.read_csv(io.StringIO(r.text))
    except: return None
    return None

def process_stats(df):
    try:
        df = df[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        sc = df.groupby('HomeTeam')[['FTHG','FTAG']].mean()
        st = df.groupby('AwayTeam')[['FTAG','FTHG']].mean()
        fc = df.groupby('HomeTeam')[['FTHG','FTAG']].apply(lambda x: x.tail(5).mean())
        ft = df.groupby('AwayTeam')[['FTAG','FTHG']].apply(lambda x: x.tail(5).mean())
        
        sc.columns, st.columns = ['H_GF_S','H_GS_S'], ['A_GF_S','A_GS_S']
        fc.columns, ft.columns = ['H_GF_F','H_GS_F'], ['A_GF_F','A_GS_F']
        
        tot = pd.concat([sc,st,fc,ft], axis=1)
        PS, PF = 0.60, 0.40
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
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg}
        ]
        
        valid = [o for o in options if o['Prob'] > (0.33 if "X" in o['Tip'] else 0.50)]
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0}

        return {"c": h, "o": a, "Best": best, "All": options, "xG_H": lh, "xG_A": la}
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
st.title("⚽ BETTING PRO")

bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)

tab_radar, tab_cart = st.tabs(["RADAR AUTO", "SCHEDINA"])

# --- TAB RADAR ---
with tab_radar:
    c1, c2 = st.columns(2)
    if c1.button("OGGI", use_container_width=True): t_scan = 0
    elif c2.button("DOMANI", use_container_width=True): t_scan = 1
    else: t_scan = None
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi {target_d}...")
        found = False
        
        for db in DATABASE:
            df_cal = get_data(db['fixture'])
            if df_cal is not None:
                cd = 'Date' if 'Date' in df_cal.columns else 'Match Date'
                df_cal[cd] = pd.to_datetime(df_cal[cd], errors='coerce')
                matches = df_cal[df_cal[cd].dt.strftime('%Y-%m-%d') == target_d]
                
                if not matches.empty:
                    df_h = get_data(db['history'])
                    if df_h is not None:
                        stats, ah, aa = process_stats(df_h)
                        if stats is not None:
                            for _, r in matches.iterrows():
                                c = r.get('Home Team', r.get('HomeTeam','')).strip()
                                o = r.get('Away Team', r.get('AwayTeam','')).strip()
                                m = {"Man Utd":"Man United", "Utd":"United"}
                                c, o = m.get(c,c), m.get(o,o)
                                
                                res = analyze_math(c, o, stats, ah, aa)
                                if res and res['Best']['Prob'] > 0.60:
                                    found = True
                                    best = res['Best']
                                    
                                    # BOX SICURO (USIAMO CONTAINER NATIVO)
                                    with st.container(border=True):
                                        st.markdown(f"**{c} vs {o}**")
                                        st.caption(db['nome'])
                                        k1, k2 = st.columns(2)
                                        k1.metric("CONSIGLIO", best['Tip'], f"{best['Prob']*100:.0f}%")
                                        k2.metric("QUOTA EST.", f"{best['Q']:.2f}")

        if not found: st.warning("Nessuna occasione sicura trovata.")

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

        if st.button("🚀 CALCOLA SCHEDINA", type="primary", use_container_width=True):
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    best = res['Best']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                    
                    # === QUI LA MAGIA ===
                    # Usiamo st.container e st.metric invece dell'HTML
                    # Questo previene al 100% la stampa del codice
                    with st.container(border=True):
                        st.markdown(f"#### {item['c']} <span style='color:#888'>vs</span> {item['o']}", unsafe_allow_html=True)
                        
                        col_tip, col_stake = st.columns(2)
                        
                        # Colonna Sinistra: Consiglio (Verde Neon grazie al CSS in alto)
                        col_tip.metric(
                            label="MIGLIOR SCELTA",
                            value=best['Tip'],
                            delta=f"Prob: {best['Prob']*100:.1f}%"
                        )
                        
                        # Colonna Destra: Soldi
                        col_stake.metric(
                            label="PUNTA",
                            value=f"€{stake}",
                            delta=f"Quota: {best['Q']:.2f}"
                        )
                        
                        st.markdown(f"<div style='font-size:12px; color:#666'>xG Casa: {res['xG_H']:.2f} | xG Ospite: {res['xG_A']:.2f}</div>", unsafe_allow_html=True)

        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
