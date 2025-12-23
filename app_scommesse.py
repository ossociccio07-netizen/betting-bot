import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
DEFAULT_BUDGET = 100.0

st.set_page_config(page_title="BETTING PRO VISIBLE", page_icon="💎", layout="centered")

# ==============================================================================
# DESIGN "HIGH CONTRAST" (LEGGI BENISSIMO)
# ==============================================================================
st.markdown("""
<style>
    /* Nasconde elementi inutili */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* SFONDO GENERALE */
    .stApp {
        background-color: #000000; /* Nero assoluto per contrasto */
    }

    /* CARD PARTITA */
    .bet-card {
        background-color: #121212; /* Grigio molto scuro */
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* TITOLI */
    .match-header {
        color: #ffffff; /* BIANCO PURO */
        font-size: 20px;
        font-weight: bold;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    
    /* CONSIGLIO (IL RISULTATO) */
    .tip-box {
        background-color: #1a1a1a;
        border-left: 5px solid #00ffea; /* Ciano Neon */
        padding: 15px;
        margin-bottom: 10px;
    }
    .tip-label {
        color: #aaaaaa;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .tip-value {
        color: #00ffea; /* SCRITTA NEON */
        font-size: 28px;
        font-weight: 900;
        margin-top: 5px;
    }
    
    /* BOX SOLDI (PUNTATA) */
    .stake-box {
        background-color: #ffffff; /* SFONDO BIANCO */
        color: #000000; /* SCRITTA NERA */
        border-radius: 8px;
        padding: 10px;
        text-align: center;
        margin-top: 5px;
    }
    .stake-label { font-size: 10px; font-weight: bold; text-transform: uppercase; }
    .stake-value { font-size: 22px; font-weight: 900; }
    
    /* STATISTICHE */
    .stats-row {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
        color: #dddddd; /* Grigio chiaro leggibile */
        font-size: 13px;
        background: #1a1a1a;
        padding: 10px;
        border-radius: 6px;
    }
    
    /* BOTTONI TAB */
    .stTabs [data-baseweb="tab"] {
        color: #ffffff;
        background-color: #222;
        border: 1px solid #444;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffea !important;
        color: #000000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTORE MATEMATICO (Invariato perché perfetto)
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "I2", "nome": "🇮🇹 Serie B", "history": "https://www.football-data.co.uk/mmz4281/2526/I2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-b-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Champ", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
    {"id": "E2", "nome": "🇬🇧 L. One", "history": "https://www.football-data.co.uk/mmz4281/2526/E2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-league-one-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bund", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
    {"id": "P1", "nome": "🇵🇹 Port", "history": "https://www.football-data.co.uk/mmz4281/2526/P1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/portugal-primeira-liga-2025.csv"},
    {"id": "T1", "nome": "🇹🇷 Turc", "history": "https://www.football-data.co.uk/mmz4281/2526/T1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/turkey-super-lig-2025.csv"},
    {"id": "B1", "nome": "🇧🇪 Belgio", "history": "https://www.football-data.co.uk/mmz4281/2526/B1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/belgium-jupiler-pro-league-2025.csv"}
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
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph if ph>0 else 0},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa if pa>0 else 0},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd if pd>0 else 0},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25 if p_o25>0 else 0},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25 if p_u25>0 else 0},
            {"Tip": "GOAL (GG)", "Prob": p_gg, "Q": 1/p_gg if p_gg>0 else 0}
        ]
        
        valid = []
        for o in options:
            thr = 0.33 if "X" in o['Tip'] else (0.55 if "OVER" in o['Tip'] or "GOAL" in o['Tip'] else 0.52)
            if o['Prob'] > thr: valid.append(o)
            
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0}

        return {
            "c": h, "o": a, "Best": best, "All": options,
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
# UI VISIBILE
# ==============================================================================
st.title("💎 BETTING PRO v11")

# Box Budget (Stile pulito)
st.caption("Budget Totale (€)")
bankroll = st.number_input("", value=DEFAULT_BUDGET, step=10.0, label_visibility="collapsed")

tab_radar, tab_cart = st.tabs(["RADAR", "SCHEDINA"])

# --- RADAR ---
with tab_radar:
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI", use_container_width=True): t_scan = 0
    if c2.button("DOMANI", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi {target_d}...")
        
        found = False
        bar = st.progress(0)
        
        for i, db in enumerate(DATABASE):
            bar.progress((i+1)/len(DATABASE))
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
                                    # CARD COMPATTA E LEGGIBILE
                                    st.markdown(f"""
                                    <div class="bet-card" style="padding:15px; margin-bottom:10px;">
                                        <div style="color:white; font-weight:bold;">{c} vs {o}</div>
                                        <div style="font-size:12px; color:#aaa;">{db['nome']}</div>
                                        <hr style="border-color:#333; margin:8px 0;">
                                        <div style="display:flex; justify-content:space-between; align-items:center;">
                                            <div style="color:#00ffea; font-weight:bold; font-size:18px;">{best['Tip']}</div>
                                            <div style="color:white; font-weight:bold;">{best['Prob']*100:.0f}%</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
        bar.empty()
        if not found: st.warning("Nessuna occasione Top (>60%) trovata.")

# --- CARRELLO ---
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
            if h != a: st.session_state['cart'].append({'c': h, 'o': a, 'lega': sel, 'stats': st.session_state['cur_stats'], 'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']})

    st.divider()
    
    if st.session_state['cart']:
        st.subheader(f"Ticket ({len(st.session_state['cart'])})")
        for i, item in enumerate(st.session_state['cart']):
            c1, c2 = st.columns([5,1])
            c1.text(f"{item['c']} vs {item['o']}")
            if c2.button("🗑️", key=f"del_{i}"):
                st.session_state['cart'].pop(i)
                st.rerun()

        if st.button("🚀 ANALIZZA ORA", type="primary", use_container_width=True):
            
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                
                if res:
                    best = res['Best']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                    
                    # HTML CARD AD ALTO CONTRASTO
                    st.markdown(f"""
                    <div class="bet-card">
                        <div class="match-header">
                            {item['c']} <span style="color:#888; font-size:16px;">vs</span> {item['o']}
                        </div>
                        
                        <div style="display:flex; gap:10px;">
                            <div class="tip-box" style="flex:2;">
                                <div class="tip-label">MIGLIOR SCELTA</div>
                                <div class="tip-value">{best['Tip']}</div>
                                <div style="color:#ccc; font-size:12px;">Prob: <b>{best['Prob']*100:.1f}%</b></div>
                            </div>
                            
                            <div class="stake-box" style="flex:1;">
                                <div class="stake-label">PUNTA</div>
                                <div class="stake-value">€{stake}</div>
                            </div>
                        </div>
                        
                        <div class="stats-row">
                            <div>xG Casa: <b>{res['xG_H']:.2f}</b></div>
                            <div>xG Ospite: <b>{res['xG_A']:.2f}</b></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("📊 Tutti i dati (Clicca qui)"):
                        st.caption("Percentuali complete:")
                        for o in res['All']:
                            st.progress(o['Prob'], f"{o['Tip']}: {o['Prob']*100:.1f}%")

        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
