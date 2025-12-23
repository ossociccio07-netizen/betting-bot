import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE INIZIALE
# ==============================================================================
DEFAULT_BUDGET = 100.0

st.set_page_config(page_title="QUANTUM BETTING PRO", page_icon="💎", layout="centered")

# ==============================================================================
# CSS PER LOOK "APP DA 4000€" (Dark & Neon)
# ==============================================================================
st.markdown("""
<style>
    /* Pulizia Interfaccia */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    
    /* Sfondo e Testi */
    .stApp {
        background-color: #0e1117;
    }
    
    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #0e1117; padding: 10px 0; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; background-color: #1f1f1f; border-radius: 8px; 
        color: #888; flex: 1; border: 1px solid #333; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d26a !important; color: #000 !important; border: none; box-shadow: 0 0 10px rgba(0,210,106,0.4);
    }
    
    /* CARD PRINCIPALE */
    .bet-card {
        background: linear-gradient(145deg, #1a1a1a, #252525);
        border: 1px solid #333; border-radius: 15px; padding: 20px;
        margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        position: relative; overflow: hidden;
    }
    
    /* HEADER PARTITA */
    .match-title {
        font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 5px;
        display: flex; justify-content: space-between; align-items: center;
    }
    .league-tag {
        font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
        background-color: #333; padding: 3px 8px; border-radius: 4px; color: #ccc;
    }
    
    /* SEZIONE CONSIGLIO (VERDE/BLU/ARANCIO) */
    .tip-section {
        display: flex; justify-content: space-between; align-items: center;
        margin-top: 15px; padding: 10px; border-radius: 10px;
        background-color: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05);
    }
    .tip-label { font-size: 10px; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .tip-value { font-size: 24px; font-weight: 900; letter-spacing: -1px; }
    
    /* COLORI DINAMICI */
    .color-green { color: #00d26a; text-shadow: 0 0 10px rgba(0,210,106,0.3); }
    .color-blue { color: #00bfff; text-shadow: 0 0 10px rgba(0,191,255,0.3); }
    .color-orange { color: #ffaa00; text-shadow: 0 0 10px rgba(255,170,0,0.3); }
    
    /* BOX SOLDI (STAKE) */
    .stake-container {
        text-align: right;
    }
    .stake-val {
        font-size: 22px; font-weight: bold; color: #fff;
        background: #00d26a; padding: 5px 15px; border-radius: 8px;
        color: #000; display: inline-block;
    }
    
    /* STATS GRID */
    .stats-grid {
        display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 15px;
        padding-top: 15px; border-top: 1px solid #333;
    }
    .stat-item { text-align: center; }
    .stat-num { font-size: 14px; font-weight: bold; color: #ddd; }
    .stat-desc { font-size: 10px; color: #666; }

</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MOTORE DATI (Database Completo)
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
    {"id": "N1", "nome": "🇳🇱 Erediv", "history": "https://www.football-data.co.uk/mmz4281/2526/N1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/netherlands-eredivisie-2025.csv"},
    {"id": "P1", "nome": "🇵🇹 Port", "history": "https://www.football-data.co.uk/mmz4281/2526/P1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/portugal-primeira-liga-2025.csv"},
    {"id": "T1", "nome": "🇹🇷 Turc", "history": "https://www.football-data.co.uk/mmz4281/2526/T1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/turkey-super-lig-2025.csv"},
    {"id": "B1", "nome": "🇧🇪 Belgio", "history": "https://www.football-data.co.uk/mmz4281/2526/B1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/belgium-jupiler-pro-league-2025.csv"}
]

if 'cart' not in st.session_state: st.session_state['cart'] = []
if 'loaded_league' not in st.session_state: st.session_state['loaded_league'] = None

# ==============================================================================
# ALGORITMI MATEMATICI (CORE)
# ==============================================================================
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
        
        # Pesiamo: 60% Stagione intera, 40% Ultime 5 partite (Forma)
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
        
        # Poisson Lambda
        lh = stats.at[h,'Att_H'] * stats.at[a,'Dif_A'] * ah
        la = stats.at[a,'Att_A'] * stats.at[h,'Dif_H'] * aa
        
        # Probabilità 1X2
        ph, pd, pa = 0, 0, 0
        for i in range(6):
            for j in range(6):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i>j: ph+=p
                elif i==j: pd+=p
                else: pa+=p
        
        # Probabilità Gol
        p_o25 = 1 - (poisson.pmf(0, lh+la) + poisson.pmf(1, lh+la) + poisson.pmf(2, lh+la))
        p_u25 = 1 - p_o25
        p_gg = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))
        
        options = [
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph if ph>0 else 0, "Color": "color-blue"},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa if pa>0 else 0, "Color": "color-blue"},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd if pd>0 else 0, "Color": "color-orange"},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25 if p_o25>0 else 0, "Color": "color-green"},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25 if p_u25>0 else 0, "Color": "color-orange"},
            {"Tip": "GOAL (GG)", "Prob": p_gg, "Q": 1/p_gg if p_gg>0 else 0, "Color": "color-green"}
        ]
        
        # Logica di Selezione Migliore
        # Preferiamo Over/Goal se > 55%, Esiti finali se > 50%, X se > 33%
        valid = []
        for o in options:
            thr = 0.33 if "X" in o['Tip'] else (0.55 if "OVER" in o['Tip'] or "GOAL" in o['Tip'] else 0.50)
            if o['Prob'] > thr: valid.append(o)
            
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0, "Color": "gray"}

        return {
            "c": h, "o": a, "Best": best, "All": options,
            "xG_H": lh, "xG_A": la,
            "Force_H": stats.at[h,'Att_H'], "Force_A": stats.at[a,'Att_A']
        }
    except: return None

def calculate_stake(prob, quota, bankroll):
    """Calcolo Money Management Professionale (Kelly / 5)"""
    try:
        if quota <= 1: return 0
        f = ((quota - 1) * prob - (1 - prob)) / (quota - 1)
        stake = bankroll * (f * 0.20) # 20% del Kelly per sicurezza massima
        return round(max(0, stake), 2)
    except: return 0

# ==============================================================================
# INTERFACCIA UTENTE (UI)
# ==============================================================================
st.title("💎 QUANTUM BETTING PRO")

# BANKROLL HEADER
with st.container():
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 🏦 Bankroll Management")
        st.caption("Inserisci il tuo budget totale. L'algoritmo calcolerà l'investimento ottimale per ogni singola giocata.")
    with c2:
        bankroll = st.number_input("Budget (€)", value=DEFAULT_BUDGET, step=10.0, format="%.2f")

# TABS
st.write("") # Spacer
tab_radar, tab_cart = st.tabs(["📡 RADAR AUTOMATICO", "🛠️ SCHEDINA MANUALE"])

# --- TAB 1: RADAR (Scanner Veloce) ---
with tab_radar:
    c_today, c_tom = st.columns(2)
    t_scan = None
    if c_today.button("SCANSIONA OGGI 📅", use_container_width=True): t_scan = 0
    if c_tom.button("SCANSIONA DOMANI 📆", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi dei mercati per il {target_d}...")
        
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
                                # Mostra solo High Confidence (>60%)
                                if res and res['Best']['Prob'] > 0.60:
                                    found = True
                                    best = res['Best']
                                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)
                                    
                                    # CARD MINIMAL PER IL RADAR
                                    st.markdown(f"""
                                    <div class="bet-card" style="padding: 10px; margin-bottom: 10px;">
                                        <div style="display:flex; justify-content:space-between;">
                                            <div style="font-weight:bold; color:white;">{c} vs {o}</div>
                                            <div class="league-tag">{db['nome']}</div>
                                        </div>
                                        <div style="display:flex; justify-content:space-between; margin-top:5px;">
                                            <div style="color:{'#00d26a' if 'OVER' in best['Tip'] else '#00bfff'}; font-weight:bold;">{best['Tip']}</div>
                                            <div style="color:#aaa;">{best['Prob']*100:.0f}%</div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
        bar.empty()
        if not found: st.warning("Nessuna occasione 'High Value' trovata per questa data.")


# --- TAB 2: CARRELLO (Analisi Profonda) ---
with tab_cart:
    names = [d['nome'] for d in DATABASE]
    sel = st.selectbox("Seleziona Campionato", names)
    
    # Auto-Load
    if st.session_state['loaded_league'] != sel:
        with st.spinner(f"Caricamento dati {sel}..."):
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
        
        if st.button("➕ AGGIUNGI ALLA LISTA", use_container_width=True):
            if h != a:
                st.session_state['cart'].append({
                    'c': h, 'o': a, 'lega': sel,
                    'stats': st.session_state['cur_stats'],
                    'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']
                })
                st.toast("Match aggiunto", icon="✅")

    st.divider()
    
    if st.session_state['cart']:
        st.markdown(f"### 🛒 Schedina ({len(st.session_state['cart'])})")
        
        # Gestione Lista
        for i, item in enumerate(st.session_state['cart']):
            c_txt, c_del = st.columns([5,1])
            c_txt.text(f"{item['c']} vs {item['o']}")
            if c_del.button("🗑️", key=f"del_{i}"):
                st.session_state['cart'].pop(i)
                st.rerun()

        # BOTTONE ANALISI
        if st.button("🚀 ELABORA STRATEGIA VINCENTE", type="primary", use_container_width=True):
            
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                
                if res:
                    best = res['Best']
                    # Quota stimata dal bot + 5% spread
                    est_quota = best['Q'] * 1.05
                    stake = calculate_stake(best['Prob'], est_quota, bankroll)
                    
                    # --- HTML CARD ---
                    st.markdown(f"""
                    <div class="bet-card">
                        <div class="match-title">
                            {item['c']} <span style="color:#666; font-size:14px;">vs</span> {item['o']}
                            <span class="league-tag">{item['lega']}</span>
                        </div>
                        
                        <div class="tip-section">
                            <div>
                                <div class="tip-label">MIGLIOR CONSIGLIO</div>
                                <div class="tip-value {best['Color']}">{best['Tip']}</div>
                                <div style="font-size:11px; color:#888;">Sicurezza: <b>{best['Prob']*100:.1f}%</b></div>
                            </div>
                            <div class="stake-container">
                                <div class="tip-label">INVESTIMENTO</div>
                                <div class="stake-val">€{stake}</div>
                            </div>
                        </div>
                        
                        <div class="stats-grid">
                            <div class="stat-item">
                                <div class="stat-num">{res['xG_H']:.2f}</div>
                                <div class="stat-desc">Expected Goals Casa</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-num">{res['xG_A']:.2f}</div>
                                <div class="stat-desc">Expected Goals Ospite</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-num">{res['Force_H']:.2f}</div>
                                <div class="stat-desc">Potenza Attacco Home</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-num">{res['Force_A']:.2f}</div>
                                <div class="stat-desc">Debolezza Difesa Away</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # EXPANDER CON TUTTI I DATI
                    with st.expander("📊 Analisi Completa Mercati"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Esito Finale")
                            for o in res['All'][:3]: # 1, 2, X
                                st.progress(o['Prob'], f"{o['Tip']}: {o['Prob']*100:.1f}%")
                        with c2:
                            st.caption("Gol")
                            for o in res['All'][3:]: # Over, Under, Goal
                                st.progress(o['Prob'], f"{o['Tip']}: {o['Prob']*100:.1f}%")

        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
