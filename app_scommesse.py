import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# 1. CONFIGURAZIONE PREMIUM (CSS & LAYOUT)
# ==============================================================================
st.set_page_config(page_title="AI Betting Pro", page_icon="💎", layout="wide")

# CSS PERSONALIZZATO PER LOOK "APP DA 2000€"
st.markdown("""
<style>
    /* Nasconde menu standard Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Stile Card Partita */
    .match-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Stile Metriche */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #00ff00;
    }
    
    /* Pulsanti Primary più fighi */
    .stButton>button {
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. MOTORE DATI (BACKEND)
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "I2", "nome": "🇮🇹 Serie B", "history": "https://www.football-data.co.uk/mmz4281/2526/I2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-b-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier League", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Championship", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
    {"id": "E2", "nome": "🇬🇧 League One", "history": "https://www.football-data.co.uk/mmz4281/2526/E2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-league-one-2025.csv"},
    {"id": "E3", "nome": "🇬🇧 League Two", "history": "https://www.football-data.co.uk/mmz4281/2526/E3.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-league-two-2025.csv"},
    {"id": "SP1", "nome": "🇪🇸 La Liga", "history": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/spain-la-liga-2025.csv"},
    {"id": "D1", "nome": "🇩🇪 Bundesliga", "history": "https://www.football-data.co.uk/mmz4281/2526/D1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/germany-bundesliga-2025.csv"},
    {"id": "F1", "nome": "🇫🇷 Ligue 1", "history": "https://www.football-data.co.uk/mmz4281/2526/F1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/france-ligue-1-2025.csv"},
    {"id": "N1", "nome": "🇳🇱 Olanda", "history": "https://www.football-data.co.uk/mmz4281/2526/N1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/netherlands-eredivisie-2025.csv"},
    {"id": "P1", "nome": "🇵🇹 Portogallo", "history": "https://www.football-data.co.uk/mmz4281/2526/P1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/portugal-primeira-liga-2025.csv"},
    {"id": "T1", "nome": "🇹🇷 Turchia", "history": "https://www.football-data.co.uk/mmz4281/2526/T1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/turkey-super-lig-2025.csv"},
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
        
        # Medie Lega
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        # Stagione
        sc = df.groupby('HomeTeam')[['FTHG','FTAG']].mean()
        st = df.groupby('AwayTeam')[['FTAG','FTHG']].mean()
        # Forma (last 5)
        fc = df.groupby('HomeTeam')[['FTHG','FTAG']].apply(lambda x: x.tail(5).mean())
        ft = df.groupby('AwayTeam')[['FTAG','FTHG']].apply(lambda x: x.tail(5).mean())
        
        # Merge
        # Fix colonne
        sc.columns, st.columns = ['H_GF_S','H_GS_S'], ['A_GF_S','A_GS_S']
        fc.columns, ft.columns = ['H_GF_F','H_GS_F'], ['A_GF_F','A_GS_F']
        
        tot = pd.concat([sc,st,fc,ft], axis=1)
        PS, PF = 0.70, 0.30 # Pesi
        
        tot['Att_H'] = ((tot['H_GF_S']*PS + tot['H_GF_F']*PF) / avg_h)
        tot['Dif_H'] = ((tot['H_GS_S']*PS + tot['H_GS_F']*PF) / avg_a)
        tot['Att_A'] = ((tot['A_GF_S']*PS + tot['A_GF_F']*PF) / avg_a)
        tot['Dif_A'] = ((tot['A_GS_S']*PS + tot['A_GS_F']*PF) / avg_h)
        
        return tot, avg_h, avg_a
    except: return None, None, None

def analyze(h, a, stats, ah, aa):
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
        
        cons, qmin, color = "NO BET", 0.0, "grey"
        marg = 1.05
        
        if ph > 0.50: qmin, cons, color = (1/ph)*marg, "PUNTA 1", "green"
        elif pa > 0.50: qmin, cons, color = (1/pa)*marg, "PUNTA 2", "blue"
        elif pd > 0.33: qmin, cons, color = (1/pd)*marg, "RISCHIO X", "orange"
        
        return {
            "Match": f"{h} - {a}", "p1": ph, "px": pd, "p2": pa,
            "Tip": cons, "Quota": qmin, "Color": color, "c": h, "o": a
        }
    except: return None

# ==============================================================================
# 3. INTERFACCIA UTENTE (FRONTEND)
# ==============================================================================

# --- SIDEBAR (NAVIGAZIONE) ---
with st.sidebar:
    st.title("💎 AI Betting Pro")
    st.markdown("---")
    mode = st.radio("MODALITÀ", ["⚡️ Auto-Scan (Radar)", "🛠️ Manual Builder"], index=1)
    st.markdown("---")
    st.caption("v3.0 - Ultimate Edition")
    st.caption("Powered by Poisson & xG Models")

# --- MODALITÀ AUTOMATICA ---
if mode == "⚡️ Auto-Scan (Radar)":
    st.title("📡 Radar Partite")
    st.markdown("Scansione satellitare di tutti i campionati per trovare valore.")
    
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("PARTITE OGGI", use_container_width=True, type="primary"): t_scan = 0
    if c2.button("PARTITE DOMANI", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.toast(f"Inizio scansione per il {target_d}...", icon="🛰️")
        
        results = []
        progress = st.progress(0)
        status_text = st.empty()
        
        for i, db in enumerate(DATABASE):
            status_text.text(f"Analisi: {db['nome']}...")
            progress.progress((i+1)/len(DATABASE))
            
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
                                m = {"Man Utd":"Man United", "Utd":"United", "Nottm Forest":"Nott'm Forest"}
                                c, o = m.get(c,c), m.get(o,o)
                                
                                res = analyze(c, o, stats, ah, aa)
                                if res and res['Tip'] != 'NO BET':
                                    res['Lega'] = db['nome']
                                    results.append(res)
        
        progress.empty()
        status_text.empty()
        
        if results:
            st.success(f"Trovate {len(results)} Value Bets!")
            for res in results:
                with st.container(border=True):
                    cols = st.columns([4, 2, 2])
                    cols[0].markdown(f"**{res['Match']}**")
                    cols[0].caption(res['Lega'])
                    cols[1].metric("Consiglio", res['Tip'], delta=f"Min: {res['Quota']:.2f}")
                    cols[2].progress(res['p1'], f"1: {res['p1']*100:.0f}%")
                    cols[2].progress(res['px'], f"X: {res['px']*100:.0f}%")
                    cols[2].progress(res['p2'], f"2: {res['p2']*100:.0f}%")
        else:
            st.info("Nessuna occasione trovata. I Bookmakers sono allineati o non ci sono partite.")

# --- MODALITÀ MANUALE ---
elif mode == "🛠️ Manual Builder":
    st.title("🏗️ Schedina Builder")
    
    col_main, col_cart = st.columns([2, 1])
    
    with col_main:
        # SELEZIONE AUTOMATICA (Nessun bottone carica!)
        names = [d['nome'] for d in DATABASE]
        sel_name = st.selectbox("Seleziona Campionato", names)
        
        # LOGICA CARICAMENTO SILENZIOSO
        if st.session_state['loaded_league'] != sel_name:
            with st.spinner(f"Calibrazione algoritmi per {sel_name}..."):
                sel_db = next(d for d in DATABASE if d['nome'] == sel_name)
                df = get_data(sel_db['history'])
                if df is not None:
                    stats, ah, aa = process_stats(df)
                    st.session_state['cur_stats'] = stats
                    st.session_state['cur_ah'] = ah
                    st.session_state['cur_aa'] = aa
                    st.session_state['cur_teams'] = sorted(stats.index.tolist()) if stats is not None else []
                    st.session_state['loaded_league'] = sel_name
                else:
                    st.error("Errore connessione database.")
        
        # INPUT SQUADRE
        if 'cur_teams' in st.session_state and st.session_state['cur_teams']:
            c1, c2 = st.columns(2)
            h = c1.selectbox("Casa", st.session_state['cur_teams'])
            a = c2.selectbox("Ospite", st.session_state['cur_teams'], index=1)
            
            if st.button("➕ AGGIUNGI AL TICKET", use_container_width=True):
                if h != a:
                    st.session_state['cart'].append({
                        'c': h, 'o': a, 'lega': sel_name,
                        'stats': st.session_state['cur_stats'],
                        'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']
                    })
                    st.toast(f"Aggiunta: {h} vs {a}", icon="✅")
                else:
                    st.toast("Seleziona due squadre diverse!", icon="⚠️")

    # --- CARRELLO LATERALE ---
    with col_cart:
        st.subheader(f"🎫 Ticket ({len(st.session_state['cart'])})")
        
        if st.session_state['cart']:
            for i, item in enumerate(st.session_state['cart']):
                with st.container(border=True):
                    c_txt, c_del = st.columns([5, 1])
                    c_txt.text(f"{item['c']}\nvs {item['o']}")
                    if c_del.button("🗑️", key=f"del_{i}"):
                        st.session_state['cart'].pop(i)
                        st.rerun()
            
            st.divider()
            
            if st.button("🚀 ELABORA PRONOSTICI", type="primary", use_container_width=True):
                st.session_state['analysis_done'] = True
            
            if st.button("Svuota tutto", use_container_width=True):
                st.session_state['cart'] = []
                st.session_state['analysis_done'] = False
                st.rerun()
        else:
            st.info("Il ticket è vuoto.")

    # --- RISULTATI ANALISI (SOTTO) ---
    if st.session_state.get('analysis_done') and st.session_state['cart']:
        st.divider()
        st.subheader("📊 Report Analisi")
        
        for item in st.session_state['cart']:
            res = analyze(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
            if res:
                # CARD PRO DESIGN
                with st.container(border=True):
                    head_col, score_col = st.columns([3, 1])
                    
                    with head_col:
                        st.markdown(f"### {res['c']} vs {res['o']}")
                        st.caption(f"📍 {item['lega']}")
                    
                    with score_col:
                        if "PUNTA" in res['Tip']: st.success(f"**{res['Tip']}**")
                        elif "RISCHIO" in res['Tip']: st.warning(f"**{res['Tip']}**")
                        else: st.error(f"**{res['Tip']}**")
                    
                    st.divider()
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Quota Min", f"{res['Quota']:.2f}")
                    m2.metric("Prob 1", f"{res['p1']*100:.0f}%")
                    m3.metric("Prob X", f"{res['px']*100:.0f}%")
                    m4.metric("Prob 2", f"{res['p2']*100:.0f}%")
                    
                    # Barre visive
                    st.progress(res['p1'], "Forza Casa")
                    st.progress(res['p2'], "Forza Ospite")