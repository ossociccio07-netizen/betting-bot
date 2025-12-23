import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE PRO (SOLO PER I TUOI OCCHI)
# ==============================================================================
st.set_page_config(page_title="Sniper Bet AI", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;} 
    
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #111; border-radius: 8px; color: #888; flex: 1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00e676 !important; color: black !important; font-weight: bold;
    }
    .metric-box {
        background-color: #222; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATI & LOGICA
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Champ", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
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
        PS, PF = 0.60, 0.40 # Aumentato peso forma recente per precisione
        
        tot['Att_H'] = ((tot['H_GF_S']*PS + tot['H_GF_F']*PF) / avg_h)
        tot['Dif_H'] = ((tot['H_GS_S']*PS + tot['H_GS_F']*PF) / avg_a)
        tot['Att_A'] = ((tot['A_GF_S']*PS + tot['A_GF_F']*PF) / avg_a)
        tot['Dif_A'] = ((tot['A_GS_S']*PS + tot['A_GS_F']*PF) / avg_h)
        return tot, avg_h, avg_a
    except: return None, None, None

def calculate_stake(prob, quota, bankroll):
    """Calcola la puntata usando il Criterio di Kelly Frazionario (più sicuro)"""
    try:
        if quota <= 1: return 0
        b = quota - 1
        p = prob
        q = 1 - p
        # Formula di Kelly: (bp - q) / b
        f = (b * p - q) / b
        # Usiamo Kelly/4 (molto conservativo) per non bruciare la cassa
        safe_stake_pct = (f * 0.25) 
        if safe_stake_pct < 0: return 0
        stake = bankroll * safe_stake_pct
        return round(stake, 2)
    except: return 0

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
        prob_win = 0.0
        marg = 1.05
        
        if ph > 0.50: 
            qmin, cons, color = (1/ph)*marg, "PUNTA 1", "green"
            prob_win = ph
        elif pa > 0.50: 
            qmin, cons, color = (1/pa)*marg, "PUNTA 2", "blue"
            prob_win = pa
        elif pd > 0.35: 
            qmin, cons, color = (1/pd)*marg, "RISCHIO X", "orange"
            prob_win = pd
        
        return {
            "Match": f"{h} - {a}", "p1": ph, "px": pd, "p2": pa, 
            "Tip": cons, "Quota": qmin, "c": h, "o": a, "ProbWin": prob_win
        }
    except: return None

# ==============================================================================
# UI: SNIPER MODE
# ==============================================================================
st.title("🎯 Sniper Bet AI")

# CASSA (INPUT UTENTE)
with st.expander("💰 Gestione Bankroll (Clicca per settare)", expanded=True):
    bankroll = st.number_input("Il tuo Budget Attuale (€):", value=100.0, step=10.0)
    st.caption("Il bot calcolerà quanto puntare per massimizzare il profitto e minimizzare il rischio.")

tab_auto, tab_manual = st.tabs(["RADAR AUTO", "SCHEDINA"])

# --- TAB 1: AUTO ---
with tab_auto:
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI", use_container_width=True, type="primary"): t_scan = 0
    if c2.button("DOMANI", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi del {target_d}...")
        
        results = []
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
                                res = analyze(c, o, stats, ah, aa)
                                if res and res['Tip'] != 'NO BET':
                                    res['Lega'] = db['nome']
                                    # Calcolo Stake Consigliato
                                    res['Stake'] = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                                    results.append(res)
        bar.empty()
        
        if results:
            # Ordina per sicurezza (Probabilità più alta prima)
            results.sort(key=lambda x: x['ProbWin'], reverse=True)
            
            st.success(f"Trovate {len(results)} occasioni.")
            for res in results:
                with st.container(border=True):
                    head, stake_box = st.columns([3, 1])
                    with head:
                        st.markdown(f"**{res['Match']}**")
                        st.caption(f"{res['Lega']} | {res['Tip']}")
                    with stake_box:
                        if res['Stake'] > 0:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div style="font-size:12px">PUNTA</div>
                                <div style="font-size:18px; color:#00e676; font-weight:bold">€{res['Stake']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.caption("No Valore")
                    
                    st.divider()
                    
                    # Dettagli Quota
                    c_q, c_p = st.columns([1, 3])
                    c_q.metric("Quota Min", f"{res['Quota']:.2f}")
                    if "1" in res['Tip']: c_p.progress(res['p1'], f"Prob 1: {res['p1']*100:.0f}%")
                    elif "2" in res['Tip']: c_p.progress(res['p2'], f"Prob 2: {res['p2']*100:.0f}%")
                    else: c_p.progress(res['px'], f"Prob X: {res['px']*100:.0f}%")
        else:
            st.warning("Nessuna occasione trovata.")

# --- TAB 2: MANUALE ---
with tab_manual:
    names = [d['nome'] for d in DATABASE]
    sel_name = st.selectbox("Campionato", names)
    
    if st.session_state['loaded_league'] != sel_name:
        with st.spinner("Loading..."):
            sel_db = next(d for d in DATABASE if d['nome'] == sel_name)
            df = get_data(sel_db['history'])
            if df is not None:
                stats, ah, aa = process_stats(df)
                st.session_state['cur_stats'] = stats
                st.session_state['cur_ah'] = ah
                st.session_state['cur_aa'] = aa
                st.session_state['cur_teams'] = sorted(stats.index.tolist()) if stats is not None else []
                st.session_state['loaded_league'] = sel_name

    if 'cur_teams' in st.session_state and st.session_state['cur_teams']:
        c1, c2 = st.columns(2)
        h = c1.selectbox("Casa", st.session_state['cur_teams'])
        a = c2.selectbox("Ospite", st.session_state['cur_teams'], index=1)
        
        if st.button("➕ Aggiungi", use_container_width=True):
            if h != a:
                st.session_state['cart'].append({
                    'c': h, 'o': a, 'lega': sel_name,
                    'stats': st.session_state['cur_stats'],
                    'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']
                })

    st.divider()
    
    if st.session_state['cart']:
        st.subheader(f"Portafoglio ({len(st.session_state['cart'])})")
        
        for i, item in enumerate(st.session_state['cart']):
            with st.container(border=True):
                ct, cd = st.columns([5,1])
                ct.text(f"{item['c']} vs {item['o']}")
                if cd.button("❌", key=f"del_{i}"):
                    st.session_state['cart'].pop(i)
                    st.rerun()

        if st.button("🚀 CALCOLA INVESTIMENTO", type="primary", use_container_width=True):
             for item in st.session_state['cart']:
                res = analyze(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    res['Stake'] = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                    with st.container(border=True):
                        st.markdown(f"### {res['c']} vs {res['o']}")
                        
                        m1, m2 = st.columns([2, 1])
                        with m1:
                            if "PUNTA" in res['Tip']: st.success(f"**{res['Tip']}** @ {res['Quota']:.2f}")
                            elif "RISCHIO" in res['Tip']: st.warning(f"**{res['Tip']}** @ {res['Quota']:.2f}")
                            else: st.error("NO BET")
                        
                        with m2:
                            st.markdown(f"""
                            <div class="metric-box">
                                <div style="font-size:10px">PUNTA</div>
                                <div style="color:#00e676; font-weight:bold">€{res['Stake']}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
