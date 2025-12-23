import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE GRAFICA "TOTAL FOOTBALL"
# ==============================================================================
st.set_page_config(page_title="AI Betting Pro 4.0", page_icon="🔥", layout="centered")

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;} 
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; background-color: #1a1a1a; border-radius: 8px; 
        color: #bbb; flex: 1; border: 1px solid #333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00d26a !important; color: black !important; font-weight: bold; border: none;
    }
    
    .stake-box {
        background-color: #000; border: 1px solid #00d26a; border-radius: 8px;
        padding: 8px; text-align: center;
    }
    .stake-title { font-size: 9px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .stake-value { font-size: 18px; font-weight: bold; color: #00d26a; }
    
    .metric-row { display: flex; justify-content: space-between; margin-bottom: 5px; }
    .metric-label { font-size: 12px; color: #aaa; }
    .metric-val { font-size: 14px; font-weight: bold; color: white; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATI & MOTORE
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
    {"id": "I2", "nome": "🇮🇹 Serie B", "history": "https://www.football-data.co.uk/mmz4281/2526/I2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-b-2025.csv"},
    {"id": "E0", "nome": "🇬🇧 Premier", "history": "https://www.football-data.co.uk/mmz4281/2526/E0.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"},
    {"id": "E1", "nome": "🇬🇧 Champ", "history": "https://www.football-data.co.uk/mmz4281/2526/E1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-championship-2025.csv"},
    {"id": "E2", "nome": "🇬🇧 L. One", "history": "https://www.football-data.co.uk/mmz4281/2526/E2.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-league-one-2025.csv"},
    {"id": "E3", "nome": "🇬🇧 L. Two", "history": "https://www.football-data.co.uk/mmz4281/2526/E3.csv", "fixture": "https://fixturedownload.com/download/csv/2025/england-league-two-2025.csv"},
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
        PS, PF = 0.65, 0.35
        
        tot['Att_H'] = ((tot['H_GF_S']*PS + tot['H_GF_F']*PF) / avg_h)
        tot['Dif_H'] = ((tot['H_GS_S']*PS + tot['H_GS_F']*PF) / avg_a)
        tot['Att_A'] = ((tot['A_GF_S']*PS + tot['A_GF_F']*PF) / avg_a)
        tot['Dif_A'] = ((tot['A_GS_S']*PS + tot['A_GS_F']*PF) / avg_h)
        return tot, avg_h, avg_a
    except: return None, None, None

def calculate_stake(prob, quota, bankroll):
    """Kelly Criterion Frazionario (Conservativo)"""
    try:
        if quota <= 1: return 0
        b = quota - 1
        p = prob
        q = 1 - p
        f = (b * p - q) / b
        stake_pct = (f * 0.25) 
        if stake_pct < 0: return 0
        stake = bankroll * stake_pct
        return round(stake, 2)
    except: return 0

def analyze(h, a, stats, ah, aa):
    try:
        if h not in stats.index or a not in stats.index: return None
        
        # Expected Goals (xG)
        lh = stats.at[h,'Att_H'] * stats.at[a,'Dif_A'] * ah
        la = stats.at[a,'Att_A'] * stats.at[h,'Dif_H'] * aa
        
        # 1. Calcolo 1X2
        ph, pd, pa = 0, 0, 0
        for i in range(6):
            for j in range(6):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i>j: ph+=p
                elif i==j: pd+=p
                else: pa+=p
        
        # 2. Calcolo GOAL / NO GOAL
        # Prob che Home segni almeno 1 gol: 1 - P(0)
        p_h_score = 1 - poisson.pmf(0, lh)
        p_a_score = 1 - poisson.pmf(0, la)
        prob_gg = p_h_score * p_a_score
        prob_ng = 1 - prob_gg
        
        # 3. Calcolo OVER / UNDER 2.5
        # Somma lambda totale
        l_tot = lh + la
        # P(0) + P(1) + P(2) è la probabilità di Under 2.5
        prob_u25 = poisson.pmf(0, l_tot) + poisson.pmf(1, l_tot) + poisson.pmf(2, l_tot)
        prob_o25 = 1 - prob_u25
        
        # --- DECISIONE INTELLIGENTE (IL "CONSIGLIO") ---
        cons = "NO BET"
        qmin = 0.0
        prob_win = 0.0
        marg = 1.05
        
        # Lista di tutte le probabilità
        options = [
            ("PUNTA 1", ph, 1/ph if ph>0 else 0),
            ("PUNTA 2", pa, 1/pa if pa>0 else 0),
            ("RISCHIO X", pd, 1/pd if pd>0 else 0),
            ("OVER 2.5", prob_o25, 1/prob_o25 if prob_o25>0 else 0),
            ("UNDER 2.5", prob_u25, 1/prob_u25 if prob_u25>0 else 0),
            ("GOAL (GG)", prob_gg, 1/prob_gg if prob_gg>0 else 0)
        ]
        
        # Filtriamo quelle sopra una soglia di sicurezza
        safe_options = [o for o in options if o[1] > 0.55 or (o[0]=="RISCHIO X" and o[1]>0.33)]
        
        if safe_options:
            # Ordiniamo per probabilità più alta
            safe_options.sort(key=lambda x: x[1], reverse=True)
            best = safe_options[0]
            
            cons = best[0]
            prob_win = best[1]
            qmin = best[2] * marg
        
        return {
            "Match": f"{h} - {a}", 
            "p1": ph, "px": pd, "p2": pa,
            "p_o25": prob_o25, "p_u25": prob_u25, "p_gg": prob_gg,
            "Tip": cons, "Quota": qmin, "ProbWin": prob_win, "c": h, "o": a
        }
    except: return None

# ==============================================================================
# UI
# ==============================================================================
st.title("🔥 AI Betting Pro 4.0")

with st.expander("💰 Bankroll Setting", expanded=False):
    bankroll = st.number_input("Cassa Totale (€):", value=100.0, step=10.0)

tab_auto, tab_manual = st.tabs(["RADAR AUTO", "SCHEDINA"])

# --- TAB AUTO ---
with tab_auto:
    c1, c2 = st.columns(2)
    t_scan = None
    if c1.button("OGGI", use_container_width=True, type="primary"): t_scan = 0
    if c2.button("DOMANI", use_container_width=True): t_scan = 1
    
    if t_scan is not None:
        target_d = (datetime.now() + timedelta(days=t_scan)).strftime('%Y-%m-%d')
        st.info(f"Analisi multi-mercato del {target_d}...")
        
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
                                    res['Stake'] = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                                    results.append(res)
        bar.empty()
        
        if results:
            results.sort(key=lambda x: x['ProbWin'], reverse=True)
            st.success(f"Trovate {len(results)} opportunità.")
            
            for res in results:
                with st.container(border=True):
                    # Header
                    st.markdown(f"**{res['Match']}**")
                    st.caption(f"{res['Lega']}")
                    
                    # Main Info
                    c_main, c_stake = st.columns([3,1])
                    with c_main:
                        if "PUNTA" in res['Tip']: st.success(f"**{res['Tip']}**")
                        elif "OVER" in res['Tip'] or "GOAL" in res['Tip']: st.info(f"**{res['Tip']}**")
                        elif "UNDER" in res['Tip']: st.error(f"**{res['Tip']}**")
                        else: st.warning(res['Tip'])
                        st.caption(f"Quota Minima > {res['Quota']:.2f}")
                    
                    with c_stake:
                        val = res['Stake'] if res['Stake'] > 0 else 0
                        st.markdown(f"""<div class="stake-box"><div class="stake-title">PUNTA</div><div class="stake-value">€{val}</div></div>""", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Dettagli 360
                    st.markdown("###### 📊 Analisi Statistica")
                    # 1X2
                    st.caption("Esito Finale")
                    c1, c2, c3 = st.columns(3)
                    c1.progress(res['p1'], f"1: {res['p1']*100:.0f}%")
                    c2.progress(res['px'], f"X: {res['px']*100:.0f}%")
                    c3.progress(res['p2'], f"2: {res['p2']*100:.0f}%")
                    
                    # GOALS
                    st.caption("Gol & Spettacolo")
                    g1, g2 = st.columns(2)
                    g1.progress(res['p_o25'], f"Over 2.5: {res['p_o25']*100:.0f}%")
                    g2.progress(res['p_gg'], f"Goal: {res['p_gg']*100:.0f}%")

        else:
            st.warning("Nessuna occasione trovata.")

# --- TAB MANUALE ---
with tab_manual:
    names = [d['nome'] for d in DATABASE]
    sel_name = st.selectbox("Seleziona Campionato", names)
    
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
        
        if st.button("➕ Aggiungi al Ticket", use_container_width=True):
            if h != a:
                st.session_state['cart'].append({
                    'c': h, 'o': a, 'lega': sel_name,
                    'stats': st.session_state['cur_stats'],
                    'ah': st.session_state['cur_ah'], 'aa': st.session_state['cur_aa']
                })
                st.toast("Aggiunta!", icon="✅")

    st.divider()
    
    if st.session_state['cart']:
        st.subheader(f"Ticket ({len(st.session_state['cart'])})")
        
        for i, item in enumerate(st.session_state['cart']):
            with st.container(border=True):
                ct, cd = st.columns([5,1])
                ct.text(f"{item['c']} vs {item['o']}")
                if cd.button("❌", key=f"del_{i}"):
                    st.session_state['cart'].pop(i)
                    st.rerun()

        if st.button("🚀 ANALIZZA TUTTO", type="primary", use_container_width=True):
             for item in st.session_state['cart']:
                res = analyze(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    res['Stake'] = calculate_stake(res['ProbWin'], res['Quota'], bankroll)
                    with st.container(border=True):
                        st.markdown(f"### {res['c']} vs {res['o']}")
                        
                        # Main Box
                        c_tip, c_money = st.columns([3,1])
                        with c_tip:
                            if "PUNTA" in res['Tip']: st.success(f"**{res['Tip']}**")
                            elif "OVER" in res['Tip']: st.info(f"**{res['Tip']}**")
                            elif "UNDER" in res['Tip']: st.error(f"**{res['Tip']}**")
                            else: st.warning(res['Tip'])
                            st.caption(f"Quota > {res['Quota']:.2f}")
                        
                        with c_money:
                             val = res['Stake'] if res['Stake'] > 0 else 0
                             st.markdown(f"""<div class="stake-box"><div class="stake-title">PUNTA</div><div class="stake-value">€{val}</div></div>""", unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Dettagli
                        g1, g2 = st.columns(2)
                        g1.progress(res['p_o25'], f"Over 2.5: {res['p_o25']*100:.0f}%")
                        g2.progress(res['p_gg'], f"Goal: {res['p_gg']*100:.0f}%")
                        
                        st.caption("1X2:")
                        st.progress(res['p1'], f"1: {res['p1']*100:.0f}%")
                            
        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
            st.rerun()
