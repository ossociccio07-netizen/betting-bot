import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import io
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAZIONE & STILE NEON
# CONFIGURAZIONE & STILE
# ==============================================================================
DEFAULT_BUDGET = 100.0

st.set_page_config(page_title="BETTING PRO", page_icon="⚽", layout="centered")
st.set_page_config(page_title="BETTING PRO 1X2", page_icon="⚽", layout="centered")

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
    /* TAB STYLE */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #000; padding: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; background-color: #1a1a1a; border: 1px solid #333;
        border-radius: 8px; color: #888; font-weight: bold; font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ff00 !important; color: #000 !important; border: none;
        background-color: #00d26a !important; color: #000 !important; border: none;
    }

    /* TRUCCO PER I METRIC (NUMERI GRANDI) */
    /* METRICHE */
    [data-testid="stMetricLabel"] {
        font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px;
        font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px;
    }
    [data-testid="stMetricValue"] {
        font-size: 28px; font-weight: 900; color: #fff;
        font-size: 24px; font-weight: 900; color: #fff;
    }
    /* Colora il valore del primo metric (Consiglio) in Verde Neon */
    
    /* Colori Specifici Colonne */
    /* Col 1 (Best Tip): Verde Neon */
    div[data-testid="column"]:nth-of-type(1) [data-testid="stMetricValue"] {
        color: #00ff00 !important; text-shadow: 0 0 10px rgba(0,255,0,0.4);
    }
    /* Colora il valore del secondo metric (Soldi) in Bianco */
    /* Col 2 (1X2): Blu Elettrico */
    div[data-testid="column"]:nth-of-type(2) [data-testid="stMetricValue"] {
        color: #ffffff !important;
        color: #00bfff !important;
    }
    /* Col 3 (Soldi): Bianco */
    div[data-testid="column"]:nth-of-type(3) [data-testid="stMetricValue"] {
        color: #ffffff !important; background-color: #222; border-radius: 5px; padding: 0 5px;
    }

    /* BORDI DEI CONTENITORI */
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
# LOGICA (Invariata)
# LOGICA MATEMATICA
# ==============================================================================
DATABASE = [
    {"id": "I1", "nome": "🇮🇹 Serie A", "history": "https://www.football-data.co.uk/mmz4281/2526/I1.csv", "fixture": "https://fixturedownload.com/download/csv/2025/italy-serie-a-2025.csv"},
@@ -106,6 +114,7 @@
def analyze_math(h, a, stats, ah, aa):
    try:
        if h not in stats.index or a not in stats.index: return None
        
        lh = stats.at[h,'Att_H'] * stats.at[a,'Dif_A'] * ah
        la = stats.at[a,'Att_A'] * stats.at[h,'Dif_H'] * aa

@@ -121,23 +130,41 @@
        p_u25 = 1 - p_o25
        p_gg = (1 - poisson.pmf(0, lh)) * (1 - poisson.pmf(0, la))

        # Tutte le opzioni
        options = [
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg}
            {"Tip": "PUNTA 1", "Prob": ph, "Q": 1/ph if ph>0 else 0},
            {"Tip": "PUNTA 2", "Prob": pa, "Q": 1/pa if pa>0 else 0},
            {"Tip": "RISCHIO X", "Prob": pd, "Q": 1/pd if pd>0 else 0},
            {"Tip": "OVER 2.5", "Prob": p_o25, "Q": 1/p_o25 if p_o25>0 else 0},
            {"Tip": "UNDER 2.5", "Prob": p_u25, "Q": 1/p_u25 if p_u25>0 else 0},
            {"Tip": "GOAL", "Prob": p_gg, "Q": 1/p_gg if p_gg>0 else 0}
        ]

        # Calcolo il Favorito 1X2 per mostrarlo a parte
        probs_1x2 = {"1": ph, "X": pd, "2": pa}
        fav_sign = max(probs_1x2, key=probs_1x2.get)
        fav_prob = probs_1x2[fav_sign]
        
        if fav_sign == "1": fav_label = "CASA"
        elif fav_sign == "2": fav_label = "OSPITE"
        else: fav_label = "PAREGGIO"

        # Trovo la Best Option assoluta
        valid = [o for o in options if o['Prob'] > (0.33 if "X" in o['Tip'] else 0.50)]
        if valid:
            valid.sort(key=lambda x: x['Prob'], reverse=True)
            best = valid[0]
        else:
            best = {"Tip": "NO BET", "Prob": 0, "Q": 0}

        return {"c": h, "o": a, "Best": best, "All": options, "xG_H": lh, "xG_A": la}
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
@@ -151,7 +178,7 @@
# ==============================================================================
# UI
# ==============================================================================
st.title("⚽ BETTING PRO")
st.title("⚽ BETTING PRO 1X2")

bankroll = st.number_input("Tuo Budget (€)", value=DEFAULT_BUDGET, step=10.0)

@@ -191,16 +218,14 @@
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
                                        k1, k2, k3 = st.columns(3)
                                        k1.metric("TOP", best['Tip'], f"{best['Prob']*100:.0f}%")
                                        k2.metric("1X2", res['Fav_1X2']['Label'], f"{res['Fav_1X2']['Prob']*100:.0f}%")
                                        k3.metric("QUOTA", f"{best['Q']:.2f}")

        if not found: st.warning("Nessuna occasione sicura trovata.")
        if not found: st.warning("Nessuna occasione sicura al 100% trovata.")

# --- TAB CARRELLO ---
with tab_cart:
@@ -233,36 +258,56 @@
                st.session_state['cart'].pop(i)
                st.rerun()

        if st.button("🚀 CALCOLA SCHEDINA", type="primary", use_container_width=True):
        if st.button("🚀 CALCOLA ANALISI COMPLETA", type="primary", use_container_width=True):
            for item in st.session_state['cart']:
                res = analyze_math(item['c'], item['o'], item['stats'], item['ah'], item['aa'])
                if res:
                    best = res['Best']
                    fav = res['Fav_1X2']
                    stake = calculate_stake(best['Prob'], best['Q']*1.05, bankroll)

                    # === QUI LA MAGIA ===
                    # Usiamo st.container e st.metric invece dell'HTML
                    # Questo previene al 100% la stampa del codice
                    with st.container(border=True):
                        st.markdown(f"#### {item['c']} <span style='color:#888'>vs</span> {item['o']}", unsafe_allow_html=True)
                        # Intestazione
                        st.markdown(f"#### {item['c']} <span style='color:#666'>vs</span> {item['o']}", unsafe_allow_html=True)

                        col_tip, col_stake = st.columns(2)
                        # 3 Colonne Magiche
                        c_top, c_1x2, c_soldi = st.columns(3)

                        # Colonna Sinistra: Consiglio (Verde Neon grazie al CSS in alto)
                        col_tip.metric(
                            label="MIGLIOR SCELTA",
                        # 1. Miglior Scelta (es. Over 2.5)
                        c_top.metric(
                            label="STRATEGIA",
                            value=best['Tip'],
                            delta=f"Prob: {best['Prob']*100:.1f}%"
                            delta=f"{best['Prob']*100:.1f}% Sicurezza"
                        )
                        
                        # 2. Chi Vince (1X2)
                        c_1x2.metric(
                            label="FAVORITO 1X2",
                            value=fav['Label'],
                            delta=f"{fav['Prob']*100:.1f}% Prob."
                        )

                        # Colonna Destra: Soldi
                        col_stake.metric(
                            label="PUNTA",
                        # 3. Soldi
                        c_soldi.metric(
                            label="PUNTARE",
                            value=f"€{stake}",
                            delta=f"Quota: {best['Q']:.2f}"
                            delta=f"Quota {best['Q']:.2f}"
                        )

                        st.markdown(f"<div style='font-size:12px; color:#666'>xG Casa: {res['xG_H']:.2f} | xG Ospite: {res['xG_A']:.2f}</div>", unsafe_allow_html=True)
                        st.divider()
                        
                        # Barre 1X2 visibili subito
                        st.caption("Probabilità Esito Finale (1X2):")
                        p = res['Probs']
                        
                        # Creiamo 3 colonne piccole per le barre
                        b1, b2, b3 = st.columns(3)
                        b1.progress(p['1'], f"1: {p['1']*100:.0f}%")
                        b2.progress(p['X'], f"X: {p['X']*100:.0f}%")
                        b3.progress(p['2'], f"2: {p['2']*100:.0f}%")
                        
                        # xG Info
                        st.markdown(f"<div style='text-align:center; font-size:11px; color:#666; margin-top:10px;'>xG Attesi: {res['xG_H']:.2f} - {res['xG_A']:.2f}</div>", unsafe_allow_html=True)

        if st.button("Svuota tutto", use_container_width=True):
            st.session_state['cart'] = []
