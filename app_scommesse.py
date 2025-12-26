import streamlit as st
import pandas as pd
import requests
import io
from scipy.stats import poisson
import difflib

# ==============================================================================
# CONFIGURAZIONE
# ==============================================================================
st.set_page_config(page_title="TEST CONNESSIONE", page_icon="🔌")

# Link che proviamo a usare (Premier League come test)
# Se questo non va, il radar non può funzionare.
URL_CALENDARIO = "https://fixturedownload.com/download/csv/2025/england-premier-league-2025.csv"
URL_STORICO = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

# ==============================================================================
# FUNZIONI DI CALCOLO (Quelle che funzionavano)
# ==============================================================================
def smart_match_name(name, known_teams):
    matches = difflib.get_close_matches(name, known_teams, n=1, cutoff=0.5)
    return matches[0] if matches else name

def analyze_match(h, a, df_hist):
    try:
        df = df_hist[['Date','HomeTeam','AwayTeam','FTHG','FTAG']].dropna()
        avg_h, avg_a = df['FTHG'].mean(), df['FTAG'].mean()
        
        sc = df.groupby('HomeTeam')[['FTHG','FTAG']].mean()
        st = df.groupby('AwayTeam')[['FTAG','FTHG']].mean()
        
        sc.columns, st.columns = ['H_GF','H_GS'], ['A_GF','A_GS']
        tot = pd.concat([sc,st], axis=1).fillna(1.0) # Fillna per evitare crash
        
        if h not in tot.index or a not in tot.index: return None
        
        att_h = tot.at[h,'H_GF'] / avg_h
        dif_h = tot.at[h,'H_GS'] / avg_a
        att_a = tot.at[a,'A_GF'] / avg_a
        dif_a = tot.at[a,'A_GS'] / avg_h
        
        lh = att_h * dif_a * avg_h
        la = att_a * dif_h * avg_a
        
        ph, pd, pa = 0, 0, 0
        for i in range(6):
            for j in range(6):
                p = poisson.pmf(i, lh) * poisson.pmf(j, la)
                if i>j: ph+=p
                elif i==j: pd+=p
                else: pa+=p
                
        return {"1": ph, "X": pd, "2": pa}
    except: return None

# ==============================================================================
# INTERFACCIA DI DIAGNOSTICA
# ==============================================================================
st.title("🔌 DIAGNOSTICA RADAR")
st.write("Vediamo esattamente cosa succede quando scarichiamo i dati.")

if st.button("AVVIA TEST CONNESSIONE", type="primary"):
    
    # 1. TEST CALENDARIO
    st.divider()
    st.subheader("1. Test Calendario (Fixtures)")
    try:
        r = requests.get(URL_CALENDARIO, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if r.status_code == 200:
            st.success("✅ Calendario Scaricato con successo!")
            df_cal = pd.read_csv(io.StringIO(r.text))
            st.write("Ecco le prime 3 righe grezze del file (Controlla la data):")
            st.dataframe(df_cal.head(3))
            
            # Controllo colonna data
            col_date = next((c for c in df_cal.columns if 'Date' in c or 'Time' in c), None)
            if col_date:
                st.info(f"Colonna data trovata: '{col_date}'. Provo a leggere le partite...")
                # NON FILTRO PER DATA, LE MOSTRO TUTTE LE PRIME 5
                matches = df_cal.head(5)
            else:
                st.error("❌ Non trovo una colonna che sembra una data nel file.")
                matches = pd.DataFrame()
        else:
            st.error(f"❌ Errore scaricamento: Codice {r.status_code}. Il link potrebbe essere vecchio.")
            df_cal = None
            matches = pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Errore connessione: {e}")
        df_cal = None
        matches = pd.DataFrame()

    # 2. TEST STORICO
    st.divider()
    st.subheader("2. Test Storico (Dati)")
    try:
        r2 = requests.get(URL_STORICO, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if r2.status_code == 200:
            st.success("✅ Storico Scaricato con successo!")
            df_hist = pd.read_csv(io.StringIO(r2.text))
            teams_list = sorted(df_hist['HomeTeam'].dropna().unique().tolist())
            st.write(f"Squadre trovate nel database ({len(teams_list)}):", teams_list[:5])
        else:
            st.error(f"❌ Errore scaricamento storico: Codice {r2.status_code}")
            df_hist = None
    except Exception as e:
        st.error(f"❌ Errore connessione storico: {e}")
        df_hist = None

    # 3. TENTATIVO DI INCROCIO
    st.divider()
    st.subheader("3. Tentativo di Analisi")
    
    if df_cal is not None and df_hist is not None and not matches.empty:
        st.write("Provo ad analizzare le prime partite trovate nel calendario (senza filtri di data):")
        
        for _, row in matches.iterrows():
            raw_h = row.get('Home Team', row.get('HomeTeam', 'Unknown'))
            raw_a = row.get('Away Team', row.get('AwayTeam', 'Unknown'))
            
            # Mapping
            real_h = smart_match_name(raw_h, teams_list)
            real_a = smart_match_name(raw_a, teams_list)
            
            st.markdown(f"**Calendario:** {raw_h} vs {raw_a} --> **Database:** {real_h} vs {real_a}")
            
            res = analyze_match(real_h, real_a, df_hist)
            if res:
                prob_1 = res['1'] * 100
                st.write(f"📊 Probabilità 1: {prob_1:.1f}%")
            else:
                st.warning("⚠️ Impossibile calcolare (Dati insufficienti)")
            st.markdown("---")
    else:
        st.warning("Non posso fare l'analisi perché uno dei due file non è stato scaricato.")
