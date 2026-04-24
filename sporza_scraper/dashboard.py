"""Streamlit dashboard for Premier League sentiment & predictions.

Launch with:  streamlit run sporza_scraper/dashboard.py
"""
import sqlite3
import os

import pandas as pd
import streamlit as st

from sporza_scraper.predictor import predict_match

# 1. Page settings
st.set_page_config(page_title="Sporza PL Dashboard", page_icon="⚽", layout="wide")
st.title("Premier League Sentiment & Voorspellingen")
st.markdown(
    "Interactief dashboard aangedreven door **Sporza.be** nieuwsdata "
    "en een hybride Poisson-model."
)

# 2. Database connection
DB_PATH = os.path.join("data", "sporza_predictions.db")
if not os.path.exists(DB_PATH):
    st.warning(
        f"⚠️ Database niet gevonden op `{DB_PATH}`. "
        "Laat eerst de scraper draaien met `--pl` om data op te halen!"
    )
    st.stop()

conn = sqlite3.connect(DB_PATH)

# --- SECTION 1: TEAM MOMENTUM ---
st.header("1. Team Momentum (Nieuws Sentiment)")
st.write("Welke clubs zijn het meest positief (of negatief) in het nieuws?")

query_teams = """
    SELECT club, AVG(score) as avg_score
    FROM signals
    WHERE club != 'Unknown'
    GROUP BY club
    ORDER BY avg_score DESC
"""
df_teams = pd.read_sql_query(query_teams, conn)

if not df_teams.empty:
    st.bar_chart(df_teams.set_index("club")["avg_score"])
else:
    st.info("Nog geen teamdata beschikbaar. Draai de scraper om data op te halen.")

st.divider()

# --- SECTION 2: TOP PLAYERS ---
st.header("2. Uitgelichte Spelers (Vorm & Gevaar)")

query_players = """
    SELECT
        player_name AS Speler,
        club        AS Club,
        AVG(score)  AS Gemiddelde_Score,
        COUNT(*)    AS Vermeldingen
    FROM signals
    WHERE player_name IS NOT NULL
    GROUP BY player_name
    HAVING Vermeldingen >= 1
    ORDER BY Gemiddelde_Score DESC
    LIMIT 10
"""
df_players = pd.read_sql_query(query_players, conn)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Spelers")
    st.dataframe(df_players, use_container_width=True)
with col2:
    st.subheader("Visuele Trend")
    if not df_players.empty:
        st.bar_chart(df_players.set_index("Speler")["Gemiddelde_Score"])

st.divider()

# --- SECTION 3: MATCH PREDICTOR ---
st.header("3. Hybride Match Voorspeller")
st.write("Selecteer twee teams om de winstkansen en verwachte score te berekenen.")

pl_clubs = sorted([
    "Arsenal", "Aston Villa", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leicester City",
    "Liverpool", "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Southampton", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers",
])

team_col1, team_col2 = st.columns(2)
with team_col1:
    team_home = st.selectbox("Thuisteam", pl_clubs, index=pl_clubs.index("Arsenal"))
with team_col2:
    team_away = st.selectbox("Uitteam", pl_clubs, index=pl_clubs.index("Manchester City"))

if st.button("Voorspel Wedstrijd", type="primary"):
    if team_home == team_away:
        st.error("Kies twee verschillende teams!")
    else:
        with st.spinner("Statistieken en sentiment combineren..."):
            voorspelling = predict_match(team_home, team_away, data_root="data")

        st.success(
            f"Analyse voltooid! Betrouwbaarheid model: **{voorspelling.confidence.upper()}**"
        )

        met1, met2, met3 = st.columns(3)
        met1.metric(f"Winst {team_home}", f"{voorspelling.home_win_prob * 100:.1f}%")
        met2.metric("Gelijkspel", f"{voorspelling.draw_prob * 100:.1f}%")
        met3.metric(f"Winst {team_away}", f"{voorspelling.away_win_prob * 100:.1f}%")

        st.markdown(
            f"**Verwachte score:** {team_home} "
            f"**{voorspelling.expected_home_goals:.1f} - "
            f"{voorspelling.expected_away_goals:.1f}** {team_away}"
        )

        if voorspelling.sporza_modifiers:
            st.info(
                "💡 **Impact van recent Sporza Nieuws op deze match:**\n"
                + "\n".join(f"- {mod}" for mod in voorspelling.sporza_modifiers)
            )

conn.close()
