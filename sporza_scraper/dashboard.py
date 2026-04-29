"""Streamlit dashboard for Premier League sentiment & predictions.

Launch with:  streamlit run sporza_scraper/dashboard.py
"""
import sqlite3
import os
import sys

# Ensure the project root is on the Python path so imports work
# regardless of how Streamlit is launched.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from pathlib import Path
from sporza_scraper.predictor import predict_match

# 1. Page settings
st.set_page_config(page_title="Sporza PL Dashboard", page_icon="⚽", layout="wide")
st.title("Premier League Sentiment & Predictions")
st.markdown(
    "Interactive dashboard powered by **Sporza.be** news data "
    "and a hybrid Poisson model."
)

# 2. Database connection
DB_PATH = os.path.join("data", "sporza_predictions.db")
if not os.path.exists(DB_PATH):
    st.warning(
        f"⚠️ Database not found at `{DB_PATH}`. "
        "Run the scraper first with `--pl` to fetch data!"
    )
    st.stop()

@st.cache_resource
def _get_connection():
    """Open a single read-only connection, reused across Streamlit re-runs."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


conn = _get_connection()

# --- SECTION 1: TEAM MOMENTUM ---
st.header("1. Team Momentum (News Sentiment)")
st.write("Which clubs are most positive (or negative) in the news?")

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
    st.info("No team data available yet. Run the scraper to fetch data.")

st.divider()

# --- SECTION 2: TOP PLAYERS ---
st.header("2. Featured Players (Form & Threat)")

query_players = """
    SELECT
        player_name AS Player,
        club        AS Club,
        AVG(score)  AS Average_Score,
        COUNT(*)    AS Mentions
    FROM signals
    WHERE player_name IS NOT NULL
    GROUP BY player_name
    HAVING Mentions >= 1
    ORDER BY Average_Score DESC
    LIMIT 10
"""
df_players = pd.read_sql_query(query_players, conn)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top 10 Players")
    st.dataframe(df_players, use_container_width=True)
with col2:
    st.subheader("Visual Trend")
    if not df_players.empty:
        st.bar_chart(df_players.set_index("Player")["Average_Score"])

st.divider()

# --- SECTION 3: MATCH PREDICTOR ---
st.header("3. Hybrid Match Predictor")
st.write("Select two teams to calculate win probabilities and expected scores.")

pl_clubs = sorted([
    "Arsenal", "Aston Villa", "Brentford", "Brighton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Ipswich Town", "Leicester City",
    "Liverpool", "Manchester City", "Manchester United", "Newcastle United",
    "Nottingham Forest", "Southampton", "Tottenham Hotspur", "West Ham United",
    "Wolverhampton Wanderers",
])

team_col1, team_col2 = st.columns(2)
with team_col1:
    team_home = st.selectbox("Home Team", pl_clubs, index=pl_clubs.index("Arsenal"))
with team_col2:
    team_away = st.selectbox("Away Team", pl_clubs, index=pl_clubs.index("Manchester City"))

if st.button("Predict Match", type="primary"):
    if team_home == team_away:
        st.error("Please select two different teams!")
    else:
        with st.spinner("Combining statistics and sentiment..."):
            prediction = predict_match(team_home, team_away, data_root=Path("data"))

        st.success(
            f"Analysis complete! Model confidence: **{prediction.confidence.upper()}**"
        )

        met1, met2, met3 = st.columns(3)
        met1.metric(f"Win {team_home}", f"{prediction.home_win_prob * 100:.1f}%")
        met2.metric("Draw", f"{prediction.draw_prob * 100:.1f}%")
        met3.metric(f"Win {team_away}", f"{prediction.away_win_prob * 100:.1f}%")

        st.markdown(
            f"**Expected score:** {team_home} "
            f"**{prediction.expected_home_goals:.1f} - "
            f"{prediction.expected_away_goals:.1f}** {team_away}"
        )

        if prediction.sporza_modifiers:
            st.info(
                "💡 **Impact of recent Sporza News on this match:**\n"
                + "\n".join(f"- {mod}" for mod in prediction.sporza_modifiers)
            )

# Connection is managed by st.cache_resource — no manual close needed.
