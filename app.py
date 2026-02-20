import streamlit as st
import pandas as pd

MATCHES_CSV = "matches_with_elo (2).csv"

st.set_page_config(page_title="Football AI Dashboard", layout="wide")

st.title("⚽ Upcoming Matches")

# Load matches
matches_df = pd.read_csv(MATCHES_CSV)
matches_df["Date"] = pd.to_datetime(matches_df["Date"], errors="coerce")
today = pd.Timestamp.today().normalize()
upcoming_matches = matches_df[matches_df["Date"] >= today].sort_values("Date")

# Session state
if "selected_match" not in st.session_state:
    st.session_state.selected_match = None

# Show matches
for match_date, group in upcoming_matches.groupby(upcoming_matches["Date"].dt.date):
    st.subheader(f"📅 {match_date}")
    
    for idx, row in group.iterrows():
        col1, col2, col3, col4 = st.columns([4,1,4,2])
        col1.write(row["Team1"])
        col2.write("vs")
        col3.write(row["Team2"])

        if col4.button("Select Match", key=f"m_{idx}"):
            st.session_state.selected_match = row.to_dict()

            # ✅ AUTO REDIRECT TO PREDICTION PAGE
            st.switch_page("pages/1_Prediction.py")