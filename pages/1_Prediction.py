import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from models import predict_match_dynamic

# ================= CONFIG =================
BEST_MODEL_PATH = "best_match_outcome_model.pkl"
XG_TEAM1_PATH = "xg_model_team1.pkl"
XG_TEAM2_PATH = "xg_model_team2.pkl"
MATCHES_CSV = "matches_with_elo (2).csv"

st.set_page_config(page_title="Match Prediction", layout="wide")

# ================= DARK THEME CSS =================
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
div[data-testid="metric-container"] {
    background-color:#111827;
    padding:15px;
    border-radius:12px;
    border:1px solid #374151;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODELS =================
@st.cache_resource
def load_models():
    return joblib.load(BEST_MODEL_PATH), joblib.load(XG_TEAM1_PATH), joblib.load(XG_TEAM2_PATH)

best_model, xg1, xg2 = load_models()

# ================= CHECK SELECTED MATCH =================
if "selected_match" not in st.session_state:
    st.error("No match selected")
    st.stop()

selected = st.session_state.selected_match
team1 = selected["Team1"]
team2 = selected["Team2"]
match_date = selected["Date"]
match_type = selected.get("Match_type", "Unknown")

# ================= RUN AI =================
with st.spinner("Running AI prediction..."):
    raw = predict_match_dynamic(team1, team2, best_model, xg1, xg2, csv_path=MATCHES_CSV)
    result = raw["result"] if "result" in raw else raw

# ================= BUTTON NAVIGATION =================
if "section" not in st.session_state:
    st.session_state.section = "Match Info"   # DEFAULT

col1, col2, col3, col4 = st.columns(4)
if col1.button("🧾 Match Info"): st.session_state.section = "Match Info"
if col2.button("🔮 Prediction"): st.session_state.section = "Prediction"
if col3.button("📈 Form"): st.session_state.section = "Form"
if col4.button("🤝 H2H"): st.session_state.section = "H2H"

st.markdown("---")

# ================= MATCH INFO =================
if st.session_state.section == "Match Info":
    st.header("📅 Match Information")
    c1, c2, c3 = st.columns(3)
    c1.metric("Team 1", team1)
    c2.metric("Team 2", team2)
    c3.metric("Match Type", match_type)
    st.metric("Match Date", match_date.strftime("%Y-%m-%d"))

# ================= PREDICTION =================
elif st.session_state.section == "Prediction":
    st.header("🔮 AI Prediction")
    outcome = result["Match Outcome (%)"]

    # Pie chart for Match Outcome
    df_outcome = pd.DataFrame({
        "Outcome": [team1, "Draw", team2],
        "Probability": [outcome["Team1"], outcome["Draw"], outcome["Team2"]]
    })
    fig = px.pie(df_outcome, names="Outcome", values="Probability", hole=0.55,
                 color="Outcome", color_discrete_map={team1:"#1f77b4","Draw":"gray",team2:"#d62728"})
    st.plotly_chart(fig, use_container_width=True)

    # Top Scorelines with probability instead of odds
    st.subheader("Top Scorelines")
    score_df = pd.DataFrame(result["top_scorelines"])

    # Convert to percentage if not already
    score_df["prob_percent"] = score_df["prob"] * 100  # assuming prob is in 0-1

    fig2 = px.bar(
        score_df,
        x="score",
        y="prob_percent",
        text=score_df["prob_percent"].map(lambda x: f"{x:.1f}%"),
        color="prob_percent",
        color_continuous_scale="blues",
        labels={"prob_percent": "Probability (%)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Goal Markets
    st.subheader("Goal Markets")
    btts = result["BTTS (%)"]
    over25 = result["Over 2.5 Goals (%)"]

    st.write("BTTS Probability")
    st.progress(int(btts))
    st.write(f"{btts:.1f}%")

    st.write("Over 2.5 Goals Probability")
    st.progress(int(over25))
    st.write(f"{over25:.1f}%")

# ================= FORM =================
elif st.session_state.section == "Form":
    st.header("📈 Recent Form Comparison")
    form = result["Form"]

    # Metrics and their keys for each team
    metrics = {
        "Wins": ("team1_Wins", "team2_Wins"),
        "Draws": ("team1_Draws", "team2_Draws"),
        "Losses": ("team1_Losing", "team2_Losing"),
        "Goals Scored": ("team1_Goals_per_game", "team2_Goals_per_game"),
        "Goals Conceded": ("team1_Goals_conceded_per_game", "team2_Goals_conceded_per_game"),
        "Clean Sheets": ("team1_Clean_sheets", "team2_Clean_sheets"),
        "Shots": ("team1_Shots", "team2_Shots"),
        "Shots on Target": ("team1_Shots_on_target", "team2_Shots_on_target"),
        "Possession %": ("team1_Possession", "team2_Possession"),
        "Corners": ("team1_Corners", "team2_Corners")
    }

    # Custom colors for the progress bars
    team_colors = {team1: "#1f77b4", team2: "#d62728"}

    for metric_name, (t1_key, t2_key) in metrics.items():
        st.subheader(metric_name)

        # Columns for the two teams
        c1, c2 = st.columns(2)

        # Display team names at the top
        c1.markdown(f"**{team1}**", unsafe_allow_html=True)
        c2.markdown(f"**{team2}**", unsafe_allow_html=True)

        # Get the last 10 matches
        t1_values = form[t1_key][-10:] if isinstance(form[t1_key], list) else [form[t1_key]]
        t2_values = form[t2_key][-10:] if isinstance(form[t2_key], list) else [form[t2_key]]

        # Find max for scaling
        max_val = max(max(t1_values), max(t2_values), 1)  # avoid division by zero

        # Team 1 progress bars
        for val in t1_values[::-1]:
            c1.markdown(f'<div style="background-color:#374151; border-radius:5px; padding:2px; margin-bottom:2px;">'
                        f'<div style="width:{(val/max_val)*100}%; background-color:{team_colors[team1]}; color:white; '
                        f'padding:2px; border-radius:5px;">{val}</div></div>', unsafe_allow_html=True)

        # Team 2 progress bars
        for val in t2_values[::-1]:
            c2.markdown(f'<div style="background-color:#374151; border-radius:5px; padding:2px; margin-bottom:2px;">'
                        f'<div style="width:{(val/max_val)*100}%; background-color:{team_colors[team2]}; color:white; '
                        f'padding:2px; border-radius:5px;">{val}</div></div>', unsafe_allow_html=True)

# ================= H2H =================
elif st.session_state.section == "H2H":
    st.header("🤝 Head to Head")
    h2h = result["h2h"]
    df_wins = pd.DataFrame({
        "Team": [team1, team2],
        "Wins": [h2h["wins_team1"], h2h["wins_team2"]]
    })
    fig = px.bar(df_wins, x="Team", y="Wins", color="Team",
                 color_discrete_map={team1:"#1f77b4", team2:"#d62728"})
    st.plotly_chart(fig, use_container_width=True)
    df_goals = pd.DataFrame({
        "Team": [team1, team2],
        "Goals": [h2h["goals_team1"], h2h["goals_team2"]]
    })
    fig2 = px.bar(df_goals, x="Team", y="Goals", color="Team",
                  color_discrete_map={team1:"#1f77b4", team2:"#d62728"})
    st.plotly_chart(fig2, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Total Matches", h2h["total_matches"])
    c2.metric("Goal Difference", h2h["goal_diff"])