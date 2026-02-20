import os
import pandas as pd
from scipy.stats import poisson
from typing import Dict, Any

import math

def bivariate_poisson_pmf(x, y, l1, l2, rho):
    s = 0.0
    for k in range(min(x, y) + 1):
        s += (
            l1**(x-k) *
            l2**(y-k) *
            rho**k
        ) / (
            math.factorial(x-k) *
            math.factorial(y-k) *
            math.factorial(k)
        )
    return math.exp(-(l1 + l2 + rho)) * s

def dixon_coles_tau(x, y, l1, l2, rho):
    if x == 0 and y == 0:
        return 1 - l1*l2*rho
    elif x == 1 and y == 0:
        return 1 + l2*rho
    elif x == 0 and y == 1:
        return 1 + l1*rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1

def dc_bivariate_score_prob(x, y, l1, l2, rho):
    base = bivariate_poisson_pmf(x, y, l1, l2, rho)
    tau = dixon_coles_tau(x, y, l1, l2, rho)
    return max(base * tau, 0)

def scoreline_probabilities(l1, l2, rho, max_goals=6):
    probs = {}
    total = 0.0

    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            p = dc_bivariate_score_prob(x, y, l1, l2, rho)
            probs[(x, y)] = p
            total += p

    # Normalize
    for k in probs:
        probs[k] /= total

    return probs

def top_scorelines(probs, n=5):
    return sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

def fair_odds(p):
    return round(1 / p, 2) if p > 0 else None


def full_match_prediction(features_dict: Dict[str, Any],
                          best_model_pipeline,
                          xg_model_team1,
                          xg_model_team2,
                          max_goals: int = 10) -> Dict[str, Any]:
    """
    Predict match outcome, expected goals, Over/Under 2.5, BTTS.
    - features_dict: dictionary aligned with your model features
    - best_model_pipeline: classifier pipeline with predict_proba and classes_
    - xg_model_team1, xg_model_team2: regressors that predict expected goals for each team
    """

    # 1) Match Outcome probabilities
    X_input = pd.DataFrame([features_dict])
    proba = best_model_pipeline.predict_proba(X_input)[0]
    classes = list(best_model_pipeline.classes_)
    match_outcome = {str(classes[i]): round(float(proba[i]) * 100, 2) for i in range(len(proba))}

    # 2) Expected Goals
    # Build X for each xg model. The keys below should match how the xg models were trained.
    xg_input_team1 = pd.DataFrame([{
        'team1_Goals_per_shot': features_dict.get('team1_Goals_per_shot', 0),
        'team1_Goals_conceded_per_shot': features_dict.get('team1_Goals_conceded_per_shot', 0),
        'shot_accuracy_team1': features_dict.get('shot_accuracy_team1', 0),
        'Home_Elo': features_dict.get('Home_Elo', 0),
        'Away_Elo': features_dict.get('Away_Elo', 0)
    }])

    xg_input_team2 = pd.DataFrame([{
        'team2_Goals_per_shot': features_dict.get('team2_Goals_per_shot', 0),
        'team2_Goals_conceded_per_shot': features_dict.get('team2_Goals_conceded_per_shot', 0),
        # Some original code inverted shot accuracy for away; preserve behavior if model expects it.
        'shot_accuracy_team2': -features_dict.get('shot_accuracy_team2', 0),
        'Away_Elo': features_dict.get('Away_Elo', 0),
        'Home_Elo': features_dict.get('Home_Elo', 0)
    }])

    xg_team1 = float(xg_model_team1.predict(xg_input_team1)[0])
    xg_team2 = float(xg_model_team2.predict(xg_input_team2)[0])
    
    rho = 0.10

    # 3️⃣ Score matrix
    probs = scoreline_probabilities(xg_team1, xg_team2, rho)
    top_scores = top_scorelines(probs)

    # 3) Over/Under 2.5 Goals (probability that total goals > 2)
    over_25_prob = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if (i + j) > 2:
                over_25_prob += poisson.pmf(i, xg_team1) * poisson.pmf(j, xg_team2)
    over_25_prob *= 100.0

    # 4) BTTS: probability both teams score at least once
    btts_prob = 0.0
    for i in range(1, max_goals + 1):
        for j in range(1, max_goals + 1):
            btts_prob += poisson.pmf(i, xg_team1) * poisson.pmf(j, xg_team2)
    btts_prob *= 100.0

    return {
        "Match Outcome (%)": match_outcome,
	"top_scorelines": [
            {"score": f"{x}-{y}", "prob": round(p,4), "odds": fair_odds(p)}
            for (x,y),p in top_scores
        ],
        "Expected Goals": {"Team1": round(xg_team1, 2), "Team2": round(xg_team2, 2)},
        "Over 2.5 Goals (%)": round(over_25_prob, 2),
        "BTTS (%)": round(btts_prob, 2)
    }


def predict_match_dynamic(team1_name: str,
                          team2_name: str,
                          best_model_pipeline,
                          xg_model_team1,
                          xg_model_team2,
                          csv_path: str = None) -> Dict[str, Any]:
    """
    Build features from dataset and call full_match_prediction.
    - team1_name, team2_name: team names as present in the CSV
    - csv_path: path to dataset CSV. If None, will look for env var MATCHES_CSV or './matches_with_elo.csv'
    """

    if csv_path is None:
        csv_path = os.environ.get("MATCHES_CSV", "./matches_with_elo.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    # Normalize column names
    df.columns = df.columns.str.strip()
    # Parse Date if present
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure numeric columns are numeric
    numeric_cols = [
        'Form_points_team1', 'Form_points_team2',
        'team1_Goals scored', 'team2_Goals scored',
        'team1_Goals conceded', 'team2_Goals conceded',
        'team1_Shots', 'team2_Shots',
        'team1_Shots on Target', 'team2_Shots on Target',
        'team1_Goals scored per game', 'team2_Goals scored per game',
        'team1_Goals conceded per game', 'team2_Goals conceded per game',
        'Home_Elo', 'Away_Elo',
        'team1_Clean sheets', 'team2_Clean sheets',
        'team1_Wins', 'team2_Wins',
        'team1_Draws', 'team2_Draws',
        'team1_Losing', 'team2_Losing',
        'h2h_Wins_team1', 'h2h_Wins_team2',
        'h2h_Scored goals_team1', 'h2h_Scored goals_team2',
        'team1_Goals scored', 'team2_Goals scored'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get most recent row where team played as Team1 (home) and team2 played as Team2 (away)
    team1_rows = df[(df['Team1'] == team1_name) & (df['Team2'] == team2_name)].sort_values(by='Date', ascending=False).head(1)
    team2_rows = df[(df['Team1'] == team1_name) & (df['Team2'] == team2_name)].sort_values(by='Date', ascending=False).head(1)

    if team1_rows.empty:
        raise ValueError(f"No rows found for team1 '{team1_name}' as Team1 in dataset.")
    if team2_rows.empty:
        raise ValueError(f"No rows found for team2 '{team2_name}' as Team2 in dataset.")

    team1_data = team1_rows.iloc[0]
    team2_data = team2_rows.iloc[0]

    # Build features dictionary (mirrors the original structure)
    def safe_get(row, key, default=0):
        return float(row[key]) if key in row and pd.notna(row[key]) else default

    features_dict = {
        'Form_points_diff': safe_get(team1_data, 'Form_points_team1') - safe_get(team2_data, 'Form_points_team2'),
        'Goals_diff': safe_get(team1_data, 'team1_Goals scored') - safe_get(team2_data, 'team2_Goals scored'),
        'Goals_conceded_diff': safe_get(team1_data, 'team1_Goals conceded') - safe_get(team2_data, 'team2_Goals conceded'),
        'Clean_sheets_diff': safe_get(team1_data, 'team1_Clean sheets') - safe_get(team2_data, 'team2_Clean sheets'),
        'Shots_diff': safe_get(team1_data, 'team1_Shots') - safe_get(team2_data, 'team2_Shots'),
        'Shots_on_target_diff': safe_get(team1_data, 'team1_Shots on Target') - safe_get(team2_data, 'team2_Shots on Target'),
        'Possession_diff': safe_get(team1_data, 'team1_Possession') - safe_get(team2_data, 'team2_Possession'),
        'h2h_Wins_diff': safe_get(team1_data, 'h2h_Wins_team1') - safe_get(team2_data, 'h2h_Wins_team2'),
        'h2h_Goals_diff': safe_get(team1_data, 'h2h_Scored goals_team1') - safe_get(team2_data, 'h2h_Scored goals_team2'),
        'h2h_Goals_ratio': (safe_get(team1_data, 'h2h_Scored goals_team1') / max(1.0, safe_get(team2_data, 'h2h_Scored goals_team2'))),
        'Elo_diff': safe_get(team1_data, 'Home_Elo') - safe_get(team2_data, 'Away_Elo'),
        'Elo_ratio': safe_get(team1_data, 'Home_Elo') / max(1.0, safe_get(team2_data, 'Away_Elo')),
        'Goals_per_shot_diff': (safe_get(team1_data, 'team1_Goals scored per game') - safe_get(team2_data, 'team2_Goals scored per game')),
        'Defense_diff': (safe_get(team1_data, 'team1_Goals conceded per game') - safe_get(team2_data, 'team2_Goals conceded per game')),
        'team1_Goals_per_shot': safe_get(team1_data, 'team1_Goals scored per game'),
        'team2_Goals_per_shot': safe_get(team2_data, 'team2_Goals scored per game'),
        'team1_Goals_conceded_per_shot': safe_get(team1_data, 'team1_Goals conceded per game'),
        'team2_Goals_conceded_per_shot': safe_get(team2_data, 'team2_Goals conceded per game'),
        'h2h_Scored goals_team1': safe_get(team1_data, 'h2h_Scored goals_team1'),
        'h2h_Scored goals_team2': safe_get(team2_data, 'h2h_Scored goals_team2'),
        'h2h_Wins_team1': safe_get(team1_data, 'h2h_Wins_team1'),
        'h2h_Wins_team2': safe_get(team2_data, 'h2h_Wins_team2'),
        'Home_Elo': safe_get(team1_data, 'Home_Elo'),
        'Away_Elo': safe_get(team2_data, 'Away_Elo'),
        'team1_Wins': safe_get(team1_data, 'team1_Wins'),
        'team2_Wins': safe_get(team2_data, 'team2_Wins'),
        'team1_Draws': safe_get(team1_data, 'team1_Draws'),
        'team2_Draws': safe_get(team2_data, 'team2_Draws'),
        'team1_Losing': safe_get(team1_data, 'team1_Losing'),
        'team2_Losing': safe_get(team2_data, 'team2_Losing'),
        'shot_accuracy_team1': safe_get(team1_data, 'team1_Shots on Target') / max(1.0, safe_get(team1_data, 'team1_Shots')),
        'shot_accuracy_team2': safe_get(team2_data, 'team2_Shots on Target') / max(1.0, safe_get(team2_data, 'team2_Shots')),
        'shot_accuracy_diff': (safe_get(team1_data, 'team1_Shots on Target') / max(1.0, safe_get(team1_data, 'team1_Shots'))) - (safe_get(team2_data, 'team2_Shots on Target') / max(1.0, safe_get(team2_data, 'team2_Shots')))
    }
        # --- Head-to-head summary (for API/UI response only) ---
    h2h_summary = {
        "team1": team1_name,
        "team2": team2_name,
        "wins_team1": safe_get(team1_data, 'h2h_Wins_team1'),
        "wins_team2": safe_get(team2_data, 'h2h_Wins_team2'),
        "goals_team1": safe_get(team1_data, 'h2h_Scored goals_team1'),
        "goals_team2": safe_get(team2_data, 'h2h_Scored goals_team2'),
        "goal_diff": (
            safe_get(team1_data, 'h2h_Scored goals_team1')
            - safe_get(team2_data, 'h2h_Scored goals_team2')
        ),
        "total_matches": (
            safe_get(team1_data, 'h2h_Wins_team1')
            + safe_get(team2_data, 'h2h_Wins_team2')
            + safe_get(team1_data, 'h2h_Draws', 0)
        )
    }
    Form_summary = {
    	# Basic last 10 matches form
    	'team1_Wins': safe_get(team1_data, 'team1_Wins'),
    	'team2_Wins': safe_get(team2_data, 'team2_Wins'),
    	'team1_Draws': safe_get(team1_data, 'team1_Draws'),
    	'team2_Draws': safe_get(team2_data, 'team2_Draws'),
    	'team1_Losing': safe_get(team1_data, 'team1_Losing'),
    	'team2_Losing': safe_get(team2_data, 'team2_Losing'),

	    # Elo ratings
	    'Home_Elo': safe_get(team1_data, 'Home_Elo'),
	    'Away_Elo': safe_get(team2_data, 'Away_Elo'),

    # Goals
	    'team1_Goals_scored': safe_get(team1_data, 'team1_Goals scored'),
        'team2_Goals_scored': safe_get(team2_data, 'team2_Goals scored'),
        'team1_Goals_conceded': safe_get(team1_data, 'team1_Goals conceded'),
    	'team2_Goals_conceded': safe_get(team2_data, 'team2_Goals conceded'),
    	'team1_Goals_per_game': safe_get(team1_data, 'team1_Goals scored per game'),
    	'team2_Goals_per_game': safe_get(team2_data, 'team2_Goals scored per game'),
    	'team1_Goals_conceded_per_game': safe_get(team1_data, 'team1_Goals conceded per game'),
    	'team2_Goals_conceded_per_game': safe_get(team2_data, 'team2_Goals conceded per game'),

   	 # Clean sheets
    	'team1_Clean_sheets': safe_get(team1_data, 'team1_Clean sheets'),
    	'team2_Clean_sheets': safe_get(team2_data, 'team2_Clean sheets'),

    # BTTS
    	'team1_BTTS': safe_get(team1_data, 'team1_both teams to score'),
    	'team2_BTTS': safe_get(team2_data, 'team2_both teams to score'),

    # Over/Under 2.5
    	'team1_TU_2.5': safe_get(team1_data, 'team1_TU 2.5'),
    	'team2_TU_2.5': safe_get(team2_data, 'team2_TU 2.5'),
    	'team1_TO_2.5': safe_get(team1_data, 'team1_TO 2.5'),
    	'team2_TO_2.5': safe_get(team2_data, 'team2_TO 2.5'),

    # Shots and on target
    	'team1_Shots': safe_get(team1_data, 'team1_Shots'),
    	'team2_Shots': safe_get(team2_data, 'team2_Shots'),
    	'team1_Shots_on_target': safe_get(team1_data, 'team1_Shots on Target'),
    	'team2_Shots_on_target': safe_get(team2_data, 'team2_Shots on Target'),
    	'team1_Shots_competitor': safe_get(team1_data, 'team1_Shots  (competitor)'),
    	'team2_Shots_competitor': safe_get(team2_data, 'team2_Shots  (competitor)'),
    	'team1_Shots_on_target_competitor': safe_get(team1_data, 'team1_Shots on Target  (competitor)'),
    	'team2_Shots_on_target_competitor': safe_get(team2_data, 'team2_Shots on Target  (competitor)'),

    # Possession
    	'team1_Possession': safe_get(team1_data, 'team1_Possession'),
        'team2_Possession': safe_get(team2_data, 'team2_Possession'),
        'team1_Possession_competitor': safe_get(team1_data, 'team1_Possession  (competitor)'),
        'team2_Possession_competitor': safe_get(team2_data, 'team2_Possession  (competitor)'),

        # Corners
        'team1_Corners': safe_get(team1_data, 'team1_Corners'),
        'team2_Corners': safe_get(team2_data, 'team2_Corners'),
        'team1_Corners_competitor': safe_get(team1_data, 'team1_Corners  (competitor)'),
        'team2_Corners_competitor': safe_get(team2_data, 'team2_Corners  (competitor)'),

        # Fouls
        'team1_Fouls': safe_get(team1_data, 'team1_Fouls'),
        'team2_Fouls': safe_get(team2_data, 'team2_Fouls'),
        'team1_Fouls_competitor': safe_get(team1_data, 'team1_Fouls  (competitor)'),
        'team2_Fouls_competitor': safe_get(team2_data, 'team2_Fouls  (competitor)'),

        # Offsides
        'team1_Offsides': safe_get(team1_data, 'team1_Offsides'),
        'team2_Offsides': safe_get(team2_data, 'team2_Offsides'),
        'team1_Offsides_competitor': safe_get(team1_data, 'team1_Offsides  (competitor)'),
        'team2_Offsides_competitor': safe_get(team2_data, 'team2_Offsides  (competitor)'),

        # Cards
        'team1_Yellow': safe_get(team1_data, 'team1_Yellow Cards'),
        'team2_Yellow': safe_get(team2_data, 'team2_Yellow Cards'),
        'team1_Yellow_competitor': safe_get(team1_data, 'team1_Yellow Cards  (competitor)'),
        'team2_Yellow_competitor': safe_get(team2_data, 'team2_Yellow Cards  (competitor)'),
        'team1_Red': safe_get(team1_data, 'team1_Red cards'),
        'team2_Red': safe_get(team2_data, 'team2_Red cards'),
        'team1_Red_competitor': safe_get(team1_data, 'team1_Red cards  (competitor)'),
        'team2_Red_competitor': safe_get(team2_data, 'team2_Red cards  (competitor)')
    }

    
    # Optionally print or log features for debugging
    # print(features_dict)
    prediction = full_match_prediction(
        features_dict,
        best_model_pipeline,
        xg_model_team1,
        xg_model_team2
    )

    prediction["h2h"] = h2h_summary
    prediction["Form"] = Form_summary
    return prediction
