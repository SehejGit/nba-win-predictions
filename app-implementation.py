import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from nba_api.stats.endpoints import leaguegamefinder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="NBA Win Probability Predictor",
    page_icon="🏀",
    layout="wide"
)

# App title and description
st.title("🏀 NBA Game Win Probability Predictor")
st.markdown("""
This app predicts the win probability for NBA games based on team performance metrics from their last 10 games.
The model analyzes statistics like points, field goal percentage, assists, and plus/minus to estimate each team's chances.
""")

# Define the list of NBA teams
NBA_TEAMS = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 
             'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 
             'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

TEAM_COLORS = {
    'ATL': '#E03A3E', 'BOS': '#007A33', 'BKN': '#000000', 'CHA': '#1D1160', 
    'CHI': '#CE1141', 'CLE': '#860038', 'DAL': '#00538C', 'DEN': '#0E2240', 
    'DET': '#C8102E', 'GSW': '#1D428A', 'HOU': '#CE1141', 'IND': '#002D62', 
    'LAC': '#C8102E', 'LAL': '#552583', 'MEM': '#5D76A9', 'MIA': '#98002E', 
    'MIL': '#00471B', 'MIN': '#0C2340', 'NOP': '#0C2340', 'NYK': '#006BB6', 
    'OKC': '#007AC1', 'ORL': '#0077C0', 'PHI': '#006BB6', 'PHX': '#1D1160', 
    'POR': '#E03A3E', 'SAC': '#5A2D81', 'SAS': '#C4CED4', 'TOR': '#CE1141', 
    'UTA': '#002B5C', 'WAS': '#002B5C'
}

TEAM_NAMES = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

# Cache the data fetching to prevent excessive API calls
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_nba_games():
    try:
        games = leaguegamefinder.LeagueGameFinder().get_data_frames()[0]
        games = games[games['TEAM_ABBREVIATION'].isin(NBA_TEAMS)]
        games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
        return games
    except Exception as e:
        st.error(f"Error fetching NBA data: {e}")
        # Return a sample empty dataframe with the required columns if API fails
        return pd.DataFrame(columns=['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 
                                     'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS',
                                     'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                     'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 
                                     'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS'])

def last_n_games(games_df, team_abbrev, n=10):
    """Get the last n games for a specific team."""
    team_games = games_df[games_df['TEAM_ABBREVIATION'] == team_abbrev]
    return team_games.nlargest(n, 'GAME_DATE')

def get_team_performance(games_df, team_abbrev, n=10):
    """Get performance stats for a team based on their last n games."""
    recent_games = last_n_games(games_df, team_abbrev, n)
    
    if recent_games.empty:
        return {
            'avg_plus_minus': 0,
            'avg_points': 0,
            'avg_points_allowed': 0,
            'avg_fg_pct': 0,
            'avg_fg3_pct': 0,
            'avg_ast': 0,
            'record': "0-0"
        }
    
    record = recent_games['WL'].value_counts().to_dict()
    wins = record.get('W', 0)
    losses = record.get('L', 0)

    stats = {
        'avg_plus_minus': round(recent_games['PLUS_MINUS'].mean(), 2),
        'avg_points': round(recent_games['PTS'].mean(), 2),
        'avg_points_allowed': round(recent_games['PTS'].mean() - recent_games['PLUS_MINUS'].mean(), 2),
        'avg_fg_pct': round(recent_games['FG_PCT'].mean() * 100, 2),
        'avg_fg3_pct': round(recent_games['FG3_PCT'].mean() * 100, 2),
        'avg_ast': round(recent_games['AST'].mean(), 2),
        'record': f"{wins}-{losses}"
    }

    return stats

def compare_teams(games_df, team1_abbrev, team2_abbrev, n=10):
    """Compare two teams based on their recent performance."""
    team1_stats = get_team_performance(games_df, team1_abbrev, n)
    team2_stats = get_team_performance(games_df, team2_abbrev, n)

    comparison = pd.DataFrame({
        f'{team1_abbrev}': [
            team1_stats['avg_plus_minus'],
            team1_stats['avg_points'],
            team1_stats['avg_points_allowed'],
            team1_stats['avg_fg_pct'],
            team1_stats['avg_fg3_pct'],
            team1_stats['avg_ast'],
            team1_stats['record']
        ],
        f'{team2_abbrev}': [
            team2_stats['avg_plus_minus'],
            team2_stats['avg_points'],
            team2_stats['avg_points_allowed'],
            team2_stats['avg_fg_pct'],
            team2_stats['avg_fg3_pct'],
            team2_stats['avg_ast'],
            team2_stats['record']
        ]
    }, index=['Plus/Minus', 'Points', 'Points Allowed', 'FG%', '3P%', 'Assists', 'Record'])
    
    return comparison

def prepare_ml_features(games_df, team1, team2, n_games=10):
    """Prepare features for ML model from team comparison stats."""
    team1_stats = get_team_performance(games_df, team1, n_games)
    team2_stats = get_team_performance(games_df, team2, n_games)
    
    features = pd.DataFrame({
        'pts_diff': [team1_stats['avg_points'] - team2_stats['avg_points']],
        'plus_minus_diff': [team1_stats['avg_plus_minus'] - team2_stats['avg_plus_minus']],
        'fg_pct_diff': [team1_stats['avg_fg_pct'] - team2_stats['avg_fg_pct']],
        'fg3_pct_diff': [team1_stats['avg_fg3_pct'] - team2_stats['avg_fg3_pct']],
        'ast_diff': [team1_stats['avg_ast'] - team2_stats['avg_ast']]
    })
    
    return features

def train_model(games_df, train_period_days=365):
    """Train ML model on historical matchups."""
    with st.spinner("Training prediction model..."):
        cutoff_date = games_df['GAME_DATE'].max() - pd.Timedelta(days=train_period_days)
        recent_games = games_df[games_df['GAME_DATE'] >= cutoff_date]
        
        features_list = []
        results = []
     
        unique_games = recent_games.drop_duplicates(subset=['GAME_ID'])
        
        for _, game in unique_games.iterrows():
            home_team = game['TEAM_ABBREVIATION']
            away_team_row = recent_games[
                (recent_games['GAME_ID'] == game['GAME_ID']) & 
                (recent_games['TEAM_ABBREVIATION'] != home_team)
            ]
            if away_team_row.empty:
                continue
            away_team = away_team_row['TEAM_ABBREVIATION'].iloc[0]

            historical_games = games_df[games_df['GAME_DATE'] < game['GAME_DATE']]
            if len(historical_games) < 10: 
                continue
                
            features = prepare_ml_features(historical_games, home_team, away_team)
            features_list.append(features)

            result = 1 if game['WL'] == 'W' else 0
            results.append(result)
        
        if not features_list:
            st.error("Not enough data to train the model")
            return None, None
            
        X = pd.concat(features_list)
        y = pd.Series(results)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        return model, scaler

def predict_game(games_df, team1, team2, model, scaler):
    """Predict outcome of game between two teams."""
    features = prepare_ml_features(games_df, team1, team2)
    features_scaled = scaler.transform(features)
    
    win_prob = model.predict_proba(features_scaled)[0][1]
    
    return win_prob

def predict_both_teams(games, team1, team2, model, scaler):
    """Predicts the win probabilities for both teams in a game."""
    prob_team1 = predict_game(games, team1, team2, model, scaler)
    prob_team2 = 1 - prob_team1

    return prob_team1 * 100, prob_team2 * 100

def determine_winner(games, team1, team2, model, scaler):
    """Determines the winner based on the win probabilities of two teams."""
    prob_team1, prob_team2 = predict_both_teams(games, team1, team2, model, scaler)
    
    if prob_team1 > prob_team2:
        winner = team1
    elif prob_team2 > prob_team1:
        winner = team2
    else:
        winner = np.random.choice([team1, team2])  # In case the probabilities are exactly equal

    return {
        "team1": team1,
        "team1_prob": prob_team1,
        "team2": team2,
        "team2_prob": prob_team2,
        "winner": winner
    }

def predict_games(home_teams, away_teams, games, model, scaler):
    """Predicts the winners for a list of home and away teams."""
    results = []

    for home, away in zip(home_teams, away_teams):
        prediction = determine_winner(games, home, away, model, scaler)
        
        results.append({
            "home_team": home,
            "home_prob": prediction["team1_prob"],
            "away_team": away,
            "away_prob": prediction["team2_prob"],
            "winner": prediction["winner"]
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def create_matchup_visualization(team1, team2, comparison_df, prediction_result):
    """Create a visual representation of the matchup and prediction."""
    col1, col2 = st.columns(2)
    
    # Extract probabilities
    team1_prob = prediction_result["team1_prob"]
    team2_prob = prediction_result["team2_prob"]
    
    # Display team names and probabilities
    with col1:
        st.markdown(f"### {TEAM_NAMES[team1]}")
        st.markdown(f"<h1 style='color:{TEAM_COLORS[team1]};'>{team1_prob:.1f}%</h1>", unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"### {TEAM_NAMES[team2]}")
        st.markdown(f"<h1 style='color:{TEAM_COLORS[team2]};'>{team2_prob:.1f}%</h1>", unsafe_allow_html=True)
    
    # Create a progress bar visualization of the probabilities
    fig = go.Figure()
    
    # Add the rectangular bars
    fig.add_trace(go.Bar(
        x=[team1_prob],
        y=["Win Probability"],
        orientation='h',
        name=team1,
        marker=dict(color=TEAM_COLORS[team1]),
        hoverinfo='text',
        hovertext=f"{team1}: {team1_prob:.1f}%",
        text=f"{team1}: {team1_prob:.1f}%",
        textposition='inside'
    ))
    
    fig.add_trace(go.Bar(
        x=[team2_prob],
        y=["Win Probability"],
        orientation='h',
        name=team2,
        marker=dict(color=TEAM_COLORS[team2]),
        hoverinfo='text',
        hovertext=f"{team2}: {team2_prob:.1f}%",
        text=f"{team2}: {team2_prob:.1f}%",
        textposition='inside'
    ))
    
    # Customize layout
    fig.update_layout(
        barmode='stack',
        height=150,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        title=f"Win Probability: {TEAM_NAMES[team1]} vs {TEAM_NAMES[team2]}"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the comparison table
    st.subheader("Team Comparison (Last 10 Games)")
    
    # Style the DataFrame
    def highlight_winner(s):
        if s.name in ['Plus/Minus', 'Points', 'FG%', '3P%', 'Assists']:
            max_val = pd.to_numeric(s).max()
            return ['background-color: rgba(144, 238, 144, 0.6)' if v == max_val else '' for v in s]
        elif s.name == 'Points Allowed':
            min_val = pd.to_numeric(s).min()
            return ['background-color: rgba(144, 238, 144, 0.6)' if v == min_val else '' for v in s]
        return ['' for _ in s]
    
    # Apply styling
    styled_comparison = comparison_df.style.apply(highlight_winner)
    
    # Display the styled table
    st.dataframe(styled_comparison, use_container_width=True)

    # Predicted winner callout
    winner = prediction_result["winner"]
    winner_prob = team1_prob if winner == team1 else team2_prob
    
    st.markdown(f"""
    <div style='background-color: rgba(144, 238, 144, 0.3); padding: 20px; border-radius: 10px; text-align: center;'>
        <h2>Predicted Winner: {TEAM_NAMES[winner]} ({winner_prob:.1f}%)</h2>
    </div>
    """, unsafe_allow_html=True)

def get_upcoming_games():
    """Get placeholder for upcoming games."""
    # In a real app, you would fetch this from an API or schedule
    # For now, we'll just use today and tomorrow's date as placeholders
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    upcoming_games = [
        {"date": today, "home": "MIL", "away": "PHX"},
        {"date": today, "home": "NYK", "away": "PHI"},
        {"date": today, "home": "ATL", "away": "POR"},
        {"date": today, "home": "MEM", "away": "GSW"},
        {"date": today, "home": "CHI", "away": "TOR"},
        {"date": today, "home": "SAS", "away": "ORL"},
        {"date": today, "home": "DEN", "away": "MIN"}
        # {"date": tomorrow, "home": "MEM", "away": "GSW"},
        # {"date": tomorrow, "home": "CHI", "away": "TOR"},
        # {"date": tomorrow, "home": "SAS", "away": "ORL"},
        # {"date": tomorrow, "home": "DEN", "away": "MIN"}
    ]
    
    return upcoming_games

def display_todays_games(games, model, scaler):
    """Display today's games predictions automatically."""
    st.header("Today's NBA Game Predictions")
    
    # Get upcoming games
    upcoming_games = get_upcoming_games()
    today = datetime.now().strftime("%Y-%m-%d")
    todays_games = [game for game in upcoming_games if game["date"] == today]
    
    if not todays_games:
        st.info("No games scheduled for today.")
        
        st.subheader("Enter Custom Games")
        
        num_games = st.number_input("Number of games", min_value=1, max_value=15, value=2)
        
        home_teams = []
        away_teams = []
        
        for i in range(num_games):
            col1, col2 = st.columns(2)
            with col1:
                home = st.selectbox(f"Home Team {i+1}", options=NBA_TEAMS, index=i % len(NBA_TEAMS), key=f"home_{i}")
                home_teams.append(home)
            
            with col2:
                away_options = [team for team in NBA_TEAMS if team != home]
                away = st.selectbox(f"Away Team {i+1}", options=away_options, index=i % len(away_options), key=f"away_{i}")
                away_teams.append(away)
        
        if st.button("Predict Custom Games"):
            with st.spinner("Analyzing matchups..."):
                predict_and_display_games(home_teams, away_teams, games, model, scaler)
    else:
        home_teams = [game["home"] for game in todays_games]
        away_teams = [game["away"] for game in todays_games]
        
        # Automatically predict and display without requiring button click
        with st.spinner("Analyzing today's matchups..."):
            predict_and_display_games(home_teams, away_teams, games, model, scaler)

def predict_and_display_games(home_teams, away_teams, games, model, scaler):
    """Helper function to predict and display game results."""
    predictions = predict_games(home_teams, away_teams, games, model, scaler)
    
    for i, row in predictions.iterrows():
        st.markdown("---")
        st.subheader(f"Game {i+1}: {TEAM_NAMES[row['home_team']]} vs {TEAM_NAMES[row['away_team']]}")
        
        # Get comparison for this matchup
        comparison = compare_teams(games, row['home_team'], row['away_team'])
        
        # Create prediction result dict for visualization
        prediction_result = {
            "team1": row['home_team'],
            "team1_prob": row['home_prob'],
            "team2": row['away_team'],
            "team2_prob": row['away_prob'],
            "winner": row['winner']
        }
        
        create_matchup_visualization(row['home_team'], row['away_team'], comparison, prediction_result)

def display_custom_matchup(games, model, scaler):
    """Display the custom matchup prediction tab."""
    st.header("Custom Matchup Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Home Team", options=NBA_TEAMS, index=1)
    
    with col2:
        # Filter out team1 from options for team2
        team2_options = [team for team in NBA_TEAMS if team != team1]
        team2 = st.selectbox("Select Away Team", options=team2_options, index=0)
    
    if st.button("Predict Winner"):
        with st.spinner("Analyzing matchup..."):
            # Get comparison
            comparison = compare_teams(games, team1, team2)
            
            # Get prediction
            prediction = determine_winner(games, team1, team2, model, scaler)
            
            # Display results
            create_matchup_visualization(team1, team2, comparison, prediction)

def display_team_analysis(games, model, scaler):
    """Display the team analysis tab."""
    st.header("Team Performance Analysis")
    
    team = st.selectbox("Select Team", options=NBA_TEAMS)
    
    if st.button("Analyze Team"):
        with st.spinner("Analyzing team performance..."):
            recent_games = last_n_games(games, team)
            
            if recent_games.empty:
                st.error(f"No recent games found for {team}.")
                return
            
            stats = get_team_performance(games, team)
            
            # Create a team stats card
            st.subheader(f"{TEAM_NAMES[team]} Recent Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Record (Last 10)", stats['record'])
            
            with col2:
                st.metric("Avg Points", f"{stats['avg_points']:.1f}")
            
            with col3:
                st.metric("Avg Points Allowed", f"{stats['avg_points_allowed']:.1f}")
            
            with col4:
                st.metric("Plus/Minus", f"{stats['avg_plus_minus']:.1f}")
            
            # Additional stats row
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("FG%", f"{stats['avg_fg_pct']:.1f}%")
            
            with col2:
                st.metric("3P%", f"{stats['avg_fg3_pct']:.1f}%")
            
            with col3:
                st.metric("Assists", f"{stats['avg_ast']:.1f}")
            
            # Show recent games
            st.subheader("Recent Games")
            
            # Convert matchup and date for better display
            display_games = recent_games.copy()
            display_games['GAME_DATE'] = display_games['GAME_DATE'].dt.strftime('%Y-%m-%d')
            
            # Select columns to display
            columns_to_show = ['GAME_DATE', 'MATCHUP', 'WL', 'PTS', 'FG_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV', 'PLUS_MINUS']
            
            # Apply styling
            def highlight_wins(row):
                return ['background-color: rgba(144, 238, 144, 0.3)' if row['WL'] == 'W' else 'background-color: rgba(255, 182, 193, 0.3)' for _ in row]
            
            styled_games = display_games[columns_to_show].style.apply(highlight_wins, axis=1)
            
            st.dataframe(styled_games, use_container_width=True)
            
            # Show win probability against each team
            st.subheader(f"{team} Win Probability Against All Teams")
            
            # Calculate win probability against all other teams
            other_teams = [t for t in NBA_TEAMS if t != team]
            probabilities = []
            
            for other_team in other_teams:
                prob = predict_game(games, team, other_team, model, scaler) * 100
                probabilities.append({
                    'opponent': other_team,
                    'opponent_name': TEAM_NAMES[other_team],
                    'win_probability': prob
                })
            
            prob_df = pd.DataFrame(probabilities)
            
            # Sort by win probability
            prob_df = prob_df.sort_values('win_probability', ascending=False)
            
            # Create bar chart
            fig = px.bar(
                prob_df, 
                x='opponent', 
                y='win_probability',
                color='win_probability',
                color_continuous_scale=['red', 'yellow', 'green'],
                range_color=[0, 100],
                labels={'opponent': 'Opponent', 'win_probability': 'Win Probability (%)'},
                title=f"{TEAM_NAMES[team]} Win Probability Against Each Team",
                hover_data=['opponent_name', 'win_probability']
            )
            
            fig.update_layout(xaxis_title="Opponent", yaxis_title="Win Probability (%)")
            
            st.plotly_chart(fig, use_container_width=True)

# Main app function
def main():
    # Sidebar - change the order of options to make Today's Games first
    st.sidebar.title("NBA Win Predictor")
    
    app_mode = st.sidebar.radio(
        "Select Mode",
        ["Today's Games", "Custom Matchup", "Team Analysis"]
    )
    
    # Load NBA data
    games = fetch_nba_games()
    
    if games.empty:
        st.error("Unable to load NBA data. Please try again later.")
        return
    
    # Train the model
    model, scaler = train_model(games)
    
    if model is None or scaler is None:
        st.error("Unable to train the prediction model. Please try again later.")
        return

    # Display the selected tab
    if app_mode == "Today's Games":
        display_todays_games(games, model, scaler)
    elif app_mode == "Custom Matchup":
        display_custom_matchup(games, model, scaler)
    elif app_mode == "Team Analysis":
        display_team_analysis(games, model, scaler)

if __name__ == "__main__":
    main()