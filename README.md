# Win Probability Predictor

This repository contains a Python-based project for predicting the win probabilities of sports teams in various matchups. The primary tool is a Jupyter Notebook that implements machine learning models to calculate and compare probabilities for home and away teams, ultimately determining the predicted winner for each game.

## Features

- **Win Probability Prediction**: Calculates win probabilities for home and away teams using a trained machine learning model.
- **Dynamic Team Matchups**: Allows predictions for specific team matchups provided by the user.
- **Batch Predictions**: Supports predictions for multiple games happening on a given day.
- **Outcome Determination**: Identifies the predicted winner based on probabilities.

## Installation

To get started with this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sehej-Git/nba-win-predictions.git
   cd nba-win-predictions

2.	Set up the environment:
	•	Install Python 3.8 or later.
	•	Create a virtual environment and activate it:

python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`


3.	Install dependencies:

pip install -r requirements.txt


4.	Run the Notebook:
	•	Open the Jupyter Notebook to interact with the project: jupyter notebook



Usage

Predicting a Single Game

Use the determine_winner function to predict the winner of a single game:

home_team = 'SAC'
away_team = 'IND'
result = determine_winner(games, home_team, away_team, model, scaler)
print(f"Predicted Winner: {result['winner']}")

Predicting Multiple Games

To predict multiple games happening today, use:

home_teams = ['SAC', 'TOR', 'NOP']
away_teams = ['IND', 'HOU', 'DEN']
predicted_results = predict_games(home_teams, away_teams, games, model, scaler)
print(predicted_results)

Randomly Choosing Winners

You can simulate random winner selection:

winner = random_choice('Team A', 'Team B')
print(f"Randomly selected winner: {winner}")

Files
	•	win-probability.ipynb: Main Jupyter Notebook containing the implementation.
	•	README.md: Documentation for the repository.
	•	requirements.txt: List of required Python libraries.

Contributing

Contributions are welcome! To contribute:
	1.	Fork this repository.
	2.	Create a feature branch:

git checkout -b feature-name


	3.	Commit your changes:

git commit -m "Add feature-name"


	4.	Push to the branch:

git push origin feature-name


	5.	Create a pull request.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments
	•	Libraries Used: pandas, numpy, scikit-learn, matplotlib
	•	Inspiration: Sports data analysis and predictive modeling

You can copy this text directly into your `README.md` file for your repository.
