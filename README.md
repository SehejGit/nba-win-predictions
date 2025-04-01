# NBA Game Win Probability Predictor 🏀

![NBA Win Predictor](https://img.shields.io/badge/NBA-Win%20Predictor-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0%2B-red)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-yellow)

## Overview

The NBA Game Win Probability Predictor is an interactive web application that uses machine learning to forecast the outcome of NBA games based on teams' recent performance statistics. By analyzing metrics from the last 10 games, such as points scored, field goal percentage, assists, and plus/minus ratings, the model calculates win probabilities for any NBA matchup.

## Features

- **Today's Games**: Automatically displays win predictions for all NBA games scheduled for the current day
- **Custom Matchup**: Create hypothetical matchups between any two NBA teams
- **Team Analysis**: Deep dive into a specific team's recent performance and predicted win rates against all other teams
- **Real-time Data**: Leverages the NBA API to pull the latest game statistics for accurate predictions
- **Interactive Visualizations**: Colorful and intuitive charts show win probabilities and team comparisons
- **Comprehensive Stats**: Displays key performance metrics including points, FG%, 3P%, and plus/minus

## How It Works

The application utilizes a Random Forest machine learning model trained on historical NBA game data. The prediction process involves:

1. **Data Collection**: Fetching recent game data for all NBA teams using the NBA API
2. **Feature Engineering**: Calculating team performance metrics based on the last 10 games
3. **Model Training**: Using a Random Forest Classifier to learn from historical matchups
4. **Prediction**: Applying the trained model to estimate win probabilities for upcoming games

## Development

The prediction model was developed in the `win-probability.ipynb` notebook, which contains:
- Data exploration and preprocessing steps
- Feature engineering techniques
- Model selection and training process
- Validation and performance evaluation

## Installation

1. Clone this repository
```bash
git clone https://github.com/SehejGit/nba-win-predictions.git
cd nba-win-predictions
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app
```bash
streamlit run app-implementation.py
```

## Requirements

The application requires the following packages:
- streamlit
- pandas
- numpy
- scikit-learn
- nba_api
- plotly
- datetime
- matplotlib

A complete list with version specifications is available in the `requirements.txt` file.

## Future Improvements

- Add player-specific metrics to improve prediction accuracy
- Incorporate injury reports and lineup changes
- Implement betting odds comparison
- Expand to historical season simulation
- Add playoff-specific prediction models

## Acknowledgments

- NBA API for providing access to comprehensive NBA statistics
- Streamlit for the interactive web application framework
- The scikit-learn team for the machine learning tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

[Sehej Singh](https://github.com/SehejGit)

---

Feel free to star ⭐ this repository if you find it useful!
