# ⚽ English Premier League Prediction AI

## Overview
The English Premier League Prediction AI is a Streamlit-based web application that leverages machine learning techniques to predict football match outcomes using historical Premier League data. It provides insightful data visualizations, performs exploratory data analysis (EDA), and predicts match results using a trained Random Forest classifier.

## Features

- **Data Loading & Validation:**
  - Ensures accurate file path specification.
  - Handles file existence checks and encoding compatibility.

- **Exploratory Data Analysis (EDA):**
  - Dataset previews and statistical summaries.
  - Identification of duplicate and missing data.
  - Visualizations of missing values per season and per feature.
  - Detection of outliers using boxplots for numerical features.

- **Match Outcome Insights:**
  - Comparison of home wins, away wins, and draws per season.
  - Team performance metrics including home wins, away wins, and overall total wins.

- **Prediction Model:**
  - Machine learning-based predictions using RandomForestClassifier.
  - Label encoding for categorical variables.
  - Clear model performance metrics including accuracy and classification reports.

- **Interactive Match Prediction:**
  - User-friendly interface to select teams for prediction.
  - Predicts outcomes of future matches based on historical data.

## Installation

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn
```

## Usage

Run the application:

```bash
streamlit run your_script.py
```

- Ensure the dataset file (`results.csv`) is correctly specified and available.
- Use the app to explore data insights and predict match outcomes interactively.

## Required Dataset

- `results.csv`: Historical English Premier League match data with columns like 'Season', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', and other relevant statistics.

## File Structure

- `your_script.py`: Main Streamlit application script.
- `results.csv`: Dataset file.

## Requirements

- Python >= 3.7
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Author
Developed by Ngoubi Maximillian Diangha.

## License
MIT License
