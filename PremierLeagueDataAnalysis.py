import streamlit as st
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Display the app name at the top
st.title("English Premier League Prediction AI")

# Check if the file exists before reading
file_path = "/Users/ngoubimaximilliandiamgha/Desktop/PythonProject14/results.csv"  # Update this path if needed
if not os.path.exists(file_path):
    st.error(f"Error: File '{file_path}' not found. Please check the path.")
else:
    # Load the dataset with proper encoding
    df = pd.read_csv(file_path, encoding='ISO-8859-1')

    # Display the first few rows
    st.write("### Dataset Preview")
    st.write(df.head())

    # Count duplicate rows
    duplicate_count = df.duplicated().sum()
    st.write(f"**Number of duplicate rows:** {duplicate_count}")

    # Display dataset statistics
    st.write("### Dataset Statistics")
    st.write(df.describe())

    # Count missing values per season
    missing_per_season = df.isnull().sum(axis=1).groupby(df['Season']).sum()
    # Count missing values per feature
    missing_per_feature = df.isnull().sum()

    # Plot missing values per season
    st.write("### Missing Values per Season")
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_per_season.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Missing Values per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Missing Values")
    st.pyplot(fig)

    # Plot missing values per feature
    st.write("### Missing Values per Feature")
    fig, ax = plt.subplots(figsize=(12, 6))
    missing_per_feature.plot(kind='barh', color='salmon', ax=ax)
    ax.set_title("Missing Values per Feature")
    ax.set_xlabel("Number of Missing Values")
    st.pyplot(fig)

    # List of numerical features to check for outliers
    numerical_features = [
        'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF',
        'HY', 'AY', 'HR', 'AR', 'HTHG', 'HTAG'
    ]

    # Plot boxplots for numerical features to detect outliers
    st.write("### Outliers in Numerical Features")
    fig, axes = plt.subplots(4, 4, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(numerical_features):
        sns.boxplot(x=df[feature], ax=axes[i], color='green')
        axes[i].set_title(f"Outliers in {feature}")
    plt.tight_layout()
    st.pyplot(fig)

    # Home vs Away Wins
    df['HomeWin'] = df['FTR'] == 'H'  # Home win
    df['AwayWin'] = df['FTR'] == 'A'  # Away win
    df['Draw'] = df['FTR'] == 'D'  # Draw

    # Group by season and count the occurrences of each result
    win_counts = df.groupby('Season')[['HomeWin', 'AwayWin', 'Draw']].sum()

    # Plot Home vs Away Wins per Season
    st.write("### Home vs Away Wins per Season")
    fig, ax = plt.subplots(figsize=(12, 6))
    win_counts.plot(kind='bar', color=['green', 'red', 'blue'], ax=ax)
    ax.set_title("Home vs Away Wins and Draws per Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Matches")
    st.pyplot(fig)

    # 1. Wins by Team at Home
    home_wins = df[df['FTR'] == 'H'].groupby('HomeTeam').size().sort_values(ascending=False)

    # Plot Home Wins by Team
    st.write("### Home Wins by Team")
    fig, ax = plt.subplots(figsize=(12, 6))
    home_wins.plot(kind='bar', color='green', ax=ax)
    ax.set_title("Home Wins by Team")
    ax.set_xlabel("Team")
    ax.set_ylabel("Number of Wins")
    st.pyplot(fig)

    # 2. Wins by Team Away
    away_wins = df[df['FTR'] == 'A'].groupby('AwayTeam').size().sort_values(ascending=False)

    # Plot Away Wins by Team
    st.write("### Away Wins by Team")
    fig, ax = plt.subplots(figsize=(12, 6))
    away_wins.plot(kind='bar', color='red', ax=ax)
    ax.set_title("Away Wins by Team")
    ax.set_xlabel("Team")
    ax.set_ylabel("Number of Wins")
    st.pyplot(fig)

    # 3. Total Wins by Team (Home + Away)
    home_wins_total = df[df['FTR'] == 'H'].groupby('HomeTeam').size()
    away_wins_total = df[df['FTR'] == 'A'].groupby('AwayTeam').size()

    # Combine home and away wins
    total_wins = home_wins_total.add(away_wins_total, fill_value=0).sort_values(ascending=False)

    # Plot Total Wins by Team
    st.write("### Total Wins by Team (Home + Away)")
    fig, ax = plt.subplots(figsize=(12, 6))
    total_wins.plot(kind='bar', color='purple', ax=ax)
    ax.set_title("Total Wins by Team (Home + Away)")
    ax.set_xlabel("Team")
    ax.set_ylabel("Number of Wins")
    st.pyplot(fig)

    # Model training for match prediction
    # Label encode the 'HomeTeam', 'AwayTeam', and 'FTR' columns
    label_encoder_home = LabelEncoder()
    label_encoder_away = LabelEncoder()

    # Fit the encoder on the 'HomeTeam' and 'AwayTeam' columns (both)
    df['HomeTeam'] = label_encoder_home.fit_transform(df['HomeTeam'])
    df['AwayTeam'] = label_encoder_away.fit_transform(df['AwayTeam'])

    # You can also encode the 'FTR' column as before
    label_encoder_ftr = LabelEncoder()
    df['FTR'] = label_encoder_ftr.fit_transform(df['FTR'])

    # Feature selection (same as before)
    features = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST']
    X = df[features]
    y = df['FTR']  # Target: Full-time result (Home win, Away win, or Draw)

    # Split dataset into training and testing sets (same as before)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a model (RandomForestClassifier) (same as before)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions (same as before)
    y_pred = model.predict(X_test)

    # Evaluate model performance (same as before)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred, target_names=label_encoder_ftr.classes_))

    # Streamlit user interface to make predictions for new data (fix label encoding issues)
    st.title("Premier League Match Outcome Prediction")

    # Convert team indices for prediction from label encoders
    home_team_names = label_encoder_home.inverse_transform(df['HomeTeam'].unique())
    away_team_names = label_encoder_away.inverse_transform(df['AwayTeam'].unique())

    # Streamlit input options for selecting home and away teams
    home_team = st.selectbox("Select Home Team", home_team_names)
    away_team = st.selectbox("Select Away Team", away_team_names)

    # Display the selected teams
    st.write(f"Selected Home Team: {home_team}")
    st.write(f"Selected Away Team: {away_team}")

    # Add prediction functionality
    # Get the index of the selected teams
    home_team_idx = label_encoder_home.transform([home_team])[0]
    away_team_idx = label_encoder_away.transform([away_team])[0]

    # Get additional features for the selected teams (e.g., FTHG, FTAG, etc.)
    # For this example, we'll take some random values as placeholders
    match_features = np.array(
        [[home_team_idx, away_team_idx, 1, 1, 10, 10, 5, 5]])  # Adjust this based on actual input data

    # Make prediction for the selected match
    prediction = model.predict(match_features)
    result = label_encoder_ftr.inverse_transform(prediction)

    # Display the predicted result
    st.write(f"Predicted Result: {result[0]}")
