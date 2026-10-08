
#1. Model Goal - Check if winning the toss helps a team win a match?

#2. Importing necessary libraries
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

#3. Setting DATA directory, FEATURES and TARGET variables
DATA = Path(__file__).parents[2] /"data"/"ipl_matches.csv"
FEATURES = ['venue', 'toss_decision']
TARGET = 'toss_winner_won'

#4. Defining Training function
def train_model():
    #4.1. Loading the dataset
    df = pd.read_csv(DATA).sort_values(by='date', ascending=True)
    
    #4.2. Splitting the dataset into past to train and recent to test the model
    past_df, recent_df = train_test_split(df, test_size=0.2, shuffle=False)

    #4.3. Defining the baseline
    baseline = DummyClassifier(strategy="most_frequent").fit(past_df[FEATURES], past_df[TARGET])

    #4.4. Defining the model
    model = make_pipeline(
        make_column_transformer(
            (OneHotEncoder(handle_unknown="ignore"), FEATURES)
        ),
        LogisticRegression(max_iter=1000)
    ).fit(past_df[FEATURES], past_df[TARGET])

    #4.5. Returning the model and baseline
    return {
        "model": model,
        "matches":df,
        "test_size": len(recent_df),
        "baseline_accuracy": baseline.score(recent_df[FEATURES], recent_df[TARGET]),
        "model_accuracy": model.score(recent_df[FEATURES], recent_df[TARGET])
    }