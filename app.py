#1. Importing necessary libraries
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
from smoke_test.model import train_model

#2. Plumbing - connecting the hosted app with the src directory
sys.path.insert(0, str(Path(__file__).parent/"src"))


#3. Streamlit App
st.title("IPL - Toss Win decides Match win?")
st.caption("A simple model to check if winning the toss helps a team win a match.")

result=st.cache_resource(train_model)()
matches = result["matches"]

left_column, right_column = st.columns(2)
left_column.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.0%}")
right_column.metric("Model Accuracy",f"{result['model_accuracy']:.0%}")
st.write(
    f"Trained on the oldest 80% and tested on the most recent 20% of {len(matches)} IPL matches"
)

st.subheader("Try a match prediction")
venue =st.selectbox("Venue", matches["venue"].unique())
decision = st.selectbox("Toss Winner chose to", matches["toss_decision"].unique())

row = pd.DataFrame([{"venue": venue, "toss_decision": decision}])
chance = result["model"].predict_proba(row)[0][1]
st.write(f"Chance of Toss Winner winning the match: {chance:.0%}")