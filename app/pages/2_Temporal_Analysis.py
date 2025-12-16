import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("⏰ Temporal Crime Patterns")
df = pd.read_csv("data/processed/crimes_500k_features.csv").sample(
    n=20000, random_state=42
)

df["hour"] = df["hour"].astype(int)

hourly = df["hour"].value_counts().sort_index()

fig, ax = plt.subplots()
hourly.plot(kind="bar", ax=ax)
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Number of Crimes")
ax.set_title("Crimes by Hour")

st.pyplot(fig)
