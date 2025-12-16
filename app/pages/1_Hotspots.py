import streamlit as st
import pandas as pd

st.title("📍 Crime Hotspot Identification")



df = pd.read_csv("data/processed/crimes_deploy_sample.csv")




st.write("### Sample of clustered crime data")
st.dataframe(df.head())

st.write("### Geographic Crime Hotspots")

st.map(
    df.rename(columns={"Latitude": "lat", "Longitude": "lon"})
)
