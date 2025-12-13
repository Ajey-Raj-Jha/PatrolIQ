# 🚓 PatrolIQ – Smart Safety Analytics Platform

PatrolIQ is an urban safety intelligence platform that leverages unsupervised machine learning to analyze crime patterns and optimize police resource allocation.  
The project is built using real-world crime data from the Chicago Police Department.

## 🔍 Problem Statement
Urban police departments struggle to efficiently allocate patrol resources due to the lack of actionable insights from massive crime datasets.  
This project analyzes over **500,000 crime records** to identify hotspots, temporal crime patterns, and risk zones.

## 🧠 Key Features
- Geographic crime hotspot detection using clustering algorithms
- Temporal crime pattern analysis
- Dimensionality reduction for simplified visualization
- MLflow-based experiment tracking
- Interactive Streamlit dashboards
- Cloud deployment on Streamlit Cloud

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Unsupervised Learning (KMeans, DBSCAN, Hierarchical)
- PCA, t-SNE / UMAP
- MLflow
- Streamlit
- Git & GitHub

## 📊 Dataset
- **Source:** Chicago Data Portal – Crimes 2001 to Present
- **Records Used:** 500,000 (sampled from 7.8M)
- **Features:** 22 crime, temporal, and geographic attributes
- **Crime Types:** 33 categories

## 📁 Project Structure
PatrolIQ/
├── data/
├── notebooks/
├── src/
├── streamlit_app/
├── mlruns/
├── requirements.txt
├── README.md
└── .gitignore