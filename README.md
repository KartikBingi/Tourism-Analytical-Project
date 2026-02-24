# Tourism-Analytical-Project
Tourism Market Insights
Tourism Experience Analytics Dashboard
An end-to-end Data Science application built with Python and Streamlit. This project provides interactive market insights, predicts traveler personas using XGBoost, and offers personalized attraction recommendations using Natural Language Processing (NLP) techniques.

Project Overview
This project processes real-world tourism data to solve two main problems:
Classification: Predicting the "Visit Mode" (Family, Couple, Solo, etc.) of a trip based on demographic and attraction metadata.
Recommendation: Suggesting new travel destinations based on the user's previous interests.

Key Features
Market Insights: Interactive bar charts showing top attraction categories and global visitor origins.
Trip Persona Classifier: A machine learning engine powered by XGBoost that analyzes travel parameters to predict group dynamics.
Attraction Recommender: Uses Cosine Similarity and CountVectorization to recommend places similar to those you’ve already visited.

Tech Stack
Frontend: Streamlit
Machine Learning: XGBoost
Data Analysis: Pandas, NumPy
NLP/Similarity: Scikit-Learn (CountVectorizer, Cosine Similarity)
Version Control: GitHub

Repository Structure
app.py: The main application script containing the UI and logic.
requirements.txt: Configuration file for Python dependencies (crucial for deployment).
tourism_master_data.csv: The processed dataset used by the application.
Tourism_project.ipynb: The research notebook containing the initial Exploratory Data Analysis (EDA).

Setup & Installation
If you want to run this project locally:

Clone the repository:

Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
Install requirements:
Bash
pip install -r requirements.txt
Launch the app:
Bash
streamlit run app.py

Author,
Kartik Bingi,
Data Science Enthusiast
