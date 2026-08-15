# 🎬 Movie Rating Intelligence

An explainable machine learning project that predicts IMDb-style movie ratings from movie metadata and explains the factors influencing each prediction.

## Project Overview

Movie ratings are influenced by several interconnected factors such as genre, release year, duration, audience engagement, director, and cast.

This project goes beyond simply predicting a rating. It combines exploratory data analysis, regression modeling, error analysis, feature importance, and SHAP-based explainability to understand both **what the model predicts** and **why it makes that prediction**.

An interactive Streamlit application allows users to enter the characteristics of a movie and receive an estimated rating together with an explanation of the major contributing factors.

## Key Features

- Data cleaning and preprocessing
- Exploratory analysis of movie rating patterns
- Genre, director, actor, duration, year, and vote analysis
- Multi-label genre encoding
- Frequency encoding for directors and actors
- Linear Regression baseline
- Random Forest Regression
- Gradient Boosting Regression
- MAE, RMSE, and R² model comparison
- Prediction error analysis
- Model feature importance
- Global and local SHAP explanations
- Interactive Streamlit prediction interface
- Reusable trained model and preprocessing artifacts

## Machine Learning Models

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.9205 | 1.1816 | 0.2490 |
| Random Forest | 0.8096 | 1.0679 | 0.3868 |
| **Gradient Boosting** | **0.7894** | **1.0369** | **0.4212** |

Gradient Boosting achieved the strongest overall test performance and is used as the final prediction model.

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to interpret the trained model.

The project provides:

- Global SHAP feature importance
- SHAP summary analysis
- Individual movie prediction explanations
- Features pushing a prediction higher
- Features pushing a prediction lower

SHAP values explain the behavior of the trained model and should not be interpreted as causal effects on real IMDb ratings.

## Project Structure

```text
Movie-Rating-Intelligence/
│
├── app/
│   └── app.py
│
├── data/
├── models/
│   ├── movie_rating_model.pkl
│   └── preprocessing_artifacts.pkl
│
├── notebooks/
│   └── movie_rating_intelligence.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Run the Project

Install the required packages:

```bash
py -3.11 -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
py -3.11 -m streamlit run app/app.py
```

## Technologies Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, SHAP, Streamlit, Joblib and Jupyter Notebook.

## Disclaimer

The predicted rating is a machine-learning estimate based on patterns found in historical movie data. It is not an actual IMDb rating and should not be interpreted as a guaranteed rating for a movie.