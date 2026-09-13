# EV Charging Demand Forecast

A machine learning system that predicts electric vehicle charging pile usage rates from historical operational, environmental, and temporal data, with an interactive Streamlit dashboard for model comparison and real-time predictions.

**Course project, CS316: Introduction to AI**

## Overview

Public EV charging stations face unpredictable demand driven by time of day, weather, and traffic conditions. Poor forecasting leads to congestion, inefficient energy allocation, and long queues. This project builds a regression-based pipeline to predict charging pile usage rate from historical data, helping station operators plan energy distribution and reduce congestion.

## Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **XGBoost** ★ | 0.0565 | 0.0722 | 0.9903 |
| Random Forest | 0.073 | 0.0898 | 0.9849 |
| Decision Tree | 0.1023 | 0.132 | 0.9675 |
| Linear Regression | ~0.0000 | ~0.0000 | 1.0000 |

**Best model: XGBoost.** Linear Regression's near-perfect score is a red flag rather than a win, it points to the dataset's relationships being simple enough (or engineered features being derived closely enough from the target) that a linear model trivially fits them, rather than genuine predictive skill. This was excluded when identifying the best-performing model, in favor of XGBoost's ability to capture nonlinear relationships across complex features.

## Key Findings

1. **Charging duration is the strongest predictor** of usage rate (correlation 0.89), followed by energy consumed (0.63).
2. **Weather variables showed weak correlation** with charging behavior, contrary to initial assumptions.
3. **Traffic-related features showed moderate influence**, and demand peaks during specific hours (5AM-6PM) and increases on weekends.
4. **Engineered features added predictive strength**: duration per EV, energy per minute, and rolling usage rate all contributed useful signal beyond the raw columns.
5. **XGBoost outperformed all other models** due to its ability to model nonlinear relationships and handle a large feature set without overfitting as easily as a single decision tree.

## Approach

**Dataset:** [Charging Pile Demand Forecasting](https://www.kaggle.com) dataset from Kaggle, containing timestamp, charging duration, energy consumed, number of EVs charging, traffic flow, peak-hour indicators, vehicle type, and temperature.

**Preprocessing:** no missing values were found in this dataset. Irrelevant identifier columns were dropped, and timestamps were decomposed into hour-of-day and day-of-week features to capture temporal demand patterns. Numeric features were standardized (zero mean, unit variance) before training.

**Feature engineering:**
- **Duration per EV**: average charging time per vehicle
- **Energy per minute**: rate of energy consumption, indicating station efficiency
- **Rolling usage rate**: moving average that smooths short-term fluctuations and captures weekly/hourly patterns

**Models compared:** Linear Regression (baseline), Decision Tree, Random Forest, XGBoost, evaluated using MAE, RMSE, and R².

## Dashboard

An interactive Streamlit dashboard (`myapp.py`) provides three views:
- **Overview / EDA**: dataset preview, summary statistics, correlation heatmap, and feature distributions
- **Model Comparison**: trains all four models and displays MAE/RMSE/R² side by side, plus a residual plot for the best model
- **Make a Prediction**: lets a user input feature values and get a real-time usage rate prediction from any of the trained models

## Tech Stack

Python, scikit-learn, XGBoost, Streamlit, pandas, matplotlib, seaborn

## Project Structure

- `notebook/` — ev_charging_demand_forecast.ipynb
- `app/` — myapp.py
- `report/` — ev-charging-demand-forecast-report.pdf
- `presentation/` — ev-charging-demand-forecast-slides.pdf
- `README.md`

## Team

Built as a group project with Hala Abdullah for CS316 (Introduction to AI), Prince Sultan University.

## References

Full citation list available in the project report.
