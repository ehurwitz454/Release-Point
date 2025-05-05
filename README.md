# Release-Point
This repository analyzes pitcher release point consistency using 2021 MLB Statcast pitch-by-pitch data. The project investigates how release point variability relates to whiff rate, using detailed analyses across pitch types, pitch counts, and pitch-to-pitch transitions.

Project Goals: 

Quantify release point consistency using RMSE (Root Mean Squared Error).

Compare consistency across games, pitchers, and pitch types.

Analyze how release point changes affect whiff outcomes.

Run logistic regression to model the probability of a whiff based on pitch-to-pitch release point differences.

├── data_preprocessing.py         # Loads and cleans pitch-by-pitch Statcast data

├── analysis_functions.py         # Core functions for RMSE, segmenting, and comparisons

├── statistical_analysis.py       # Correlation and modeling tools

├── visualization.py              # Plotting release point consistency and whiff trends

├── main.py                       # Orchestrates the full pipeline

├── README.md

MLB Statcast Data: https://www.kaggle.com/datasets/s903124/mlb-statcast-data

Ethan Hurwitz & Matthew White

This project is licensed under the MIT License — see the LICENSE file for details.
