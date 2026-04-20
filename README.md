# Pre-Fault Detection in Photovoltaic Inverters

This repository contains the data processing pipelines and machine learning framework developed to classify pre-failure states in solar photovoltaic inverters. By analyzing multivariate time-series data (power, voltage, current, and internal temperatures), this project applies feature engineering, feature selection, and Bayesian hyperparameter optimization to train predictive models capable of identifying anomalies minutes before a critical equipment failure occurs.

The analytical pipeline evaluates a diverse stack of baseline and ensemble models, including:
* **Linear & Distance-Based:** Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM)
* **Probabilistic:** Naive Bayes
* **Neural Networks:** Multilayer Perceptron (MLP), Long Short-Term Memory (LSTM)
* **Tree-Based Ensembles:** Random Forest, Histogram-based Gradient Boosting, XGBoost, LightGBM