import optuna
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def save_iteration_callback(study, trial, model_name, csv_filename="backup_results.csv"):
    trial_dict = trial.params.copy()
    trial_dict['Model'] = model_name
    trial_dict['Iteration'] = trial.number
    trial_dict['Score'] = trial.value
    
    df_trial = pd.DataFrame([trial_dict])
    
    if not os.path.isfile(csv_filename):
        df_trial.to_csv(csv_filename, index=False)
    else:
        df_trial.to_csv(csv_filename, mode='a', header=False, index=False)


def objective(trial, model_name, X, y):
    # Search spaces
    if model_name == 'Regressão Logística':
        C = trial.suggest_float('C', 1e-4, 10.0, log=True)
        model = LogisticRegression(C=C, class_weight='balanced', random_state=42, max_iter=1000)
        
    elif model_name == 'Random Forest':
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                       min_samples_split=min_samples_split, class_weight='balanced', random_state=42)
        
    elif model_name == 'KNN':
        n_neighbors = trial.suggest_int('n_neighbors', 3, 30)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
        
    elif model_name == 'Naive Bayes':
        var_smoothing = trial.suggest_float('var_smoothing', 1e-11, 1e-8, log=True)
        model = GaussianNB(var_smoothing=var_smoothing)
        
    elif model_name == 'Rede Neural (MLP)':
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True)
        activation = trial.suggest_categorical('activation', ['sigmoid', 'tanh'])
        model = MLPClassifier(alpha=alpha, learning_rate_init=learning_rate_init, activation=activation, max_iter=500, random_state=42)
        
    elif model_name == 'HistGradientBoost':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        model = HistGradientBoostingClassifier(learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        
    elif model_name == 'XGBoost':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, eval_metric='logloss', random_state=42)
        
    elif model_name == 'LightGBM':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        num_leaves = trial.suggest_int('num_leaves', 20, 100)
        model = LGBMClassifier(learning_rate=learning_rate, num_leaves=num_leaves, verbose=-1, random_state=42)

    # Cross validation 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    return scores.mean()

def run_bayesian_search(X, y, n_trials=30):
    modelos = [
        'Regressão Logística', 'Random Forest', 'KNN', 'Naive Bayes', 
        'Rede Neural (MLP)', 'HistGradientBoost', 'XGBoost', 'LightGBM'
    ]

    if os.path.exists("backup_results.csv"):
        os.remove("backup_results.csv")
        
    best_models = {}

    for model_name in modelos:
        
        study = optuna.create_study(direction='maximize', study_name=model_name)
        
        study.optimize(
            lambda trial: objective(trial, model_name, X, y), 
            n_trials=n_trials,
            callbacks=[lambda study, trial: save_iteration_callback(study, trial, model_name)]
        )
        
        best_models[model_name] = study.best_params
        print(f"Melhor Score ({model_name}): {study.best_value:.4f}")

    df_final = pd.read_csv("backup_results.csv")
    df_final.to_excel("resultados_bayesian_search.xlsx", index=False)
    
    return best_models  

