import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pathlib import Path

def get_baseline_models():
    return {
        'Regressão Logística': LogisticRegression(class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM (SVC)': SVC(class_weight='balanced', probability=True, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Rede Neural (MLP)': MLPClassifier(max_iter=500, random_state=42),
        'HistGradientBoost': HistGradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(verbose=-1, random_state=42)
    }

def run_training_pipeline(X_train, y_train, X_test, y_test, target_name, figures_dir=None):

    print(f"\nTraining for target: {target_name}")

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelos = get_baseline_models()
    resultados = []

    for nome, modelo in modelos.items():
        print(f"Training {nome}")
        inicio = time.time()
        
        modelo.fit(X_train_scaled, y_train)
        
        prev_test = modelo.predict(X_test_scaled)
        prob_test = modelo.predict_proba(X_test_scaled)[:, 1]
        
        tempo = time.time() - inicio

        resultados.append({
            'Modelo': nome,
            'Acurácia': accuracy_score(y_test, prev_test),
            'F1_Score': f1_score(y_test, prev_test),
            'Precisão': precision_score(y_test, prev_test, zero_division=0),
            'Recall': recall_score(y_test, prev_test),
            'AUC': roc_auc_score(y_test, prob_test),
            'Tempo (s)': tempo
        })

    df_resultados = pd.DataFrame(resultados).sort_values(by='F1_Score', ascending=False)
    return df_resultados