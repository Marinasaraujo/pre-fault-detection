import pandas as pd
from src.bayesian_search import run_bayesian_search
from src.config import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.data_processing.target_labeling import mark_critical_alarm, generate_labeled_dataset
from src.data_processing.data_processing import fill_missing_values, remove_night_period, preprocess_inverter_pipeline, handle_alarmes
from src.feature_selection import split_spatial_train_test, apply_feature_selection
from sklearn.utils import shuffle
from src.modeling import run_training_pipeline


def main():

    print("Loading datasets")
    
    inverter128 = pd.read_csv(RAW_DATA_DIR / 'inverter_128.csv').sort_values('datetime')
    inverter134 = pd.read_csv(RAW_DATA_DIR / 'inverter_134.csv').sort_values('datetime')
    inverter148 = pd.read_csv(RAW_DATA_DIR / 'inverter_148.csv').sort_values('datetime')
    solar_station = pd.read_csv(RAW_DATA_DIR / 'solarstation1.csv').sort_values('datetime')
    
    alarmes128 = pd.read_csv(RAW_DATA_DIR / 'alarmes_128.csv')
    alarmes134 = pd.read_csv(RAW_DATA_DIR / 'alarmes_134.csv')
    alarmes148 = pd.read_csv(RAW_DATA_DIR / 'alarmes_148.csv')


    print("Processing Solar Station data")
    solar_station = fill_missing_values(solar_station)
    solar_station = solar_station.dropna(axis=1, how='any')
    solar_station = remove_night_period(solar_station)
    solar_station = solar_station.drop(columns=['id_usina','Unnamed: 0'], errors='ignore')
    print(solar_station.columns)

    print("Processing Alarms")
    alarmes128 = handle_alarmes(alarmes128)
    alarmes134 = handle_alarmes(alarmes134)
    alarmes148 = handle_alarmes(alarmes148)

    print("Processing Inverter data")
    
    cols_to_drop = ['Unnamed: 0'] + [f'mppt_voltage_v_{i}' for i in range(2, 13)] + [f'mppt_current_a_{i}' for i in range(2, 13)] + ['work_state2', 'work_state3', 'tensao_bateria' ]
    
    merged128 = preprocess_inverter_pipeline(inverter128, solar_station, cols_to_drop)
    merged134 = preprocess_inverter_pipeline(inverter134, solar_station, cols_to_drop)
    merged148 = preprocess_inverter_pipeline(inverter148, solar_station, cols_to_drop)
    

    print("Target Labeling")
    lista_janelas = [30] # To try different time windows
    
    dados128 = generate_labeled_dataset(merged128, alarmes128, janelas=lista_janelas)
    dados134 = generate_labeled_dataset(merged134, alarmes134, janelas=lista_janelas)
    dados148 = generate_labeled_dataset(merged148, alarmes148, janelas=lista_janelas)
    
    print("Feature Selection ")
    train, test = split_spatial_train_test(dados128, dados134, dados148)
    
    targets = ['pre_falha_30min'] # can be used with different time windows if needed

    for target in targets:
        
        train_clean, test_clean = apply_feature_selection(train, test, target_col=target, threshold=0.95)
        
        colunas_salvar = [col for col in train_clean.columns if 'pre_falha' not in col] + [target]
        
        train_clean[colunas_salvar].to_csv(PROCESSED_DATA_DIR / f'train_limpo_{target}.csv', index=False)
        test_clean[colunas_salvar].to_csv(PROCESSED_DATA_DIR / f'test_limpo_{target}.csv', index=False)
        
    print("Pipeline completed")

    
    print("Models training and evaluation")

    train = shuffle(train, random_state=42).reset_index(drop=True)
    
    target = 'pre_falha_30min'
    
    # Isola o alvo de estudo
    X_train = train.drop(columns=[col for col in train.columns if 'pre_falha' in col])
    y_train = train[target]
    
    X_test = test.drop(columns=[col for col in test.columns if 'pre_falha' in col])
    y_test = test[target]
    

    graphs = BASE_DIR / "reports" / "figures"
    
    # Training pipeline (model training, evaluation, and report generation)
    traing_report = run_training_pipeline(X_train, y_train, X_test, y_test, target_name=target, figures_dir=graphs)
    
    best_parameters = run_bayesian_search(X_train, y_train, n_trials=50)



if __name__ == "__main__":
    main()