import pandas as pd
import numpy as np

def split_spatial_train_test(df128, df134, df148, cols_to_drop=['datetime', 'fault_code']):
    """Inverters 134 and 148 for training, 128 for testing. Drop specified columns."""
    df128 = df128.drop(columns=cols_to_drop, errors='ignore').copy()
    df134 = df134.drop(columns=cols_to_drop, errors='ignore').copy()
    df148 = df148.drop(columns=cols_to_drop, errors='ignore').copy()
    
    train = pd.concat([df134, df148], ignore_index=True)
    test = df128.copy()
    
    return train, test

def get_redundant_features(train_df, target_col, threshold=0.95):
    """ Find high correlation features in both classes (0 and 1) and return the list of features to remove."""
    features = [col for col in train_df.columns if 'pre_falha' not in col]
    
    treino_1 = train_df[train_df[target_col] == 1][features]
    treino_0 = train_df[train_df[target_col] == 0][features]

    corr_1 = treino_1.corr().abs()
    corr_0 = treino_0.corr().abs()

    upper_1 = corr_1.where(np.triu(np.ones(corr_1.shape), k=1).astype(bool))
    upper_0 = corr_0.where(np.triu(np.ones(corr_0.shape), k=1).astype(bool))

    col_remover_1 = set([col for col in upper_1.columns if any(upper_1[col] > threshold)])
    col_remover_0 = set([col for col in upper_0.columns if any(upper_0[col] > threshold)])

    # Has to be redundant in both classes
    colunas_definitivas = list(col_remover_0 & col_remover_1)
    
    return colunas_definitivas

def apply_feature_selection(train_df, test_df, target_col, threshold=0.95):
    """Selection pipeline. Returns cleaned train and test dataframes."""
    colunas_para_remover = get_redundant_features(train_df, target_col, threshold)
    
    train_clean = train_df.drop(columns=colunas_para_remover)
    test_clean = test_df.drop(columns=colunas_para_remover)
    
    print(f"[{target_col}] Features originais: {len(train_df.columns)}")
    print(f"[{target_col}] Features removidas: {len(colunas_para_remover)}")
    print(f"[{target_col}] Features mantidas: {len(train_clean.columns)}")
    
    return train_clean, test_clean