import pandas as pd

def mark_critical_alarm(dados_df, alarmes_df, nome_coluna='alarme_critico'):
    dados_df = dados_df.copy()
    dados_df[nome_coluna] = 0 

    start = dados_df['datetime'].min()
    end = dados_df['datetime'].max()

    alarmes_df = alarmes_df.copy()
    # Filtering critical alarms that overlap with the data period
    critical_alarm = alarmes_df[(alarmes_df['severidade'] == 'critico') & (alarmes_df['end_date'] >= start) & (alarmes_df['start_date'] <= end)]

    for _, alarme in critical_alarm.iterrows(): # Iterando nas linhas
        inicio_falha = alarme['start_date']
        fim_falha = alarme['end_date']
        mask = (dados_df['datetime'] >= inicio_falha) & (dados_df['datetime'] <= fim_falha)
        dados_df.loc[mask, nome_coluna] = 1

    return dados_df


# New column of ones in the pre-failure window, zeros otherwise

def generate_labeled_dataset(dados_df, alarmes_df, janelas=[30], prefixo='pre_falha'):
    """
    janelas: integers list, time windows in minutes to label as pre-failure (e.g., [15, 30, 60])
    """
    dados_df = dados_df.copy()
    alarmes_df = alarmes_df.copy()

    for minutos in janelas:
        col_name = f"{prefixo}_{minutos}min"
        dados_df[col_name] = 0

    start = dados_df['datetime'].min()
    end = dados_df['datetime'].max()

    # Filter critical alarms that overlap with the data period
    alarmes_criticos = alarmes_df[
        (alarmes_df['severidade'] == 'critico') & 
        (alarmes_df['end_date'] >= start) & 
        (alarmes_df['start_date'] <= end)
    ]

    for _, alarme in alarmes_criticos.iterrows():
        failure_begin = alarme['start_date']

        # For each alarm, we fill all the requested windows
        for minutos in janelas:
            col_name = f"{prefixo}_{minutos}min"
            window_begin = failure_begin - pd.Timedelta(minutes=minutos)
            mask = (dados_df['datetime'] >= window_begin) & (dados_df['datetime'] < failure_begin)
            dados_df.loc[mask, col_name] = 1

    return dados_df
