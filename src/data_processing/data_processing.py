import pandas as pd
import ast

def expand_list_column(df, col, n=12, prefix=None):
    if prefix is None:
        prefix = col

    # Convert string into list
    s = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    # Ensure size n
    s = s.apply(lambda x: (x + [0]*n)[:n])

    # Creates the dataframe
    new_cols = [f"{prefix}_{i+1}" for i in range(n)]
    expanded = pd.DataFrame(s.tolist(), index=df.index, columns=new_cols)

    return pd.concat([df.drop(columns=[col]), expanded], axis=1)


def fill_missing_values(df):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601') 
    df = df.drop_duplicates(subset='datetime', keep='first')
    df['datetime'] = df['datetime'].dt.round('5min')
    # Average of duplicates

    df = df.groupby('datetime').mean(numeric_only=True).reset_index()  # Categorical features removed
    df = df.set_index('datetime').sort_index()
    # Missing timestamps included
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='5min') 
    
    expected = len(full_index)
    total = len(df)
    gaps = expected - total
    
    print(f"Fail Report")
    print(f"Expected samples: {expected}")
    print(f"Actual samples: {total}")
    print(f"Gaps (missing timestamps): {gaps}")
    
    df = df.reindex(full_index)

    
    # Missing values handling
    for coluna in df.columns:

        is_nan = df[coluna].isna()

        # Single id for each consecutive block of data or NaNs
        gap_id = is_nan.ne(is_nan.shift()).cumsum()

        gap_size = is_nan.groupby(gap_id).transform('sum')

        mask_small_gap = is_nan & (gap_size <= 36)
        mask_big_gaps = is_nan & (gap_size > 36)
        
        # Rolling average from the last 7 days for the same timestamp
        time_average = df.groupby(df.index.time)[coluna].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        interpolado = df[coluna].interpolate(method='time')

        # Smaller gaps - interpolation
        df.loc[mask_small_gap, coluna] = interpolado[mask_small_gap]
        
        # Bigger gaps - time-based average
        df.loc[mask_big_gaps, coluna] = time_average[mask_big_gaps]

        df[coluna] = df[coluna].fillna(time_average)


    # For big gaps in the first seven days as there is no historical average yet
    df = df.bfill().ffill()

    df = df.reset_index().rename(columns={'index': 'datetime'})


    return df


def remove_night_period(df, col_data='datetime', start='05:00', end='18:59'):
    """Removes night period from the dataset"""
    df = df.set_index(col_data)
    df_filtered = df.between_time(start, end).reset_index()
    return df_filtered


def handle_alarmes(df):
    df = df.copy()
    df['start_date'] = pd.to_datetime(df['start_date'], format='ISO8601')
    df['end_date'] = pd.to_datetime(df['end_date'], format='ISO8601')

    df['start_date'] = df['start_date'].dt.round('5min')
    df['end_date']   = df['end_date'].dt.round('5min')

    cols_check = ['inversor', 'usina', 'codigo', 'severidade', 'descricao','start_date', 'end_date']
    df = df.drop_duplicates(subset=cols_check, keep='first')
    return df



def merge_with_solar(inv_df, solar_df):
    start = inv_df['datetime'].min()
    end = inv_df['datetime'].max()

    solar_cut = solar_df[(solar_df['datetime'] >= start) & (solar_df['datetime'] <= end)]

    merged = pd.merge(inv_df, solar_cut, on='datetime', how='left')
    return merged


def preprocess_inverter_pipeline(inverter_df, solar_df, colunas_para_remover):

    df = inverter_df.copy()
    
    df = expand_list_column(df, "mppt_voltage_v", n=12)
    df = expand_list_column(df, "mppt_current_a", n=12)

    df = df.drop(columns=colunas_para_remover, errors='ignore')
    
    df = fill_missing_values(df)
    
    df = remove_night_period(df)

    print(df.columns)
    
    df_final = merge_with_solar(df, solar_df)
    
    return df_final
