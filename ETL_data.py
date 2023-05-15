import numpy as np
import pandas as pd

def readcsv_and_calcMACD(fin_data_fl, eco_data_fl, ETF_data_fl):
    # MACD Calculation
    def calcMACD (df1, col_name, n_fast=12, n_slow=26, df2=None):
        data = df1[col_name]
        
        fastEMA = data.ewm(span=n_fast, min_periods=n_slow).mean()
        slowEMA = data.ewm(span=n_slow, min_periods=n_slow).mean()
        MACD = pd.Series(fastEMA-slowEMA, name = 'MACD_' + col_name)
        MACD = MACD.fillna(0)
        if df2 is None:
            df = df1.join(MACD)
        else:
            df = df2.join(MACD)
        
        return df

    # Read data files
    fin_data_df = pd.read_csv(fin_data_fl, dtype=float, usecols=lambda column: column != 'Dates')
    eco_data_df = pd.read_csv(eco_data_fl, dtype=float, usecols=lambda column: column != 'Dates')
    ETF_data_df = pd.read_csv(ETF_data_fl, dtype=float, usecols=lambda column: column != 'Dates')

    # Calculo MACD en las columnas que me interesan. Sobre todo en los datos financieros y de los ETFs
    fin_data_df = calcMACD(fin_data_df, 'Gold')
    fin_data_df = calcMACD(fin_data_df, 'S&P 500')
    fin_data_df = calcMACD(fin_data_df, 'USDX')
    fin_data_df = calcMACD(fin_data_df, 'UST 10Y')
    fin_data_df = calcMACD(fin_data_df, 'UST 2Y')
    fin_data_df = calcMACD(fin_data_df, 'VIX')
    fin_data_df = calcMACD(fin_data_df, 'BBG commodity index')
    fin_data_df = calcMACD(fin_data_df, 'Oil')
    fin_data_df = calcMACD(fin_data_df, 'Financial Conditions')

    for name_column in ETF_data_df.columns:
        fin_data_df = calcMACD(ETF_data_df, name_column, df2=fin_data_df)

    return fin_data_df, eco_data_df, ETF_data_df

def split_train_test(df, test_size, idx_test=0, reset_idx=True):
    # Separo set de entrenamiento y set de test
    test = df.iloc[idx_test:idx_test+test_size, :]
    train = pd.concat([df.iloc[0:idx_test, :], df.iloc[idx_test+test_size:df.shape[0], :]])
    if reset_idx:
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
    return train, test

def ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx_test=0):
    # Separo set de entrenamiento y set de validación
    # Tamaño del test es test_size
    # Empieza desde la posición idx_test

    fin_data_df, eco_data_df, ETF_data_df = readcsv_and_calcMACD(fin_data_fl, eco_data_fl, ETF_data_fl)


    train_fin_data_df, test_fin_data_df = split_train_test(fin_data_df, test_size, idx_test)
    train_eco_data_df, test_eco_data_df = split_train_test(eco_data_df, test_size, idx_test)
    train_ETF_data_df, test_ETF_data_df = split_train_test(ETF_data_df, test_size, idx_test)

    train_data = { 'fin_data': train_fin_data_df, 'eco_data': train_eco_data_df, 'ETF_prices': train_ETF_data_df}
    test_data = { 'fin_data': test_fin_data_df, 'eco_data': test_eco_data_df, 'ETF_prices': test_ETF_data_df}

    return train_data, test_data