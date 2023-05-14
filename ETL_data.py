import numpy as np
import pandas as pd

def ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl):
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

    # Drop column of dates
    #del fin_data_df['Dates']
    #del eco_data_df['Dates']
    #del ETF_data_df['Dates']

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

    # Separo set de entrenamiento y set de validación
    # Desde el 31.12.1999 hasta 31.12.2020 set de entrenamiento, y el resto validación

    n_days_train = 7672
    n_days_test =  855

    train_fin_data_df = fin_data_df.head(n_days_train).copy()
    test_fin_data_df = fin_data_df.tail(n_days_test).copy()
    test_fin_data_df = test_fin_data_df.reset_index(drop=True)

    train_eco_data_df = eco_data_df.head(n_days_train).copy()
    test_eco_data_df = eco_data_df.tail(n_days_test).copy()
    test_eco_data_df = test_eco_data_df.reset_index(drop=True)

    train_ETF_data_df = ETF_data_df.head(n_days_train).copy()
    test_ETF_data_df = ETF_data_df.tail(n_days_test).copy()
    test_ETF_data_df = test_ETF_data_df.reset_index(drop=True)

    train_data = { 'fin_data': train_fin_data_df, 'eco_data': train_eco_data_df, 'ETF_prices': train_ETF_data_df}
    test_data = { 'fin_data': test_fin_data_df, 'eco_data': test_eco_data_df, 'ETF_prices': test_ETF_data_df}
    return train_data, test_data
