import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np    
    

def plot_result(fin_data_df, titulo, columna, y_label, idx, per_long=730):

    x = fin_data_df['Dates'][idx:idx+per_long]
    y = fin_data_df.iloc[idx:idx+per_long, columna]
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='Precio fin de día')
    plt.ylabel(y_label)
    plt.legend()
    plt.savefig("./figures/fig_S&P500_"+titulo)
    plt.show() 
    plt.close()
    
    
idx_l = [2927, 7310, 4388, 6580]
#test_name_l = ['2008-2009', '2020-2021', '2012-2013', '2018-2019']
test_name_l = ['Rendimiento 2008-2009', 'Rendimiento 2020-2021', 'Rendimiento 2012-2013', 'Rendimiento 2018-2019']
    
fin_data_f = ".\data\csv\Financial Data.csv"
fin_data_df = pd.read_csv(fin_data_f, parse_dates=['Dates'] ,usecols=lambda column: column == 'Dates' or column=='S&P 500')
print(fin_data_df.loc[2927]['S&P 500'])


for idx, test_name in zip(idx_l, test_name_l):
    print(fin_data_df.loc[idx]['S&P 500'])
    fin_data_df['Rendimiento S&P500'] = fin_data_df.loc[:]['S&P 500']/fin_data_df.loc[idx]['S&P 500']*100
    plot_result(fin_data_df, test_name, 2, 'Rendimiento base 100', idx)
