import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compara_modelos(fname):
    xls = pd.read_excel(fname, sheet_name='Comparativa',header=0, index_col=0)

    # Configurar el gráfico de columnas adyacentes
    fig, ax = plt.subplots()

    n = len(xls.index)
    x = np.arange(n)
    width = 0.20

    labels = xls.columns.tolist()

    for i,label in enumerate(labels):
        plt.bar(x + (i- 2) * width, xls[label], width=width, label=label)

    plt.ylabel("USD")
    plt.xticks(x, xls.index)
    plt.legend(loc='best')
    plt.savefig('./figures/comparativa_agentes.png', format='png')
    plt.show()

def compara_modelos_SP500(fname):
    xls = pd.read_excel(fname, sheet_name='ComparativaSP500',header=0, index_col=0)

    # Configurar el gráfico de columnas adyacentes
    fig, ax = plt.subplots()

    n = len(xls.index)
    x = np.arange(n)
    width = 1/6

    labels = xls.columns.tolist()

    for i,label in enumerate(labels):
        plt.bar(x + (i- 2) * width, xls[label], width=width, label=label)

    plt.ylabel("Rendimiento (%)")
    plt.xticks(x, xls.index)
    plt.legend(loc='best')
    plt.savefig('./figures/comparativa_SP500.png', format='png')
    plt.show()

def compara_modelos_SP500_full_invested(fname):
    pd_csv = pd.read_csv(fname, header=0)
    
    pd_csv = pd_csv.pivot(index='Periodo', columns='Modelo', values='Rendimiento')

    # Configurar el gráfico de columnas adyacentes
    fig, ax = plt.subplots()

    n = len(pd_csv.index)
    x = np.arange(n)
    width = 1/6

    labels = pd_csv.columns.tolist()

    for i,label in enumerate(labels):
        plt.bar(x + (i- 2) * width, pd_csv[label], width=width, label=label)

    plt.xticks(x, pd_csv.index)
    plt.ylabel("Rendimiento (%)")
    plt.legend(loc='best')
    plt.savefig('./figures/comparativa_SP500_full_invested.png', format='png')
    plt.show()

fname = './Tabla comparativas modelos v15-06-23.xlsx'
compara_modelos(fname)

fname = './Tabla comparativas modelos v15-06-23.xlsx'
compara_modelos_SP500(fname=fname)

fname = './results/resumen_resultados_portfolio_inicial.csv'
compara_modelos_SP500_full_invested(fname=fname)