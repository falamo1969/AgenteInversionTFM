import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
#import matplotlib


#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#import pickle

class FinancialEconomicEnv(gym.Env):
    """A trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}
    def __init__(self, 
                 fin_data_df,       # Contiene todos los datos financieros y económicos
                 precios_ETFs_df,   # Contiene los precios de los ETFs sobre los que C/V
                 initial_amount,    # Inversión cash inicial
                 volume_trade,      # Número de acciones fijo sobre el que se opera (C/V)
                 trading_cost,      # Coste de operación (C/V)
                 custody_cost,      # Coste anual de mantener las posiciones que hay en el portfolio
                 portfolio = [],    # Composición del portfolio (normalmente todos las participacioines en los ETFs a cero)
                 day=0):
        
        self.CUSTODY_PERIOD = 30 # Por ahora trato así los periodos de custodia hasta que me decida por el tratamiento de fechas

        self.fin_data_df = fin_data_df
        self.precios_ETFs_df = precios_ETFs_df
        self.initial_amount = initial_amount
        self.volume_trade = volume_trade
        if portfolio == []:
            self.initial_portfolio = [0 for i in range(self.precios_ETFs_df.shape[1])]
        else:
            self.initial_portfolio = portfolio
        
        self.trading_cost = trading_cost 
        self.custody_cost = custody_cost
        self.total_trading_cost = 0
        self.total_custody_cost = 0
        self.custody_cost_accrual = 0

        self.day0 = day        
        self.plus_minus = 0
        self.episode = 0
        
        # Defino action_space como un array del número de ETFs en los que se puede tomar posición con tres posibles valores:
        #       0 Hold
        #       1 Sell
        #       2 Buy

        actions = [3 for i in range(len(self.initial_portfolio))]
        self.action_space = spaces.MultiDiscrete(actions)
        
        print(self.fin_data_df.shape[1])

        # Defino observation_space como un diccionario para mayor facilidad ya que hay mezcla de datos
        self.observation_space = spaces.Dict({
                                'cash':         spaces.Box(low=0, high=np.inf),
                                'plus_minus':   spaces.Box(low=-self.initial_amount, high=np.inf),
                                'fin_data':     spaces.Box(low=-np.inf, high=np.inf, shape=(self.fin_data_df.shape[1],)),
                                'ETF_prices':   spaces.Box(low=0, high=np.inf, shape=(self.precios_ETFs_df.shape[1],)),
                                'portfolio':    spaces.Box(low=0, high=np.inf, shape=(len(self.initial_portfolio),), dtype=int)
                                })

        self.cash = initial_amount
        self.portfolio =  self.initial_portfolio
        self.day = day
        self.valor_portfolio = initial_amount + self._get_value_portfolio() # Valor actual del portfolio
        self.valor_inversion = self.valor_portfolio  # Valor de la inversión actualizada por la inflación (objetivo a batir)

        self.inflacion = self._get_inflation()

#        self.n_indicators = fin_data_df.shape[1]
#        self.n_ETFs = precios_ETFs_df.shape[1]

#        self.previous_state = []
#        self.data = self.fin_data_df.loc[self.day, :]


    def _get_obs(self):
        """ Estructura el estado"""
        # El estado se compone del diccionario de los valores y listas siguientes:
        #   
        # Datos financieros, económicos y sus MACD correspondientes
        # Precios de los ETFs
        # Composición del portfolio (# participaciones en cada ETF)

        state = {'cash':        self.cash,
                 'plus_minus':  self.plus_minus,
                 'fin_data':    self.fin_data_df.loc[self.day, :].values.tolist(),
                 'ETF_prices':  self.precios_ETFs_df.loc[self.day, :].values.tolist(),
                 'portfolio':   self.portfolio
                }
        return state

    def _get_value_portfolio(self):
        # retorna el valor de los ETFs
        return sum(x*y for x, y in zip(self.portfolio, self.precios_ETFs_df.loc[self.day, :]))
    
    def _get_inversion_actualizada(self):
        return self.valor_inversion * (1+self.inflacion/365)
    
    def _get_inflation(self):
        return self.fin_data_df.loc[self.day, 'Inflation']
    
    def _sell_ETFs(self, actions):
        # Vendo los ETF que se indican en actions como -1, en la cantidad volume_trade

        # Se aplica el coste de trading, tipicamente 25 bp
        actions_index = np.array(actions)
        for i in np.where(actions_index==1)[0]:
            # Comprobar si hay ETFs para vender
            if self.portfolio[i]>=self.volume_trade:
                trading_cost_step = -np.round(self.volume_trade*self.precios_ETFs_df.iloc[self.day,i] * self.trading_cost, 2)
                self.cash += np.round(self.volume_trade*self.precios_ETFs_df.iloc[self.day,i],2) + trading_cost_step
                self.total_trading_cost += trading_cost_step
                self.portfolio[i]-=self.volume_trade

    def _buy_ETFs(self, actions):
        # Compro los ETF que se indica en actions como 1, en la cantidad self.volume_trade
        # Se aplica el coste de trading, tipicamente 25 bp
        # Como puede no haber suficiente cash disponible, voy a comprar con un orden aleatorio hasta que no haya más cash
        actions_index = np.array(actions)
        actions_buy = np.where(actions_index==2)[0]
        for i in np.random.choice(actions_buy, len(actions_buy), replace =False):
            trading_cost_step = -np.round(self.volume_trade*self.precios_ETFs_df.iloc[self.day,i] * self.trading_cost, 2)
            coste = np.round(self.volume_trade * self.precios_ETFs_df.iloc[self.day, i],2) + trading_cost_step
            if self.cash >= coste:
                self.cash -= coste
                self.portfolio[i] += self.volume_trade
                self.total_trading_cost += trading_cost_step

    def _get_info(self):
        return {'cash': self.cash , 'porfolio_value': self.valor_portfolio, 'inversion': self.valor_inversion,
                'total_trading_cost': self.total_trading_cost, 'total_custody_cost': self.total_custody_cost}

    def step(self, actions):

        # Vendo ETFs según las acciones
        self._sell_ETFs(actions)

        # Compro ETFs según las acciones
        self._buy_ETFs(actions)

        # Calculo de gastos de custodia
        custody_cost_step = -self._get_value_portfolio() * self.custody_cost/365
        self.custody_cost_accrual += custody_cost_step

        if (self.episode % self.CUSTODY_PERIOD) ==0:
            # Aplico gastos de custodia
        
            while self.custody_cost_accrual > self.cash:
                # No hay suficiente cash hay que vender algún ETF -random-
                actions = np.random.choice(len(self.portfolio), 1)
                self._sell_ETFs(actions)
                # El ETF que se ha vendido no acarrea gastos de custodia en este step
                self.custody_cost_accrual -= custody_cost_step
                custody_cost_step = self._get_value_portfolio() * self.custody_cost/365
                self.custody_cost_accrual += custody_cost_step

            # Cargo el gasto de custodia en el cash
            self.cash -= self.custody_cost_accrual
            self.total_custody_cost += self.custody_cost_accrual
            self.custody_cost_accrual = 0
       
        # Valoro el portfolio de ETF y le sumo el cash no invertido
        valor_portfolio_new = self.cash +  self._get_value_portfolio()

        # Actualizo el valor de la inversión  por la inflación
        self.valor_inversion =  self._get_inversion_actualizada()

        # La recompensa del step es la diferencia de valoración entre el step anterior y éste
        reward = valor_portfolio_new - self.valor_portfolio
        self.plus_minus += reward

        self.valor_portfolio = valor_portfolio_new
        info = self._get_info()
        self.day += 1        

        if self.day >= len(self.fin_data_df):
            # Ya no hay más muestras para este episodio
            state = []
            done = True
        else:
            self.inflacion = self._get_inflation()
            state = self._get_obs()
            done = False
        return state, reward, done, info

    def reset(self):
        # Inicializo todos los datos con los iniciales
        self.day = self.day0
        self.portfolio = self.initial_portfolio
        self.cash = self.initial_amount
        self.valor_portfolio = self.cash + sum(x*y for x, y in zip(self.portfolio, self.precios_ETFs_df.loc[self.day,:])) # Valor actual del portfolio
        self.valor_inversion = self.valor_portfolio  # Valor de la inversión actualizada por la inflación (objetivo a batir)
        self.inflacion = self._get_inflation()
        self.plus_minus = 0
        self.episode = 0
        self.total_trading_cost = 0
        self.total_custody_cost = 0
        self.custody_cost_accrual = 0

        return self._get_obs(), self._get_info()