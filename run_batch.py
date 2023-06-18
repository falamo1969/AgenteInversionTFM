# Batch de pruebas

from datetime import datetime
from ETL_data import ETL_data_df

from AgenteA2C import AgenteA2CTrain, AgenteA2CTest, A2CTrainedAgentLoad
#from AgentePPO import AgentePPOTrain, AgentePPOTest, PPOTrainedAgentLoad


fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
idx_l = [2927, 7310, 4388, 6580]
test_name_l = ['2008-2009_v3M', '2020-2022_v3M', '2012-2014_v3M', '2018-2020_v3M']
verbose  = 1
TRAIN =  True
TEST = True

now = datetime.now()
print("Empezando batch.", now.strftime("%d-%b %H:%M:%S"))

for idx, test_name in (zip(idx_l, test_name_l)):

    train_data = {}
    test_data = {}

    train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)
    
    if TRAIN == True:
        #Entreno modelo PPO
#        now = datetime.now()
#        print("Entrenando modelo PPO_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
#        modelPPO = AgentePPOTrain(test_name, train_data, verbose = verbose)

        #Entreno modelo A2C
        now = datetime.now()
        print("Entrenando modelo A2C_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
        modelA2C = AgenteA2CTrain(test_name, train_data, total_timesteps=3000000, verbose = verbose)

    if TEST == True:
        #Test modelo PPO
#        now = datetime.now()
#        modelPPO = PPOTrainedAgentLoad(test_name)
#        print("Test modelo PPO_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
#        AgentePPOTest(modelPPO, test_data, test_name + now.strftime("-%d-%b_%H_%M_%S"), verbose = verbose)

        #Test modelo A2C
        now = datetime.now()
        modelA2C = A2CTrainedAgentLoad(test_name)
        print("Test modelo A2C_{}: {} ".format(test_name, now.strftime("%d-%b %H:%M:%S")))
        AgenteA2CTest(modelA2C, test_data, test_name + now.strftime("-%d-%b_%H_%M_%S"), verbose = verbose)
    
print("##############################################################")    
print("###################### FINALIZADO BATCH ######################")
print("##############################################################")    
