# Batch de pruebas

from datetime import datetime
from ETL_data import ETL_data_df

from AgenteA2CLR import AgenteA2CLRTrain, AgenteA2CTest


fin_data_fl = ".\data\csv\Financial Data.csv"
eco_data_fl = ".\data\csv\Economic indicators.csv"
ETF_data_fl = ".\data\csv\ETF-prices.csv"

test_size = 730 # 2 años
# Vamos a testear 2 periodos raros, 2008-2009, 2020-2022, y dos periodos normales 2012-2014 y 2018-2020
idx_l = [2927, 7310, 4388, 6580]
test_name_l = ['2008-2009_lr', '2020-2022_lr', '2012-2014_lr', '2018-2020_lr']
l_rate_l = [0.0005, 0.001]
verbose  = 1

now = datetime.now()
print("Empezando batch.", now.strftime("%d-%b %H:%M:%S"))

for idx, test_name in (zip(idx_l, test_name_l)):
    train_data = {}
    test_data = {}

    train_data, test_data = ETL_data_df(fin_data_fl, eco_data_fl, ETF_data_fl, test_size, idx)

    for l_r in l_rate_l:
        #Entreno modelo A2C
        now = datetime.now()
        print("Entrenando modelo A2C_{}_{:.4f}:: {} ".format(test_name, l_r, now.strftime("%d-%b %H:%M:%S")))
        modelA2C = AgenteA2CLRTrain(test_name + "_{:.4f}".format(l_r), train_data, l_r=l_r, verbose = verbose)

        #Test modelo A2C
        now = datetime.now()
        print("Test modelo A2C_{}_{:.4f}: {} ".format(test_name, l_r, now.strftime("%d-%b %H:%M:%S")))
        AgenteA2CTest(modelA2C, test_data, test_name + "_{:.4f}".format(l_r), verbose = verbose)
    
print("##############################################################")    
print("###################### FINALIZADO BACHT ######################")
print("##############################################################")    
