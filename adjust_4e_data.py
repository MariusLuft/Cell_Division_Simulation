import pandas as pd
import numpy as np
import os

dir = os.path.dirname(__file__)

black = pd.read_csv(os.path.join(dir,'data\\Pul_2012_01_divisiontime_black.csv'),header=None)
blue = pd.read_csv(os.path.join(dir,'data\\Pul_2012_01_divisiontime_blue.csv'),header=None)

black[1] = black[1].apply(lambda x: x * np.log(2))
blue[1] = blue[1].apply(lambda x: x * np.log(2))

black.sort_values(by=0,axis=0,inplace=True)
blue.sort_values(by=0,axis=0,inplace=True)

black.to_csv(os.path.join(dir,'data/4e_black_adapted.csv'),header=['a','gamma'],index=False)
blue.to_csv(os.path.join(dir,'data/4e_blue_adapted.csv'),header=['a','gamma'],index=False)