import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 

data_dir = os.path.join(os.path.dirname(__file__),'data')
base_filename = 'Pul_2012_01_celldensitydistro'

def read_data(start_day = 1 ,end_day = 16):
    data = pd.DataFrame()
    for i in range(start_day,end_day):
        if i == 13 or i == 14:
            continue
        data = data.append(pd.read_csv(os.path.join(data_dir,base_filename+f'_{i}d.csv'),names=[f'A_{i}',f'p_{i}'],header=None))
    return data

def plot_data(data):
    plt.figure(figsize=(8,4))
    for i in range(0,int(data.shape[1]/2)):
        plt.plot(data[f'A_{i+1}'],data[f'p_{i+1}'],label=f'{i}')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = read_data(1,11)
    plot_data(data)