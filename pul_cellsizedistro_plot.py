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
    fig = plt.figure()
    fig.suptitle('Zellgrößenverteilung Fig 4D') 
    plt.xlabel = 'A [um^2]'
    plt.ylabel = 'p(A)'
    for i in range(0,int(data.shape[1]/2)):
        plt.plot(data[f'A_{i+1}'],data[f'p_{i+1}'],label=f'{i}')
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(__file__),'figures/cellsize_distro_plot.png'))

if __name__ == '__main__':
    data = read_data(1,11)
    plot_data(data)


# -> Tag 1(0),2(1) und 4(3) 