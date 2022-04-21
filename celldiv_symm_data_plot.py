#plotting raw data against used interpolation in integro_diff_eq_adapted

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import os

def load_exp_set(path):
    
    X = []
    Y = []

    path = os.path.join(os.path.dirname(__file__),path)
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))
     
    #remove duplicates
    xd = np.diff(X) <= 0
    yd = np.diff(Y) == 0
    
    i = 0
    while i < len(xd):
        if xd[i] or yd[i]:
            # print("deleting #" + str(i) + ", x = " + str(X[i]) + " (xd=" + str(xd[i]) + ")" + ", y = " + str(Y[i]) + " (yd=" + str(yd[i]) + ")")
            del X[i]
            del Y[i]
            xd = np.delete(xd, i)
            yd = np.delete(yd, i)
        else:
            i = i + 1 
    
    Xp = np.linspace(15, 215, num=100, endpoint=True)
    Yp = interp1d(X, Y, kind="linear", fill_value=(0, 0), bounds_error=False)
    Yp = Yp(Xp) 
    
    return X, Y, Xp, Yp

def read_data(path):
    path = os.path.join(os.path.dirname(__file__),path)

    data = pd.read_csv(path,header=None,names=['A','p'])
    return data

def plot_data(raw,interpX,interpY):
    fig = plt.figure(figsize=(8,4))
    fig.suptitle('')
    plt.plot(raw['A'],raw['p'],color = 'black')
    plt.plot(interpX,interpY,color = 'red', linestyle = 'dashed')
    plt.show()

if __name__ == '__main__':
    path = 'data/Pul_2012_01_celldensitydistro_1d.csv'
    _,_,interpX,interpY = load_exp_set(path)
    raw = read_data(path)
    plot_data(raw,interpX,interpY)