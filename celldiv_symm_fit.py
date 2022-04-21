import pandas as pd 
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import csv
import os
import matplotlib.pyplot as plt

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

def plot_data():
    return None


if __name__ == '__main__':
    _, _, x1interp, y1interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_1d.csv")
    _, _, x2interp, y2interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_2d.csv")
    _, _, x4interp, y4interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_4d.csv")
    pars = pd.read_csv(os.path.join(os.path.dirname(__file__),'data/celldiv_time_fit_parameters.csv'),header=0)
    pars.index = pars['descr']
    descr = pars['descr']
    pars.drop(axis=1,labels=['descr'],inplace=True)

    plot_data()
