import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os

dir = os.path.dirname(__file__)
#implementation of eq. 2 from main paper
#x values here are the a values 
def gamma(x,y0,a0,m):
    return y0 * np.exp(-1 * np.power(a0/x,m))

def hill(x,y0,a0,m):
    return y0 * np.power(x,m) / (np.power(x,m) + np.power(a0,m)) 

#read in the adapted data points from fig 4e
def read_data():
    black = pd.read_csv(os.path.join(dir,'data/4e_black_adapted.csv'),header=0)    
    blue = pd.read_csv(os.path.join(dir,'data/4e_blue_adapted.csv'),header=0)    
    data = black.append(blue)

    return data,black,blue

#fits data onto gamma function given the x and y values of the observed data
def fit_data(x,y):
    opt_pars,cov = curve_fit(gamma,x,y)
    return opt_pars,cov

def fit_hill(x,y):
    opt_pars,_ = curve_fit(hill,x,y)
    return opt_pars

#calculates data for given pars using gamma function
def calc_data(a_min,a_max,pars):
    a = np.linspace(a_min,a_max,30)
    data = pd.DataFrame(data=a,columns=['a'],index=None)
    gamma_vals = []
    for x in a:
        gamma_vals.append(gamma(x,pars[0],pars[1],pars[2]))
    data['gamma'] = gamma_vals
    return data

def calc_hill(a_min,a_max,pars):
    a = np.linspace(a_min,a_max,30)
    data = pd.DataFrame(data=a,columns=['a'],index=None)
    gamma_vals = []
    for x in a:
        gamma_vals.append(hill(x,pars[0],pars[1],pars[2]))
    data['gamma'] = gamma_vals
    return data

#plots the data into a single plot
def plot_data(ref_black,ref_blue,fit_data,fit_pos_std,fit_neg_std,hill_data,minimized_data=None):
    fig = plt.figure(figsize=(8,6))
    fig.suptitle('Fitting Zellteilungsrate', fontsize=30)
    axes = fig.gca()
    axes.tick_params(axis='both',which='major',labelsize = 15)
    plt.xlabel('Zellgröße A [um^2]', fontsize=20)
    plt.ylabel('Zellteilungsrate y [d^-1]', fontsize=20)
    #plt.ylim((0,3.5))
    plt.plot(ref_black['a'],ref_black['gamma'],'o',color = 'black', label='Ref Schwarz')
    plt.plot(ref_blue['a'],ref_blue['gamma'],'o',color = 'blue', label='Ref Blau')
    plt.plot(fit_data['a'],fit_data['gamma'],color = 'red', label='Fit')
    if minimized_data is not None:
        plt.plot(minimized_data['a'],minimized_data['gamma'],color = 'Yellow', label='Optimized Fit')
    plt.plot(fit_pos_std['a'],fit_pos_std['gamma'],color = 'red', label='Fit + Std',linestyle='dashed',alpha=0.5)
    plt.plot(fit_neg_std['a'],fit_neg_std['gamma'],color = 'red', label='Fit - Std',linestyle='dashed',alpha=0.5)
    plt.plot(hill_data['a'],hill_data['gamma'],color = 'green', label='Hill',linestyle='dashed')
    plt.grid(visible=True)
    plt.legend()
    plt.show()
    fig.savefig(os.path.join(os.path.dirname(__file__),'figures/celldiv_time_fitted.png'))

#prints the given parameters
def print_pars(pars,std):
    print(f'y0: {pars[0]} +/- {std[0]}, a0: {pars[1]} +/- {std[1]}, m: {pars[2]} +/- {std[2]}')

def save_pars(pars,std):
    df = pd.DataFrame(data=None,columns=['descr','y0','a0','m'])
    df['descr'] = pd.Series(['value','+ Std. Dev', '- Std. Dev'])
    df['y0'] = pd.Series([pars[0],pars[0] + std[0], pars[0] - std[0]])
    df['a0'] = pd.Series([pars[1],pars[1] + std[1], pars[1] - std[1]])
    df['m'] = pd.Series([pars[2],pars[2] + std[2], pars[2] - std[2]])
    df.to_csv(os.path.join(dir,'data/celldiv_time_fit_parameters.csv'),header=True,index=False)

    return df

if __name__ == '__main__':
    data,black,blue = read_data()
    data_p,cov = fit_data(data['a'],data['gamma'])
    hill_p = fit_hill(data['a'],data['gamma'])

    #get std dev
    perr = np.sqrt(np.diag(cov))

    data_calc = calc_data(1,800,data_p)
    #parameters obtained after minizing fitted parameters in integro_diff_eq_adapted
    minimize_data = calc_data(1,800,[1.62,105.29,1.69721935])
    data_pos_std = calc_data(1,800,data_p + perr)
    data_neg_std = calc_data(1,800,data_p - perr)
    hill_calc = calc_hill(1,800,hill_p)

    dir = os.path.dirname(__file__)
    plot_data(black,blue,data_calc,data_pos_std,data_neg_std,hill_calc,minimize_data)

    print("Calculated parameters")
    print_pars(data_p,perr)

    save_pars(data_p,perr)







