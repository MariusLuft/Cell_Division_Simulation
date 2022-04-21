import numpy as np
from numpy import random
import matplotlib.pyplot as mp
from matplotlib.widgets import TextBox 
from scipy.interpolate import interp1d
from scipy.integrate import odeint, quad, trapz
from scipy.optimize import minimize,curve_fit
import csv 
import time
import os


#loads data from path and fits a function to it, returning the original and the fitted values
#example of fitted function in test.ipynb
def load_exp_set_fitted(path):

    X = []
    Y = []
    path = os.path.join(os.path.dirname(__file__),path)
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            X.append(float(row[0]))
            Y.append(float(row[1]))

        def f(z, m, s): return np.exp(-0.5 * (np.log(z/m)/s)**2) / \
            (z * s * np.sqrt(2 * np.pi))

        p_opt, p_cov = curve_fit(
            f, X, Y, [70, 0.2], bounds=((1, 0.1), (100, 100)))
        p_std = np.sqrt(np.diag(p_cov))

    Xp = np.linspace(15, 215, num=200, endpoint=True)
    Yp = f(Xp, p_opt[0], p_opt[1])

    m = p_opt[0]
    s = p_opt[1]
    E = m * np.exp(0.5*s**2)
    sigma = E * np.sqrt(np.exp(s**2) - 1)
    # print(path + ": E = " + str(E) + ", sigma = " + str(sigma))
    # mp.figure()
    # mp.text(40, 0, "fit model:\nlog-normal-distro:" + "\nm = " + "{:3.2f}".format(p_opt[0]) + " +/- " + "{:3.2f}".format(p_std[0]) + "\ns = " + "{:3.2f}".format(p_opt[1]) + " +/- " + "{:3.2f}".format(p_std[1]),
    # bbox=dict(boxstyle="square", ec=(0., 0., 0.), fc=(1., 1., 1.)))

    # mp.plot(X, Y)
    # mp.plot(Xp, Yp)

    return X, Y, Xp, Yp, E, sigma

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

def bins2pts(bins):
    p = []
    for i in range(len(bins)-1):
        p.append(0.5 * bins[i] + 0.5 * bins[i+1])
    return p

def pts2bins(pts):
    #determine distance between every point pair
    dx = pts[1:] - pts[:-1]
    bins = np.zeros(len(pts) + 1)
    bins[0] = pts[0] - dx[0] / 2
    bins[1:-1] = pts[:-1] + dx * 0.5
    bins[-1:] = pts[-1] + dx[-1] / 2
    return bins 

#calculates the value for the cell size distribution function using the provided experimental distribution
def density(distro, x): 
    xp = pts2bins(x)
    logs = [np.log(xp[i+1]/xp[i]) for i in range(len(x))] # FIXME
    #logs = np.log(xp[1:]/xp[:-1])
    density = 1e3 * np.sum(distro * logs)
    return density

def G_simp_old(a, q, d):
    g = 0.94
    dt = 1
    
    #change: q ~ [0.5, 0.5, 0.667, 0.5] -> [50, 5, 66.7, 5]: umrechnung [100, 10, 100, 10]
    a0 = 100 * q[0]
    m = 10 * q[1]
    d0 = 100 * q[2]
    k = 10 * q[3]
    
    gd = np.power(1 - np.power(d / d0, k), 1 / k)
    ga = np.exp(-np.power(a0 / a, m))
    return g / dt * ga * gd

#calculate gamma function in accordance with equation 2 in main paper
#q = [y0,a0,m]
def G_simp(a,q):

    y0 = q[0]
    a0 = q[1]
    m =  q[2]

    return y0 * np.exp(- np.power(a0/a,m))


def G_old(a, q, d): 
    if isinstance(a, list) or isinstance(a, np.ndarray):
        X = G_simp(a, q ,d)
        nan = np.isnan(X)
        X[nan] = 0
        return X
    else:
        return G_simp(a, q, d) if a > 0 and d < q[2] else 0 # q[2] = d0


def G(a, q): 
    try:
        if isinstance(a, list) or isinstance(a, np.ndarray):
            X = G_simp(a, q)
            nan = np.isnan(X)
            X[nan] = 0
            return X
        else:
            return G_simp(a, q) if a > 0 else 0 
    except ZeroDivisionError as ex:
        return 0
    
    
def Gauss(x, m, s):
    return np.exp(- 0.5*((x-m)/s)**2) / np.sqrt(2 * np.pi * s**2)

 
def rho_prime(rho, t, q, Xb):
    # D = int_0^{\infty} da rho(a,t) / a
    # d/dt rho(a,t) = - g(a, q, D) * rho(a,t) + int_a^{\infty} dz gamma(z, q, D) rho(z,t) G(a, z)
    
    d = density(rho, Xb)
    
    drho_dt = []
    for i in range(len(Xb)):
        a = Xb[i]
        f_a = lambda z: (a / z**2) * 2 * G(z, q) * (Gauss(a / z, 0.5, 0.125) + Gauss(1 - a / z, 0.5, 0.125))
        f = np.array(f_a(Xb)) * np.array(rho)
        integral = trapz(f, Xb)
        drho_dt.append(-G(a, q) * rho[i] + integral)
         
    return drho_dt

#calculates derivative according to equation 9 in main paper
#rho represents the currently used experimental cell size distribution 
#Xb represents the experimental x values for the given cell size distribution rho
def rho_prime_new(rho, t, q, Xb):

    drho_dt = []
    for i in range(len(Xb)):
        a = Xb[i]
        #2 ist Korrekturfaktor siehe rho_prime_old
        f_p = lambda z:2* Gauss(z,0.5,0.125)*G(a / z,q) * density(rho,a/z)
        f_q = lambda z:2* Gauss(z,0.5,0.125)*G(a / (1- z),q) * density(rho,a/(1-z))
        f = np.array(f_p(Xb)) * np.array(rho) 
        integral_p = trapz(f,Xb)
        f = np.array(f_q(Xb)) * np.array(rho) 
        integral_q = trapz(f,Xb)

        #calculating final value of eq 9 
        drho_dt.append(-G(a,q) * rho[i] + integral_p + integral_q)

    return drho_dt


def solve_rho_old(x, y_init, q):
    #q = [50, 5, 66.7, 5]
    t_int = [0, 4, 9, 14]
    
    sol = odeint(rho_prime, y_init, t_int, args = (q, x))  
    
    y2 = sol[1] / trapz(sol[1], x)
    y4 = sol[2] / trapz(sol[2], x)
    #y15 = sol[3] / trapz(sol[3], x)
    
    return y2, y4

#x - experimental x values
def solve_rho(x,y_init,q):
    #days to solve for (1,2,4)
    t_int = [0,1,5]

    sol = odeint(rho_prime,y_init,t_int,args = (q,x))

    #normierung auf Fläche 1
    y1 = sol[0] / trapz(sol[0], x)
    y2 = sol[1] / trapz(sol[1], x)
    y4 = sol[2] / trapz(sol[2], x)

    return y1,y2,y4

qlist = []

def residual(q): 
    
    y1,y2, y6 = solve_rho(x1interp, y1interp, q)
    res = ((y6interp - y6)**2).sum() + ((y2interp - y2)**2).sum()
    
    print(str(q) + ": " + str(res))
    qlist.append((q,res))
    
    return res

if __name__ == '__main__':
    _, _, x1interp, y1interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_1d.csv")
    _, _, x2interp, y2interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_2d.csv")
    _, _, x6interp, y6interp = load_exp_set("data\\Pul_2012_01_celldensitydistro_6d.csv")

    #q = [0.5, 0.5, 0.667, 0.5]#[50, 5, 66.7, 5]
    #[y0,a0,m]
    q = [1.46,115,1.56]
    #q = [2.1,93,2.05]

    result = minimize(residual, q, method="L-BFGS-B", bounds = ((1.31, 1.62), (105.29, 126.23), (1.14,1.98))) #method="Nelder-Mead"
    #result = minimize(residual, q, method="L-BFGS-B", bounds = ((2.0, 2.2), (83, 103), (1.5,2.5))) #method="Nelder-Mead"
    #result.x = [20.10,93,25.05]
    y1,y2, y6 = solve_rho(x1interp, y1interp, result.x)

    print("optimal q = " + str(result.x) + ", residue = " + str(residual(result.x)))

    fig = mp.figure(figsize=(8,6))
    fig.suptitle('Fitting für Zellgrößenverteilung', fontsize=30)
    axes = fig.gca()
    axes.tick_params(axis='both',which='major',labelsize = 15)
    mp.xlabel('Zellgröße A [um^2]', fontsize=20)
    mp.ylabel('p(A)', fontsize = 20)
    mp.plot(x1interp, y1interp, label = "Exp - t = 1d",color='blue')
    mp.plot(x2interp, y2interp, label = "Exp - t = 2d",color='cyan')
    mp.plot(x6interp, y6interp, label = "Exp - t = 6d",color='green')
    ax = mp.gca()
    ax.set_ylim([0,0.06])
    ax.set_xlim([0,150])

    #mp.plot(x1interp, y1, label = "Sim - t = 1d",linestyle='dashed',color='blue')
    mp.plot(x2interp, y2, label = "Sim - t = 2d",linestyle='dashed',color='cyan')
    mp.plot(x6interp, y6, label = "Sim - t = 6d",linestyle='dashed', color = 'green')
    mp.grid(visible=True)
    

    mp.legend()
    mp.show()
    fig.savefig(os.path.join(os.path.dirname(__file__),'figures/celldiv_symm_fitted_d6_oldq.png'))

