import numpy as np
from numpy import random
import matplotlib.pyplot as mp
from matplotlib.widgets import TextBox
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize
from matplotlib.widgets import TextBox
from scipy.stats import moment
from scipy.linalg import expm
from scipy.integrate import quad, trapz
import csv
import time
import os


def bins2pts(bins):
    p = []
    for i in range(len(bins)-1):
        p.append(0.5 * bins[i] + 0.5 * bins[i+1])
    return p

def pts2bins(pts):
    dx = pts[1:] - pts[:-1]
    bins = np.zeros(len(pts) + 1)
    bins[0] = pts[0] - dx[0] / 2
    bins[1:-1] = pts[:-1] + dx * 0.5
    bins[-1:] = pts[-1] + dx[-1] / 2
    return bins 


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

def G(a, a0, m, g):
    dt = 1  # 240 # 1d / 6 min = 240
    if isinstance(a, list) or isinstance(a, np.ndarray):
        X = g / dt * np.exp(-np.power(a0 / a, m))
        nan = np.isnan(X)
        X[nan] = 0
        return X
    else:
        return (g / dt * np.exp(-np.power(a0 / a, m))) if a != 0 else 0

def Gauss(x, m, s):
    return np.exp(- 0.5*((x-m)/s)**2) / np.sqrt(2 * np.pi * s**2)

def KernelA(Xb, a0, m, g):

    dx = np.mean(Xb[1:] - Xb[:-1])

    #alter ansatz
    # S = np.zeros([len(Xb), len(Xb)])
    # T = np.zeros([len(Xb), len(Xb)])

    # for i in range(len(Xb)):
    #     T[i][i] = -G(Xb[i], a0, m, g)

    #     for j in range(i, len(Xb)):
    #         if Xb[j] > 0:
    #             def f(z): return 2 * (Xb[i] / z**2) * G(z, a0, m, g) * (
    #                 Gauss(Xb[i] / z, 0.5, 0.125) + Gauss(1 - Xb[i] / z, 0.5, 0.125))
    #             #                                 ^---könnte der faktor 2 hier von einer [i]/[i+1] geschichte kommen?
    #             G_ij = quad(f, Xb[j] - dx/2, (Xb[j+1] if j+1 < len(Xb) else 2 * Xb[-1] - Xb[-2]) - dx/2)
    #             S[i][j] = G_ij[0]

    #neuer ansatz
    S_ = np.zeros([len(Xb), len(Xb)])
    T_ = np.zeros([len(Xb), len(Xb)])
    Z = pts2bins(Xb)

    for i in range(len(Xb)):
        T_[i, i] = -G(Xb[i], a0, m, g)
        def f(x): return 2 * (Xb[i] / x**2) * G(x, a0, m, g) * (Gauss(Xb[i] / x, 0.5, 0.125) + Gauss(1 - Xb[i] / x, 0.5, 0.125))
        # def F(a,b) : return quad(f, a, b)[0]
        # G_ij = np.array(list(map(F, Z[i:-1], Z[i+1:])))
        # basically equivalent to an integral at this point
        S_[i, i:] = f(Xb[i:]) * dx

    #vergleich
    # mp.plot(S[0, :])
    # mp.plot(S_[0, :])
    # mp.show()

    return T_ + S_

def analyticDistro(Xb, a0, m, g):

    y0 = Y_init.copy()  # np.array(Y_init)

    K = expm(KernelA(Xb, a0, m, g))  # 1d

    r = y0
    y = [y0]
    for i in range(5):
        r = np.dot(K, r)
        r = r / np.trapz(r, Xb)
        y.append(r) 

    return y

def residual(q, Xb):
    g = q[0] / q_trafo[0]
    a0 = q[1] / q_trafo[1]
    m = q[2] / q_trafo[2]

    y = analyticDistro(Xb, a0, m, g)
    res = 0
    # mp.figure()
    for i in range(len(y)):
        if i == 2:
            contr = np.sum((exp[i]["y"] - y[i])**2)
            res = res + contr
    #     mp.plot(exp[i]["y"], "C" + str(i), label = "exp: " + str(i) + " d")
    #     mp.plot(y[i], "C" + str(i) + "P", label = "sim: " + str(i) + " d (+{:0.3f})".format(1000 * contr) )
    # mp.legend()
    # mp.title("here!")
    # mp.show()

    # print("g = " + str(g) +", a0 = " + str(a0) + ", m = " + str(m) + ": " + str(res))

    return res


print("loading data")
print("measuring time...")
start = time.time()
mp.figure(figsize=(14,6))
# mp.subplot(211)
exp = [{} for i in range(6)]
for i in range(6):
    _, _, xinterp, yinterp, E, sigma = load_exp_set_fitted(
        "data/Pul_2012_01_celldensitydistro_" + str(i + 1) + "d.csv")
    exp[i]["x"] = xinterp
    exp[i]["y"] = yinterp
    exp[i]["E"] = E
    exp[i]["sigma"] = sigma

    mp.plot(exp[i]["x"], exp[i]["y"], "C" + str(i),
            label="exp t = " + str(i + 1) + " d" + "$(\mu = {:0.2f}, \sigma = {:0.2f})$".format(E, sigma))

Y_init = exp[0]["y"]
stop = time.time()
print("calc took " + "{:0.4f}".format(stop - start) + " s in total")

q_trafo = np.array([1, 0.01, 0.1])
q = np.array([2.25, 61.25, 15])   # 62.90726311285763, 3.9360321446747952

print("fitting by distro curve")
print("measuring time...")
start = time.time()
result = minimize(residual, q * q_trafo, args=exp[0]["x"], method="L-BFGS-B", bounds=((0, 2.5), (0.2, 1.2), (0.1, 1)))
q = result.x / q_trafo
start2 = time.time()
y = analyticDistro(exp[0]["x"], q[1], q[2], q[0])
stop2 = time.time()
for i in range(len(y) - 1):
    mp.plot(exp[0]["x"], y[i + 1], "C" + str(i+1) + "--", label=str(i+2) + "d. g = " + "{:0.2f}".format(q[0]) + ", A = " + "{:0.2f}".format(q[1]) + ", M = " + "{:0.2f}".format(q[2]))
stop = time.time()
print("calc took " + "{:0.4f}".format(stop - start) + " s in total and " +
      "{:0.4f}".format(stop2 - start2) + " s for a single calculation")
 

mp.legend()
mp.xlabel("$\sigma / \mu m^2$")
mp.ylabel("$\mu / \mu m^2$")
mp.grid()
# mp.show()

def mu_sigma(x, y):

    xp = pts2bins(x)

    q = xp[1:]**2 - xp[:-1]**2
    mu = np.sum(q * y) * 0.5

    Q = (xp[1:]-mu)**3 - (xp[:-1]-mu)**3
    sigma = np.sqrt(np.sum(Q * y) / 3)

    return mu, sigma

def complete_trace_sim(g, a0, m, l):

    # g = 0.94  # 0.94 teilungen pro tag - für kleinere schritte entsprechend einheit umrechnen
    K = expm(KernelA(exp[0]["x"], a0, m, g))
    # y0 = np.array(exp[0]["y"])
    y0 = exp[0]["y"].copy()

    mu_list = []
    sigma_list = []

    m, s = mu_sigma(exp[0]["x"], y0)
    mu_list.append(m)
    sigma_list.append(s)

    for i in range(l-1):
        rp = np.dot(K, y0)
        y0 = rp / np.trapz(rp, exp[0]["x"])

        m, s = mu_sigma(exp[0]["x"], y0)
        mu_list.append(m)
        sigma_list.append(s)

    return mu_list, sigma_list

def complete_trace_exp(l):

    mu_list = []
    sigma_list = []

    for i in range(l):
        m, s = mu_sigma(exp[i]["x"], exp[i]["y"])
        mu_list.append(m)
        sigma_list.append(s)

    return mu_list, sigma_list

def trace_residual(q):

    g, a0, m = q / q_trafo

    e_mu, e_sigma = complete_trace_exp(6)
    s_mu, s_sigma = complete_trace_sim(g, a0, m, 6)

    e_mu = np.array(e_mu) * q_trafo[1]
    e_sigma = np.array(e_sigma) * q_trafo[2]
    s_mu = np.array(s_mu) * q_trafo[1]
    s_sigma = np.array(s_sigma) * q_trafo[2]

    L = np.sum((e_mu[0:2] - s_mu[0:2])**2 + (e_sigma[0:2] - s_sigma[0:2])**2) + np.sum((e_mu[3:5] - s_mu[3:5])**2 + (e_sigma[3:5] - s_sigma[3:5])**2)

    # print('mu = {0:3.2f}, sigma = {1:03.2f}: L = {2:3.5f}'.format(
    #     100 * np.mean(e_mu), 10 * np.mean(e_sigma), L))
    return L


# mp.subplot(212)
# sim_mu, sim_sigma = complete_trace_sim(A, M, 6)
# mp.plot(sim_sigma, sim_mu, "C0o",
#         label='sim: $a_0$ = {0:0.2f}, m = {1:0.2f}'.format(A, M))
# for i in range(len(sim_mu)):
#     mp.text(sim_sigma[i], sim_mu[i], str(i + 1) + "d")

print("fitting by mu/sigma trace")
print("measuring time...")
start = time.time()
q = np.array([2.25, 71.25, 6.09])
opt_trace = minimize(trace_residual, q * q_trafo, method="L-BFGS-B", bounds=((0.1, 2.5), (0.2, 1.2), (0.1, 2)))
q = result.x / q_trafo
stop = time.time()

start2 = time.time()
sim_mu, sim_sigma = complete_trace_sim(q[0], q[1], q[2], 6)
stop2 = time.time()

print("calc took " + "{:0.2f}".format(stop - start) + " s in total and " +
      "{:0.2f}".format(stop2 - start2) + " s for a single calculation")

# mp.plot(sim_sigma, sim_mu, "C1o",
#         label='sim: $a_0$ = {:0.2f}, m = {:0.2f}'.format(A, M))
# for i in range(len(sim_mu)):
#     mp.text(sim_sigma[i], sim_mu[i], str(i + 1) + "d")

# exp_mu, exp_sigma = complete_trace_exp(6)
# mp.plot(exp_sigma, exp_mu, "C3P", label="exp")
# for i in range(len(exp_mu)):
#     mp.text(exp_sigma[i], exp_mu[i], str(i + 1) + "d")

mp.legend()
mp.xlabel("$\sigma / \mu m^2$")
mp.ylabel("$\mu / \mu m^2$")
mp.grid()


# mp.subplot(211)
y = analyticDistro(exp[0]["x"], q[1], q[2], q[0])
for i in range(len(y)):
    mp.plot(exp[0]["x"], y[i], "C" + str(i+1) + ":",
            label=str(i+2) + "d. g = " + "{:0.2f}".format(q[0]) + ", A = " + "{:0.2f}".format(q[1]) + ", M = " + "{:0.2f}".format(q[2]))

def trace_residual2(q):

    g, a0, m = q / q_trafo

    e_mu, e_sigma = complete_trace_exp(6)
    s_mu, s_sigma = complete_trace_sim(g, a0, m, 6)

    e_mu = np.array(e_mu) * q_trafo[1]
    e_sigma = np.array(e_sigma) * q_trafo[2]
    s_mu = np.array(s_mu) * q_trafo[1]
    s_sigma = np.array(s_sigma) * q_trafo[2]

    L = (e_mu[5] - s_mu[5])**2 + (e_sigma[5] - s_sigma[5])**2 # np.sum( (e_mu[3:5] - s_mu[3:5])**2 + (e_sigma[3:5] - s_sigma[3:5])**2 ) #np.sum((e_mu[0:2] - s_mu[0:2])**2 + (e_sigma[0:2] - s_sigma[0:2])**2) + np.sum((e_mu[3:5] - s_mu[3:5])**2 + (e_sigma[3:5] - s_sigma[3:5])**2)

    # print('mu = {0:3.2f}, sigma = {1:03.2f}: L = {2:3.5f}'.format(
    #     100 * np.mean(e_mu), 10 * np.mean(e_sigma), L))
    return L

print("fitting by mu/sigma trace's last pt")
print("measuring time...")
start = time.time()
q = np.array([2.25, 71.25, 6.09])
opt_trace = minimize(trace_residual2, q * q_trafo, method="L-BFGS-B", bounds=((0.1, 2.5), (0.2, 1.2), (0.1, 2)))
q = result.x / q_trafo
stop = time.time()

start2 = time.time()
sim_mu, sim_sigma = complete_trace_sim(q[0], q[1], q[2], 6)
stop2 = time.time()

print("calc took " + "{:0.2f}".format(stop - start) + " s in total and " +
      "{:0.2f}".format(stop2 - start2) + " s for a single calculation")

# mp.subplot(212)
# mp.plot(sim_sigma, sim_mu, "C5p",
#         label='sim: $a_0$ = {:0.2f}, m = {:0.2f}'.format(A, M))
# for i in range(len(sim_mu)):
#     mp.text(sim_sigma[i], sim_mu[i], str(i + 1) + "d")

# exp_mu, exp_sigma = complete_trace_exp(6)
# mp.plot(exp_sigma, exp_mu, "C3P", label="exp")
# for i in range(len(exp_mu)):
#     mp.text(exp_sigma[i], exp_mu[i], str(i + 1) + "d")

mp.legend()
mp.xlabel("$\sigma / \mu m^2$")
mp.ylabel("$\mu / \mu m^2$")
# mp.grid()

# mp.subplot(211)
y = analyticDistro(exp[0]["x"], q[1], q[2], q[0])
for i in range(len(y)):
    mp.plot(exp[0]["x"], y[i], "C" + str(i+1) + "p",
            label=str(i+2) + "d. g = " + "{:0.2f}".format(q[0]) + ", A = " + "{:0.2f}".format(q[1]) + ", M = " + "{:0.2f}".format(q[2]))


mp.xlabel("a / $\mu m^2$")
mp.ylabel("density * $\mu m^2$")
mp.grid()
mp.legend(ncol = 2)
mp.title("fitting: $(\gamma_0, a_0, m) \\rightarrow \gamma_0 exp(-(a_0/a)^m)$")

mp.show()
