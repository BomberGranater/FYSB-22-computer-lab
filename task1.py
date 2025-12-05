import numpy as np
import matplotlib.pyplot as plt


# Parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
nu0 = 6.0
h = (ximax - ximin)/Nsteps
ep = -nu0/2


xi = np.linspace(ximin, ximax, Nsteps)

phi = np.zeros(np.shape(xi))

v = np.array([-nu0*(np.cosh(xi_)**-2) for xi_ in xi])

f = v - ep

q = np.sqrt(-ep)

phi[0] = 1
phi[1] = np.exp(q * h)

for i in range(1, np.size(xi) - 1):
    phi[i + 1] = (2 + f[i]*h**2)*phi[i] - phi[i - 1]


