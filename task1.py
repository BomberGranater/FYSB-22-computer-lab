# task 1

import numpy as np
import matplotlib.pyplot as plt

# parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
nu0 = 6.0


xi = np.linspace(ximin, ximax, Nsteps)
phi = np.zeros(Nsteps)
h = (ximax-ximin)/Nsteps
nu = -nu0*np.cosh(xi)**(-2)
epsilon = -nu0/2
q = np.sqrt(-epsilon)
phi0 = 1
phi1 = np.e**(q*h)


# recursion relation and for loop

for i in range():
    f = nu[i] - epsilon
    phi[i+1] = (2+(h**2)*f[i])*phi[i]-phi[i-1]
