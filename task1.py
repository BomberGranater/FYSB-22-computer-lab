import numpy as np
import matplotlib.pyplot as plt
# Parameters
ximax=8
ximin=-ximax
Nsteps=100*2*ximax
nu0=6.0

xi = np.linspace(ximin, ximax, Nsteps)

phi = np.zeros(Nsteps)

h = (ximax - ximin)/Nsteps

nu = np.array([-nu0 * (np.cosh(xi_)**(-2)) for xi_ in xi])
# eller ([-nu0 * (np.cosh(xi[i])**(-2)) for i in range])

# Guess

E = 
