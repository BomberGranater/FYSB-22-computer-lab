import numpy as np
import matplotlib.pyplot as plt
# Parameters
ximax=8
ximin=-ximax
Nsteps=100*2*ximax
nu0=6.0


# vectors

xi = np.linspace(ximin, ximax, Nsteps)

phi = np.zeros(Nsteps)

h = (ximax - ximin)/Nsteps 

nu = np.array([-nu0 * (np.cosh(xi_)**(-2)) for xi_ in xi])
# eller ([-nu0 * (np.cosh(xi[i])**(-2)) for i in range])


# guess epsilon

eps = -0.998295      # -3.999018575 Two nodes  # -0.998295




# define q, phi[0], and phi[1]

q = np.sqrt(-eps)

phi[0] = 1
phi[1] = np.exp(q*h)


# define f

f = nu - eps


# for loop time

for i in range(1, np.size(xi) - 1):
    phi[i+1] = (2 + h**2*f[i])*phi[i] - phi[i-1]


# plot phi vs xi

plt.plot(xi, phi, color= 'maroon')
plt.show()



