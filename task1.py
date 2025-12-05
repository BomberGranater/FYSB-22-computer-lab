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
phi[0] = 1
phi[1] = np.e**(q*h)
f = nu - epsilon

# recursion relation and for loop

for i in range(Nsteps-1):
    phi[i+1] = (2+(h**2)*f[i])*phi[i]-phi[i-1]

plt.plot(xi, phi)
plt.xlabel('$\\xi$')
plt.ylabel('$\phi$')
plt.title('wave function')
plt.show()
