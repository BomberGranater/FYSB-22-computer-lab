import numpy as np
import matplotlib.pyplot as plt

# Parameters
ximax=8
ximin=-ximax
Nsteps=100*2*ximax
nu0=6.0
ep = -4
h = (ximax - ximin)/Nsteps
b = 2

xi = np.linspace(ximin, ximax, Nsteps)
nu = np.array([-nu0*(np.cosh(xi_)**(-2)) for xi_ in xi])

def gen_phi(ep_):
    phi_ = np.zeros(np.shape(xi))
    f = nu - ep_
    q = np.sqrt(-ep_)

    phi_[0] = 1
    phi_[1] = np.exp(q * h)

    for i in range(1, np.size(xi) - 1):
        phi_[i + 1] = (2 + f[i] * h ** 2) * phi_[i] - phi_[i - 1]

    return phi_


#binary search, search for correct value between a too big and too small value.
left = -4
right = -3.999
mid = left + (right - left)/2

for i in range(1000):
    mid = left + (right - left)/2
    phi = gen_phi(mid)
    last = phi[-1]
    if abs(last) < 10**(-8):
        print("broke at", i)
        break

    if last < 0:
        right = mid

    else:
        left = mid

phi1 = gen_phi(mid)
phi2 = gen_phi(-1)  #not exactly correct value

fig = plt.figure()  #plot for first excited state?
plt.plot(xi, phi1)

fig = plt.figure()  #plot for ground state
plt.plot(xi, phi2)

ep_min = -nu0
ep_max = -0.1

phi_min = gen_phi(ep_min)
phi_max = gen_phi(ep_max)

fig = plt.figure()  #plot for
plt.plot(xi, phi_min)

fig = plt.figure()
plt.plot(xi, phi_max)


plt.show()




