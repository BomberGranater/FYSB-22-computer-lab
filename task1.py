import numpy as np
import matplotlib.pyplot as plt


# Parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
nu0 = 6.0
h = (ximax - ximin)/Nsteps
ep = -4.000


xi = np.linspace(ximin, ximax, Nsteps)

nu = np.array([-nu0 * (np.cosh(xi_) ** -2) for xi_ in xi])


def gen_phi(ep_):

    phi_ = np.zeros(np.shape(xi))

    f = nu - ep_

    q = np.sqrt(-ep_)

    phi_[0] = 1
    phi_[1] = np.exp(q * h)

    for i in range(1, np.size(xi) - 1):
        phi_[i + 1] = (2 + f[i] * h ** 2) * phi_[i] - phi_[i - 1]

    return phi_


def count_nodes(fun):
    # count number of sign changes for node nr
    n = 0
    plus = (fun[1] > 0)
    for i in range(2, np.size(fun)):
        if plus and fun[i] < 0:
            plus = False
            n += 1
            continue
        if (not plus) and fun[i] > 0:
            plus = True
            n += 1
            continue

    return n


# quick and dirty binary search time
left = ep-0.005
right = ep+0.005
mid = left + (right - left) / 2

for i in range(1000):
    mid = left + (right - left) / 2
    phi = gen_phi(mid)
    last = phi[-1]
    if abs(last) < 10**-8:
        print("broke at", i)
        break

    if last < 0:
        right = mid

    else:
        left = mid


phi = gen_phi(mid)

nodes = count_nodes(phi)

print(f"function has {nodes} nodes")
plt.plot(xi, gen_phi(mid), label=f'ep={mid}')

# actual bisection method
ep




plt.legend()
plt.show()
