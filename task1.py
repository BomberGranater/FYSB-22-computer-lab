import numpy as np
import matplotlib.pyplot as plt


# Parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
b = 3
nu0 = b*(b+1)
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
    n = 2
    plus = (fun[1] > 0)
    for i in range(10, np.size(fun)-10):
        if plus and fun[i] < 0:
            plus = False
            n += 1
            continue
        if (not plus) and fun[i] > 0:
            plus = True
            n += 1
            continue

    return n


# actual bisection method
#plt.plot(xi, gen_phi(-6))
print(f"minimum energy function has {count_nodes(gen_phi(-6))} nodes")
#plt.figure()
print(f"maximum energy function has {count_nodes(gen_phi(-0.1))} nodes")
#plt.plot(xi,  gen_phi(-0.1))


def find_ep(goal):
    ep_min = -nu0
    ep_max = -0.1
    phi_min = gen_phi(ep_min)
    phi_max = gen_phi(ep_max)

    nodes_min = count_nodes(phi_min)
    nodes_max = count_nodes(phi_max)

    if nodes_min > goal or nodes_max < goal:
        return 404

    # find right range for epsilon
    for j in range(500):
        if abs((ep_min - ep_max)/(ep_min + ep_max)) < 10**-8:
            print("converged")
            break

        ep_mid = (ep_min + ep_max)/2.0
        phi_mid = gen_phi(ep_mid)

        if count_nodes(phi_mid) < goal:
            ep_min = ep_mid
            continue

        if count_nodes(phi_mid) > goal:
            ep_max = ep_mid
            continue

        test = phi_mid[-2]*phi_mid[-1] - np.exp(np.sqrt(-ep_mid)*h)*(phi_mid[-1]**2)

        if test < 0:
            ep_min = ep_mid
            continue

        if test > 0:
            ep_max = ep_mid
            continue

    return ep_min, ep_max


interval = find_ep(4)
print("right range for one node is", interval)

phi = gen_phi(interval[0])

plt.figure()
plt.plot(xi, phi, label='found function')

plt.legend()
plt.show()
