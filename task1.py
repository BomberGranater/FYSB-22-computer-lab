import numpy as np
import matplotlib.pyplot as plt


# Parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
b = 3
nu0 = b*(b+1)
h = (ximax - ximin)/Nsteps
target = 2  # target number of nodes in search
ep = -(b-target)**2


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
print(f"minimum energy function has {count_nodes(gen_phi(-nu0))} nodes")
print(f"maximum energy function has {count_nodes(gen_phi(-0.1))} nodes")


def find_ep(goal):
    ep_min = -nu0
    ep_max = -0.1
    phi_min = gen_phi(ep_min)
    phi_max = gen_phi(ep_max)

    nodes_min = count_nodes(phi_min)
    nodes_max = count_nodes(phi_max)

    if nodes_min > goal or nodes_max < goal:
        print("Not enough nodes")
        return 404

    # find right range for epsilon
    for j in range(500):
        if abs(ep_min - ep_max) < 10**-16:
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

        if test < 1:  # Always look for maximum epsilon
            ep_min = ep_mid

        if test > 1:  # Bad guess on interval
            print("Goal outside search area")
            return 404

    return (ep_min + ep_max)/2


# Binary search double check
left = ep - 0.005
right = ep + 0.005
mid = ep

for i in range(1000):
    mid = (left + right) / 2

    phi = gen_phi(mid)

    last = phi[-1]

    if last * (-1)**target < 0:
        right = mid

    if last * (-1)**target > 0:
        left = mid


phi2 = gen_phi(mid)


Y = np.zeros(Nsteps)

nodes = np.zeros(Nsteps)

eps = np.linspace(-nu0, -0.1, Nsteps)

for i in range(Nsteps):
    phi = gen_phi(eps[i])
    Y[i] = phi[-2]*phi[-1] - np.exp(np.sqrt(-eps[i])*h)*(phi[-1]**2)
    nodes[i] = count_nodes(phi)


ep = find_ep(target)
phi = gen_phi(ep)


# normalize
phi = phi/np.max(phi)

phi2 = phi2/np.max(phi2)

print("middle value bisect:", ep)
print("binary search value:", mid)


plt.plot(xi, phi2)
plt.title("Binary search")
plt.figure()
plt.plot(xi, phi)
plt.title("Bisection method")
plt.figure()
plt.plot(eps, nodes)
plt.scatter(ep, count_nodes(phi), label="bisection")
plt.scatter(mid, count_nodes(phi2), marker="x", label="binary search")

plt.legend()
plt.show()
