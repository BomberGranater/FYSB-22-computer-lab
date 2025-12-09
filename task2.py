import numpy as np
import matplotlib.pyplot as plt

def gen_phi(ep_):


    phi_ = np.zeros(np.size(xi))
    f = nu - ep_
    q = np.sqrt(-ep_)

    phi_[0] = 1
    phi_[1] = np.exp(q * h)

    for i in range(1, np.size(xi) - 1):
        phi_[i + 1] = (2 + f[i] * h ** 2) * phi_[i] - phi_[i - 1]

    return phi_

def count_nodes(y):
    #this counts number of sign changes as a number of nodes
    n = 0
    plus = (y[1] > 0)

    for i in range(10, np.size(y) - 10):
        if plus and y[i] < 0:
            plus = False
            n += 1
            continue
        if (not plus) and y[i] > 0:
            plus = True
            n += 1
            continue

    return n

def find_ep(goal):
    ep_min = -2 * nu0
    ep_max = -0.1
    phi_min = gen_phi(ep_min)
    phi_max = gen_phi(ep_max)

    nodes_min = count_nodes(phi_min)
    nodes_max = count_nodes(phi_max)

    if nodes_min > goal or nodes_max < goal:
        print("not enough nodes")
        return 404

    #find right range for epsilon
    for j in range(500):
        if abs(ep_min - ep_max) < 10**(-9):
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

        if test < 1:    #always look for maximum epsilon
            ep_min = ep_mid

        if test > 1:    #bad guess on interval
            print("goal outside search area")
            return 404

    return (ep_min + ep_max)/2


# Parameters
S = np.linspace(0.01, 4, 10, endpoint=True)
x = []
vals = []

for i in range(np.size(S)):

    s = S[i]

    ximax = int(s + 8)
    ximin = -ximax
    Nsteps = 100*2*ximax
    b = 2
    nu0 = b*(b+1)
    h = (ximax - ximin)/Nsteps
    target = 0  #target number of nodes in search
    ep = -(b - target)**2

    xi = np.linspace(ximin, ximax, Nsteps)
    nu = np.array([-nu0*(np.cosh(xi_ + s)**(-2)) + (-nu0*(np.cosh(xi_ - s)**(-2))) for xi_ in xi])
    nodes_max = count_nodes(gen_phi(-0.1))

    for j in range(nodes_max):
        ep = find_ep(j)
        x.append(S[i])
        vals.append(ep)

plt.scatter(x, vals)
plt.show()



