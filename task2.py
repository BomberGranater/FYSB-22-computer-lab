import numpy as np
import matplotlib.pyplot as plt


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


def find_ep(goal):
    ep_min = -2 * nu0
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
        if abs(ep_min - ep_max) < 10**-9:  # Relaxed convergence check
            # print("converged")
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



S = np.linspace(0.1, 4, 15, endpoint=True)

x = []
vals = []

ep_l = np.array([])
node_l = np.array([])

for i in range(np.size(S)):

    # Parameters
    s = S[i]

    ximax = int(s + 8)
    ximin = -ximax
    Nsteps = 100*2*ximax
    b = 2
    nu0 = b*(b+1)
    h = (ximax - ximin)/Nsteps
    target = 0  # target number of nodes in search
    ep = -(b-target)**2



    xi = np.linspace(ximin, ximax, Nsteps)

    nu = np.array([(-nu0 * np.cosh(xi_+s)**-2) + (-nu0 * np.cosh(xi_-s)**-2) for xi_ in xi])

    nodes_max = count_nodes(gen_phi(-0.1))

    for j in range(nodes_max):
        ep = find_ep(j)
        x.append(S[i])
        vals.append(ep)

    if i == 0:
        ep_l = np.linspace(-2*nu0, -0.1, Nsteps)
        node_l = np.zeros(Nsteps)
        for j in range(Nsteps):
            node_l[j] = count_nodes(gen_phi(ep_l[j]))


plt.rcParams["mathtext.fontset"] = "cm"

plt.scatter(x, vals)
plt.title("Energy of bound states for double well potential")
plt.ylabel(r"Bound state energy $\epsilon$")
plt.xlabel(r"Distance between wells $s$")
plt.figure()
plt.plot(ep_l, node_l)

plt.show()