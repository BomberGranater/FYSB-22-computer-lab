import numpy as np
import matplotlib.pyplot as plt

# Parameters
ximax = 8
ximin = -ximax
Nsteps = 100*2*ximax
nu0 = 6.0

s = 2

def v(xi):
    return xi

def v2(xi):
    return v(xi + s) + v(xi - s)
