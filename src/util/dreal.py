import numpy as np
import dreal as d


def dreal_var(n, name='x'):
    return np.array([d.Variable(f"{name}{i}") for i in range(n)])

def dreal_elementwise(x, func):
    return np.array([func(xi) for xi in x])

def dreal_sigmoid(x):
    return 1 / (1 + d.exp(-x))
