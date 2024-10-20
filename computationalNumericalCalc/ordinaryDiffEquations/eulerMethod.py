# Euler Method

from __future__ import division

import math

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

# EDO
y = 3
def f(t, u):
    return 2**math.cos(y)

    # tamanho e num. de passos

h = 0.1
N = 40

# cria vetor t e u
t = np.empty(N)
u = np.copy(t)

# C.I.
t[0] = 3
u[0] = 0.4

# iteracoes
for i in np.arange(N - 1):
    t[i + 1] = t[i] + h
    u[i + 1] = u[i] + h * f(t[i], u[i])

# imprime
for i, tt in enumerate(t):
    print("%1.1f %1.4f" % (t[i], u[i]))
