import numpy as np
import sys
import math

eps = sys.float_info.epsilon

print("Overflow")
print(sys.float_info.max)
print("Underflow")
print(sys.float_info.min)
print("Eps")
print(eps)
a = 2 + 1.e-16 == 2
print(a)
b = 1 + eps == 1
print(b)
d = (math.sqrt(5))**2
print("Pode ser considerado 0 pois eh semelhante ao EPS")
print(d)