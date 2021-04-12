import numpy as np
from matplotlib import pyplot

a = np.array([[1, 2], 
              [3, 4], 
              [5, 6]]) # , 7
print(a.shape)
print(a[:, 0])

print([1, 2,])

print(sum([1, 2]))

print("a" in {"a":1, "b":2})
print(1 in {"a":1, "b":2})
print(1 in {"a":1, "b":2}.values())

a = np.array([[1, 2, 6], 
              [3, 4, 5]])
print(a.sum(), sum(a))
print(a.argmax())