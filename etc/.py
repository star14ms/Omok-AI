import numpy as np
from matplotlib import pyplot

def line(*args):
    print("================", args)

line("np.array 분석 1")

a = np.array([[1, 2], 
              [3, 4], 
              [5, 6]]) # , 7
print(a.shape)
print(a[:, 0])

# line("np.array 분석 2")

line()

print([1, 2,])
print(sum([1, 2]))

line("딕셔너리 분석")

a = 1
b = 2
dic = {"a": a, "b": b}

print("a" in dic)
print(1 in dic)
print(1 in dic.values())

c, d = dic.values()
print(c, d)

line()

a = np.array([[1, 2, 6], 
              [3, 4, 5]])
print(a.sum(), sum(a)) # axis=0
print(a.argmax())

line()

