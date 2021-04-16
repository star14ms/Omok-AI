import numpy as np
from matplotlib import pyplot

def line(*args):
    print("================", args)

line("np.array 분석 1")

a = np.array([[10, 20], 
              [30, 40], 
              [50, 60]]) # , 7
print(a.shape)
print(a[:, 0])
print(np.delete(a, 0, 0))
print(a[[range(a.shape[0])], [1, 0, 1]])

line("np.array 분석 2")

# x = ([0,4,1], [3,2,4])
# dW = np.zeros([5,6])

# np.add.at(dW,x,1)
# print(dW)

b = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]) # , 7
c = (b[:,0], b[:,1])
np.add.at(a, c, 1)
print(c)
print(a)

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

a, b = 1, 2
print(a, b)

a, b = b, a
print(a, b)

print(int(0.9999999999999999))
print(int(0.99999999999999999))
print(0.9e-16)
# import sys
# for ID in range(108):
#     print(f'\u001b[{ID}mHello\u001b[0m', end=f"{ID} ")
# print()
# for ID in range(257):
#     print(f'\u001b[38;5;{ID}mHello\u001b[0m', end=f"{ID} ")
import numpy as np
from matplotlib import pyplot

def line(*args):
    print("================", args)

line("np.array 분석 1")

a = np.array([[10, 20], 
              [30, 40], 
              [50, 60]]) # , 7
print(a.shape)
print(a[:, 0])
print(np.delete(a, 0, 0))
print(a[[range(a.shape[0])], [1, 0, 1]])

line("np.array 분석 2")

# x = ([0,4,1], [3,2,4])
# dW = np.zeros([5,6])

# np.add.at(dW,x,1)
# print(dW)

b = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]) # , 7
c = (b[:,0], b[:,1])
np.add.at(a, c, 1)
print(c)
print(a)

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
print(len(dic))

line()

a = np.array([[1, 2, 6], 
              [3, 4, 5]])
print(a.sum(), sum(a)) # axis=0
print(a.argmax())

line()

a, b = 1, 2
print(a, b)

a, b = b, a
print(a, b)

print(int(0.9999999999999999)) # 16
print(int(0.99999999999999999)) # 17

# import sys
# for ID in range(108):
#     print(f'\u001b[{ID}mHello\u001b[0m', end=f"{ID} ")
# print()
# for ID in range(257):
#     print(f'\u001b[38;5;{ID}mHello\u001b[0m', end=f"{ID} ")

line("±오차(한계) 구하기")

val_accs = [1, 2, 3, 4, 5, 6, 7]

average_acc = sum(val_accs) / len(val_accs)
standard_deviation = ( sum( (np.array(val_accs)-average_acc)**2 ) / len(val_accs) ) ** (1/2) ### 
standard_error = standard_deviation / (len(val_accs) ** (1/2))
margin_of_error99 = round(3.3 * standard_error, 2)

print(average_acc, standard_deviation, standard_error, margin_of_error99)

line("tuple")

a = (1, 2)
print(a[1])