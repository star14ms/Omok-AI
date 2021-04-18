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

line("np.array sum, argmax")

a = np.array([[1, 2, 6], 
              [3, 4, 5]], dtype=int)
print(a.sum(), sum(a)) # axis=0
print(a.argmax())
print((a * 0.5).dtype)
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

line("줄 번호 출력하기")

import logging
 
# logging.basicConfig(format='\n"%(pathname)s", line %(lineno)d, in %(module)s\n%(levelname)-8s: %(message)s')
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%d-%m-%Y:%H:%M:%S',
#     level=logging.INFO)

# logging.basicConfig(format="asctime: %(asctime)s\ncreated: %(created)f\nfilename: %(filename)s\nfuncName: %(funcName)s\n" + \
#     "levelname: %(levelname)s\nlevelno: %(levelno)s\nlineno: %(lineno)d\nmessage: %(message)s\n" + \
#     "module: %(module)s\nmsecs: %(msecs)d\nname: %(name)s\npathname: %(pathname)s\nprocess: %(process)d\n" + \
#     "processName: %(processName)s\nrelativeCreated: %(relativeCreated)d\nthread: %(thread)d\nthreadName: %(threadName)s", 
#     level=logging.DEBUG)
# logger = logging.getLogger('log_1')
# logger.error("error")
    
# logger.debug("This is a debug log")
# logger.info("This is an info log")
# logger.critical("This is critical")
# logger.error("An error occurred")

def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # 로그 포멧 정의
    formatter = logging.Formatter(
        '\n"%(pathname)s", line %(lineno)d, in %(module)s\n%(levelname)-8s: %(message)s')
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger

logger = __get_logger()

# 원하는 위치에 로그 메시지 작성 - info level로 아래의 메시지를 출력하겠다.
logger.info("안녕")

line("할당과 복사")

a = np.array([1, 2])
b = a.copy()
a[0] = 3
print(a, b)