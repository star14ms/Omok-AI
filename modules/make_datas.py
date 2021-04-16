import numpy as np
from modules.common.util import shuffle_dataset

class make_datas:

    def _4to5(one_hot_label=True, score=1, blank_score=0):
        xb_datas = np.full([2860, 1, 15, 15], blank_score, dtype=int)
        xw_datas = np.full([2860, 1, 15, 15], blank_score, dtype=int)
        t_datas = np.zeros([2860], dtype=int)
        # t2_datas = np.full([2860, 2], -1, dtype=int)
        N = 0
    
        for y in range(15): # [0:825] (가로)
            for x in range(15-4):
                for i in range(5):
                    xb_datas[N+i, 0, y, x:x+5] = score
                    xw_datas[N+i, 0, y, x:x+5] = -score
    
                    if x+5<15 and i==0:
                        # t2_datas[N+i][1] = y*15+(x+5)
                        xb_datas[N+i, 0, y, x+5] = -score
                        xw_datas[N+i, 0, y, x+5] = score
                    elif x-1>=0 and i==4:
                        # t2_datas[N+i][1] = y*15+(x-1)
                        xb_datas[N+i, 0, y, x-1] = -score
                        xw_datas[N+i, 0, y, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y, x+j] = blank_score
                    xw_datas[N+j, 0, y, x+j] = blank_score
                    t_datas[N+j] = y*15+(x+j)
                    # t2_datas[N+j][0] = y*15+(x+j)
                N += 5
        
        for x in range(15): # [825:1650] (세로)
            for y in range(15-4): 
                for i in range(5):
                    xb_datas[N+i, 0, y:y+5, x] = score
                    xw_datas[N+i, 0, y:y+5, x] = -score
    
                    if y+5<15 and i==0:
                        # t2_datas[N+i][1] = (y+5)*15+x
                        xb_datas[N+i, 0, y+5, x] = -score
                        xw_datas[N+i, 0, y+5, x] = score
                    elif y-1>=0 and i==4:
                        # t2_datas[N+i][1] = (y-1)*15+x
                        xb_datas[N+i, 0, y-1, x] = -score
                        xw_datas[N+i, 0, y-1, x] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+j, x] = blank_score
                    xw_datas[N+j, 0, y+j, x] = blank_score
                    t_datas[N+j] = (y+j)*15+x
                    # t2_datas[N+j][0] = (y+j)*15+x
                N += 5
        
        for y in range(15-4): # [1650:2255] (\대각선)
            for x in range(15-4):
                for i in range(5):
                    for i2 in range(5):
                        xb_datas[N+i, 0, y+i2, x+i2] = score
                        xw_datas[N+i, 0, y+i2, x+i2] = -score
    
                    if x+5<15 and y+5<15 and i==0:
                        # t2_datas[N+i][1] = (y+5)*15+(x+5)
                        xb_datas[N+i, 0, y+5, x+5] = -score
                        xw_datas[N+i, 0, y+5, x+5] = score
                    elif x-1>=0 and y-1>=0 and i==4:
                        # t2_datas[N+i][1] = (y-1)*15+(x-1)
                        xb_datas[N+i, 0, y-1, x-1] = -score
                        xw_datas[N+i, 0, y-1, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+j, x+j] = blank_score
                    xw_datas[N+j, 0, y+j, x+j] = blank_score
                    t_datas[N+j] = (y+j)*15+(x+j)
                    # t2_datas[N+j][0] = (y+j)*15+(x+j)
                N += 5
        
        for y in range(15-4): # [2255:2860] (/대각선)
            for x in range(15-4):
                for i in range(5):
                    for i2 in range(5):
                        xb_datas[N+i, 0, y+4-i2, x+i2] = score
                        xw_datas[N+i, 0, y+4-i2, x+i2] = -score
    
                    if x+5<15 and y-1>=0 and i==0:
                        # t2_datas[N+i][1] = (y-1)*15+(x+5)
                        xb_datas[N+i, 0, y-1, x+5] = -score
                        xw_datas[N+i, 0, y-1, x+5] = score
                    elif x-1>=0 and y+5<15 and i==4:
                        # t2_datas[N+i][1] = (y+5)*15+(x-1)  
                        xb_datas[N+i, 0, y+5, x-1] = -score
                        xw_datas[N+i, 0, y+5, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+4-j, x+j] = blank_score
                    xw_datas[N+j, 0, y+4-j, x+j] = blank_score
                    t_datas[N+j] = (y+4-j)*15+(x+j)
                    # t2_datas[N+j][0] = (y+4-j)*15+(x+j)
                N += 5
    
        if one_hot_label:
            t_datas = _change_one_hot_label(t_datas)
            # t2_datas = _change_some_hot_labels(t2_datas)

        x_datas = np.r_[xb_datas, xw_datas]
        t_datas = np.r_[t_datas, t_datas]
        # t2_datas = np.r_[t2_datas, t2_datas]

        return x_datas, t_datas # , t2_datas

    def last_one(one_hot_label=True, score=1, blank_score=0):
        x_datas = np.full([225*4, 225], score, dtype=int) # 900개
        t_datas = np.full([225*4], score, dtype=int)
        
        N = 0
        for blank_idx in range(225):
            for _ in range(4):
                white_mask = np.random.choice(225, 112)
                x_datas[N, white_mask] = -score # 두번째 차원 1 -> 0
                x_datas[N, blank_idx] = 0
                t_datas[N] = blank_idx
                N += 1

        x_datas = x_datas.reshape(225*4, 1, 15, 15)

        if one_hot_label:
            t_datas = _change_one_hot_label(t_datas)

        return x_datas, t_datas


class split_datas:

    def even_odd(xb_datas, t_datas):
        len_datas = xb_datas.shape[0]
    
        x_train, t_train = xb_datas[range(0, len_datas, 2)], t_datas[range(0, len_datas, 2)] ### 0, 1 X
        x_test, t_test = xb_datas[range(1, len_datas, 2)], t_datas[range(1, len_datas, 2)] ### 0, 1 X
    
        return (x_train, t_train), (x_test, t_test)

def _change_one_hot_label(X):
        T = np.zeros((X.size, 225))
        for idx, row in enumerate(T):
            row[X[idx]] = 1
    
        return T
    
def _change_some_hot_labels(X):
    T = np.zeros((X.shape[0], 225))
    for idx, row in enumerate(T):
        for answer in X[idx]:
            if answer == -1:
                continue
            row[answer] = 1 ### X[idx], X[answer], X[idx[answer]], idx[idx2] X / answer O

    return T
    
def sample_val(x_train, t_train, sample_size=500, validation_rate=0.2):
    x_train, t_train = shuffle_dataset(x_train, t_train)
    x_train = x_train[:sample_size]
    t_train = t_train[:sample_size]

    validation_num = int(x_train.shape[0] * validation_rate)
    x_val = x_train[:validation_num]
    t_val = t_train[:validation_num]
    x_train = x_train[validation_num:]
    t_train = t_train[validation_num:]
    
    return (x_train, t_train), (x_val, t_val)

# # 학습할 문제들 보기
# import random
# from test import print_board

# x_train, t_train = make_datas.last_one()
# print(x_train.shape, t_train.shape)

# idx = 0
# while True: ### 훈련/검증 데이터로 반으로 쪼개짐 5720 /2

#     idx += 1
#     # idx = random.randrange(0, t_train.shape[0]) ### 범위 밖으로 나감
#     # print_board(x_train[idx:idx+1], t_train[idx:idx+1], mode="QnA")
#     print()
#     print_board(x_train[idx:idx+1], mode="Q")
#     answer = input("")
#     if answer == "n":
#         break
