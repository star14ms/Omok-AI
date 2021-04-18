import numpy as np
import sys
sys.path.append(".")
from modules.common.util import shuffle_dataset

class board_datas:
    
    def make_4to5(one_hot_label=True, score=1, blank_score=0): # 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
        xb_datas = np.full([2860, 1, 15, 15], blank_score, dtype=np.float16)
        xw_datas = np.full([2860, 1, 15, 15], blank_score, dtype=np.float16)
        t_datas = np.zeros([2860], dtype=int)
        N = 0
    
        for y in range(15): # [0:825] (가로)
            for x in range(15-4):
                for i in range(5):
                    xb_datas[N+i, 0, y, x:x+5] = score
                    xw_datas[N+i, 0, y, x:x+5] = -score
    
                    if x+5<15 and i==0:
                        xb_datas[N+i, 0, y, x+5] = -score
                        xw_datas[N+i, 0, y, x+5] = score
                    elif x-1>=0 and i==4:
                        xb_datas[N+i, 0, y, x-1] = -score
                        xw_datas[N+i, 0, y, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y, x+j] = blank_score
                    xw_datas[N+j, 0, y, x+j] = blank_score
                    t_datas[N+j] = y*15+(x+j)
                N += 5
        
        for x in range(15): # [825:1650] (세로)
            for y in range(15-4): 
                for i in range(5):
                    xb_datas[N+i, 0, y:y+5, x] = score
                    xw_datas[N+i, 0, y:y+5, x] = -score
    
                    if y+5<15 and i==0:
                        xb_datas[N+i, 0, y+5, x] = -score
                        xw_datas[N+i, 0, y+5, x] = score
                    elif y-1>=0 and i==4:
                        xb_datas[N+i, 0, y-1, x] = -score
                        xw_datas[N+i, 0, y-1, x] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+j, x] = blank_score
                    xw_datas[N+j, 0, y+j, x] = blank_score
                    t_datas[N+j] = (y+j)*15+x
                N += 5
        
        for y in range(15-4): # [1650:2255] (\대각선)
            for x in range(15-4):
                for i in range(5):
                    for i2 in range(5):
                        xb_datas[N+i, 0, y+i2, x+i2] = score
                        xw_datas[N+i, 0, y+i2, x+i2] = -score
    
                    if x+5<15 and y+5<15 and i==0:
                        xb_datas[N+i, 0, y+5, x+5] = -score
                        xw_datas[N+i, 0, y+5, x+5] = score
                    elif x-1>=0 and y-1>=0 and i==4:
                        xb_datas[N+i, 0, y-1, x-1] = -score
                        xw_datas[N+i, 0, y-1, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+j, x+j] = blank_score
                    xw_datas[N+j, 0, y+j, x+j] = blank_score
                    t_datas[N+j] = (y+j)*15+(x+j)
                N += 5
        
        for y in range(15-4): # [2255:2860] (/대각선)
            for x in range(15-4):
                for i in range(5):
                    for i2 in range(5):
                        xb_datas[N+i, 0, y+4-i2, x+i2] = score
                        xw_datas[N+i, 0, y+4-i2, x+i2] = -score
    
                    if x+5<15 and y-1>=0 and i==0:
                        xb_datas[N+i, 0, y-1, x+5] = -score
                        xw_datas[N+i, 0, y-1, x+5] = score
                    elif x-1>=0 and y+5<15 and i==4:
                        xb_datas[N+i, 0, y+5, x-1] = -score
                        xw_datas[N+i, 0, y+5, x-1] = score
    
                for j in range(5):
                    xb_datas[N+j, 0, y+4-j, x+j] = blank_score
                    xw_datas[N+j, 0, y+4-j, x+j] = blank_score
                    t_datas[N+j] = (y+4-j)*15+(x+j)
                N += 5
    
        if one_hot_label:
            t_datas = _change_one_hot_label(t_datas)

        x_datas = np.r_[xb_datas, xw_datas]
        t_datas = np.r_[t_datas, t_datas]

        return x_datas, t_datas

    def make_last_one(one_hot_label=True, score=1, blank_score=0):
        x_datas = np.full([225*4, 225], score, dtype=np.float16) # 900개
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
    
    def split_train_test(x_datas, t_datas, split_even_odd=False, test_rate=0.5):
        len_datas = x_datas.shape[0]
        
        if split_even_odd:
            x_train, x_test = x_datas[range(0, len_datas, 2)], x_datas[range(1, len_datas, 2)] ### 0, 0 X
            t_train, t_test = t_datas[range(0, len_datas, 2)], t_datas[range(1, len_datas, 2)] ### 1, 1 X
        else:
            split_idx = int(len_datas * test_rate)
            x_test, x_train = x_datas[:split_idx], x_datas[split_idx:]
            t_test, t_train = t_datas[:split_idx], t_datas[split_idx:]

        return (x_train, t_train), (x_test, t_test)
    
    def split_train_val(x_train, t_train, sample_size=500, validation_rate=0.2):
        x_train, t_train = shuffle_dataset(x_train, t_train)
        x_train = x_train[:sample_size]
        t_train = t_train[:sample_size]
    
        validation_num = int(x_train.shape[0] * validation_rate)
        x_val = x_train[:validation_num]
        t_val = t_train[:validation_num]
        x_train = x_train[validation_num:]
        t_train = t_train[validation_num:]
        
        return (x_train, t_train), (x_val, t_val)
    
    def merge(*args, shuffle=True):
        datas = {}
        x_datas = args[0][0]
        t_datas = args[0][1]
        # print(args[0][0].shape, args[0][1].shape)
        for i, datas in enumerate(args[1:]):
            # print(args[i+1][0].shape, args[i+1][1].shape)
            x_datas = np.r_[x_datas, args[i+1][0]]
            t_datas = np.r_[t_datas, args[i+1][1]]

        if shuffle:
            x_datas, t_datas = shuffle_dataset(x_datas, t_datas)

        return (x_datas, t_datas)
    
    def make_toNmok(Nmok, one_hot_label=True, shuffle=True, score=1, blank_score=0, size=15):
        #1    🟫🟣⚫⚫⚫🟤⚪
        #2    🟫⚫🟣⚫⚫🟤
        #3    🟫⚫⚫🟣⚫🟤
        #4  ⚪🟫⚫⚫⚫🟣🟤
        
        #1    🟪⚫⚫⚫⚫⚪
        #2    ⬛🟣⚫⚫⚫
        #3    ⬛⚫🟣⚫⚫
        #4    ⬛⚫⚫🟣⚫
        #5  ⚪⬛⚫⚫⚫🟣
                          
        scope = Nmok + 2*(5-Nmok)
        start = 5-Nmok
        op1_case, op2_case = 1, Nmok
        op1, op2 = scope, -1

        datas_num = (2*(2*size-scope+1))*(size-scope+1) * Nmok
        xb_datas = np.full([datas_num, 1, size, size], blank_score, dtype=np.float16)
        xw_datas = np.full([datas_num, 1, size, size], blank_score, dtype=np.float16)
        t_datas = np.zeros([datas_num], dtype=int)
        N = 0
        for y in range(size): # [0:600] (가로) (size*9*4)
            for x in range(size-scope+1):
                for i in range(Nmok):
                    xb_datas[N+i, 0, y, x+start:x+start+Nmok] = score
                    xw_datas[N+i, 0, y, x+start:x+start+Nmok] = -score
    
                    if x+op1<size and i==op1_case-1:
                        xb_datas[N+i, 0, y, x+op1] = -score
                        xw_datas[N+i, 0, y, x+op1] = score
                    elif x+op2>=0 and i==op2_case-1:
                        xb_datas[N+i, 0, y, x+op2] = -score
                        xw_datas[N+i, 0, y, x+op2] = score
    
                for j in range(Nmok):
                    xb_datas[N+j, 0, y, x+start+j] = blank_score
                    xw_datas[N+j, 0, y, x+start+j] = blank_score
                    t_datas[N+j] = y*size+(x+start+j)
                N += Nmok
        # print(N, end=", ")
        for x in range(size): # [600:1200] (세로) (size*9*4)
            for y in range(size-scope+1): 
                for i in range(Nmok):
                    xb_datas[N+i, 0, y+start:y+start+Nmok, x] = score
                    xw_datas[N+i, 0, y+start:y+start+Nmok, x] = -score
    
                    if y+op1<size and i==op1_case-1:
                        xb_datas[N+i, 0, y+op1, x] = -score
                        xw_datas[N+i, 0, y+op1, x] = score
                    elif y+op2>=0 and i==op2_case-1:
                        xb_datas[N+i, 0, y+op2, x] = -score
                        xw_datas[N+i, 0, y+op2, x] = score
    
                for j in range(Nmok):
                    xb_datas[N+j, 0, y+start+j, x] = blank_score
                    xw_datas[N+j, 0, y+start+j, x] = blank_score
                    t_datas[N+j] = (y+start+j)*size+x
                N += Nmok
        # print(N, end=", ")
        for y in range(size-scope+1): # [1200:1600] (\대각선) (9*9*4)
            for x in range(size-scope+1):
                for i in range(Nmok):
                    for i2 in range(Nmok):
                        xb_datas[N+i, 0, y+start+i2, x+start+i2] = score
                        xw_datas[N+i, 0, y+start+i2, x+start+i2] = -score
    
                    if x+op1<size and y+op1<size and i==op1_case-1:
                        xb_datas[N+i, 0, y+op1, x+op1] = -score
                        xw_datas[N+i, 0, y+op1, x+op1] = score
                    elif x+op2>=0 and y+op2>=0 and i==op2_case-1:
                        xb_datas[N+i, 0, y+op2, x+op2] = -score
                        xw_datas[N+i, 0, y+op2, x+op2] = score
    
                for j in range(Nmok):
                    xb_datas[N+j, 0, y+start+j, x+start+j] = blank_score
                    xw_datas[N+j, 0, y+start+j, x+start+j] = blank_score
                    t_datas[N+j] = (y+start+j)*size+(x+start+j)
                N += Nmok
        # print(N, end=", ")
        for y in range(size-scope+1): # [1404:1728] (/대각선) (9*9*4)
            for x in range(size-scope+1):
                for i in range(Nmok):
                    for i2 in range(Nmok):
                        # print(x, y)
                        xb_datas[N+i, 0, y+scope-1-start-i2, x+start+i2] = score # y+Nomk-1-i2 -> y+scope-1-start-i2
                        xw_datas[N+i, 0, y+scope-1-start-i2, x+start+i2] = -score
    
                    if x+op1<size and y+op2>=0 and i==op1_case-1:
                        xb_datas[N+i, 0, y+op2, x+op1] = -score
                        xw_datas[N+i, 0, y+op2, x+op1] = score
                    elif x+op2>=0 and y+op1<size and i==op2_case-1:
                        xb_datas[N+i, 0, y+op1, x+op2] = -score
                        xw_datas[N+i, 0, y+op1, x+op2] = score
    
                for j in range(Nmok):
                    xb_datas[N+j, 0, y+scope-1-start-j, x+start+j] = blank_score
                    xw_datas[N+j, 0, y+scope-1-start-j, x+start+j] = blank_score
                    t_datas[N+j] = (y+scope-1-start-j)*size+(x+start+j)
                N += Nmok
        # print(N)
        
        if one_hot_label:
            t_datas = _change_one_hot_label(t_datas)

        x_datas = np.r_[xb_datas, xw_datas]
        t_datas = np.r_[t_datas, t_datas]

        if shuffle:
            x_datas, t_datas = shuffle_dataset(x_datas, t_datas)

        return (x_datas, t_datas)

def _change_one_hot_label(X):
        T = np.zeros((X.size, 225), dtype=np.float16)
        for idx, row in enumerate(T):
            row[X[idx]] = 1
    
        return T
    
def _change_some_hot_labels(X):
    T = np.zeros((X.shape[0], 225), dtype=np.float16)
    for idx, row in enumerate(T):
        for answer in X[idx]:
            if answer == -1:
                continue
            row[answer] = 1 ### X[idx], X[answer], X[idx[answer]], idx[idx2] X / answer O

    return T
    
# # 학습할 문제들 보기
# import random
# from test import print_board

# datas_4to5 = board_datas.make_toNmok(Nmok=5, shuffle=False)
# datas_3to4 = board_datas.make_toNmok(Nmok=4, shuffle=False)
# (x_datas, t_datas) = board_datas.merge(datas_4to5, datas_3to4, shuffle=False)
# print(datas_4to5[1].shape)
# print(datas_3to4[1].shape)
# print(t_datas.shape)

# # (x_train, t_train), (x_test, t_test) = board_datas.split_train_test(x_datas, t_datas)
# # print(t_train.shape, t_test.shape)

# # idx = 1599 # 6920
# print()
# while True: ### 훈련/검증 데이터로 반으로 쪼개짐 5720 /2

#     idx = random.randrange(0, t_datas.shape[0]) ### 범위 밖으로 나감
#     # print_board(x_datas[idx:idx+1], t_datas[idx:idx+1], mode="QnA")
#     print(f"Q-{idx}")
#     print_board(x_datas[idx:idx+1], t_datas[idx:idx+1], mode="QnA")
#     answer = input()
#     # idx += 1
#     if answer == "n":
#         break
