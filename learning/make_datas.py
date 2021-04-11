import numpy as np

def make_4to5_datas(one_hot_label=True, score=1, blank_score=0):
    x_datas = np.full([2860, 1, 15, 15], blank_score, dtype=int)
    t_datas = np.zeros([2860], dtype=int)
    t_datas_real = np.full([2860, 2], -1, dtype=int)
    N = 0 

    for y in range(15): # [0:825] (가로)
        for x in range(15-4):
            for i in range(5):
                x_datas[N+i, 0, y, x:x+5] = score

                if x+5<15 and i==0:
                    t_datas_real[N+i][1] = y*15+(x+5)
                elif x-1>=0 and i==4:
                    t_datas_real[N+i][1] = y*15+(x-1)

            for j in range(5):
                x_datas[N+j, 0, y, x+j] = blank_score
                t_datas[N+j] = y*15+(x+j)
                t_datas_real[N+j][0] = y*15+(x+j)
            N += 5
    
    for x in range(15): # [825:1650] (세로)
        for y in range(15-4): 
            for i in range(5):
                x_datas[N+i, 0, y:y+5, x] = score

                if y+5<15 and i==0:
                    t_datas_real[N+i][1] = (y+5)*15+x
                elif y-1>=0 and i==4:
                    t_datas_real[N+i][1] = (y-1)*15+x

            for j in range(5):
                x_datas[N+j, 0, y+j, x] = blank_score
                t_datas[N+j] = (y+j)*15+x
                t_datas_real[N+j][0] = (y+j)*15+x
            N += 5
    
    for y in range(15-4): # [1650:2255] (\대각선)
        for x in range(15-4):
            for i in range(5):
                for i2 in range(5):
                    x_datas[N+i, 0, y+i2, x+i2] = score

                if x+5<15 and y+5<15 and i==0:
                    t_datas_real[N+i][1] = (y+5)*15+(x+5)
                elif x-1>=0 and y-1>=0 and i==4:
                    t_datas_real[N+i][1] = (y-1)*15+(x-1)

            for j in range(5):
                x_datas[N+j, 0, y+j, x+j] = blank_score
                t_datas[N+j] = (y+j)*15+(x+j)
                t_datas_real[N+j][0] = (y+j)*15+(x+j)
            N += 5
    
    for y in range(15-4): # [2255:2860] (/대각선)
        for x in range(15-4):
            for i in range(5):
                for i2 in range(5):
                    x_datas[N+i, 0, y+4-i2, x+i2] = score

                if x+5<15 and y-1>=0 and i==0:
                    t_datas_real[N+i][1] = (y-1)*15+(x+5)
                elif x-1>=0 and y+5<15 and i==4:
                    t_datas_real[N+i][1] = (y+5)*15+(x-1)  

            for j in range(5):
                x_datas[N+j, 0, y+4-j, x+j] = blank_score
                t_datas[N+j] = (y+4-j)*15+(x+j)
                t_datas_real[N+j][0] = (y+4-j)*15+(x+j)
            N += 5

    if one_hot_label:
        t_datas = _change_one_hot_label(t_datas)
        t_datas_real = _change_some_hot_label(t_datas_real)

    return x_datas, t_datas, t_datas_real

def _change_one_hot_label(X):
    T = np.zeros((X.size, 225))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

def _change_some_hot_label(X):
    T = np.zeros((X.shape[0], 225))
    for idx, row in enumerate(T):
        for answer in X[idx]:
            if answer == -1:
                continue
            row[answer] = 1 ### X[idx], X[answer], X[idx[answer]], idx[idx2] X / answer O

    return T
    