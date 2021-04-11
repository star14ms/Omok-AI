import numpy as np
from learning.deep_convnet import DeepConvNet
from learning.make_datas import make_4to5_datas
from learning.common.trainer import Trainer
from learning.common.functions import softmax
from learning.common.functions import cross_entropy_error
# 알람 울리기
import datetime as dt
from selenium import webdriver
# 무작위 데이터 골라 테스트
import random
import time

# 학습할 데이터 만들기
x_datas, t_datas, t_datas_real = make_4to5_datas(score=1, blank_score=0)
len_datas = x_datas.shape[0] # 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
x_train, t_train = x_datas[range(0, len_datas, 2)], t_datas_real[range(0, len_datas, 2)] ### 0, 1 X
x_test, t_test = x_datas[range(1, len_datas, 2)], t_datas_real[range(1, len_datas, 2)] ### 0, 1 X

params_pkl_name = "Adagrad lr_0.01 ln_28600 loss_13.4896.pkl"

optimizer = 'Adagrad'
lr = 0.01
mini_batch_size = 110

# 신경망 생성
# network = DeepConvNet(
#     layers_info = [
#         'convEnd', 'Relu',
#         'softmax'
#     ], params = [(1, 15, 15),
#         # {'filter_num':10, 'filter_size':1, 'pad':0, 'stride':1},
#         # {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
#         # {'filter_num':10, 'filter_size':5, 'pad':2, 'stride':1},
#         # {'filter_num':10, 'filter_size':7, 'pad':3, 'stride':1},
#         {'filter_num':10, 'filter_size':9, 'pad':4, 'stride':1},
#     ], params_pkl_name=params_pkl_name, mini_batch_size=mini_batch_size)

# 학습
# trainer = Trainer(epochs=10, optimizer=optimizer, optimizer_param={'lr':lr}, verbose=True, 
#     mini_batch_size=mini_batch_size, give_up={'epoch':2, 'test_loss':1000},
#     network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
# trainer.train()
# network.save_params(f"{optimizer} lr_{lr} ln_{network.learning_num} loss_{trainer.loss}.pkl")

# 테스트
np.set_printoptions(precision=2, suppress=True)

def first_place_yx(array2dim, find_all=False):
    if not find_all: # return y, x
        return array2dim.argmax()//15, array2dim.argmax()%15

    first_place_indexs = np.argwhere(array2dim == np.amax(array2dim)).tolist()
    first_place_yx_list = []
    # print(first_place_indexs)

    for fp_index in first_place_indexs:
        if fp_index not in first_place_yx_list:
            first_place_yx_list.append(fp_index)

    return first_place_yx_list # return [[y1, x1], [y2, x2], ...]

def print_board(Q, A, mode="A", num_answers=1):
    len_y, len_x = Q[0, 0].shape
    if mode=="A":
        a_y, a_x = first_place_yx(A)

    for y in range(len_y):
        for x in range(len_x):

            if Q[0, 0][y][x] == 0:
                if mode=="Q":
                    if A[y][x] == 0:
                        print("🟤", end="")
                    else:
                        print("🟣", end="")
                else:
                    chance = A[y][x]*num_answers
                    if 0 < chance < 5:
                        print("🔴", end="") if x!=a_x or y!=a_y else print("🟥", end="")
                    elif 5 <= chance < 20:
                        print("🟠", end="") if x!=a_x or y!=a_y else print("🟧", end="") 
                    elif 20 <= chance < 50:
                        print("🟡", end="") if x!=a_x or y!=a_y else print("🟨", end="")
                    elif 50 <= chance < 80:
                        print("🟢", end="") if x!=a_x or y!=a_y else print("🟩", end="")
                    elif 80 <= chance < 95:
                        print("🔵", end="") if x!=a_x or y!=a_y else print("🟦", end="")
                    elif 95 <= chance:
                        print("🟣", end="") if x!=a_x or y!=a_y else print("🟪", end="")
                    else:
                        print("🟤", end="")
 
            elif Q[0, 0][y][x] == 1:
                print("⚫", end="")
            elif Q[0, 0][y][x] == -1:
                print("⚪", end="")
            else:
                print("\nprint_question_board() 오류!")
        print()

    if mode == "A":
        print("\n🟤 < 0.5% < 🔴 < 5% < 🟠 < 20% < 🟡 < 50% < 🟢 < 80% < 🔵 < 95% < 🟣\n")

def test(x_data, t_data, index):
    x = x_data[index:index+1]
    t = t_data[index:index+1].reshape(15, 15)

    score_board = network.predict(x)
    score_board_reshape = score_board.reshape(15, 15)

    winning_chance = softmax(score_board) * 100
    winning_chance = winning_chance.reshape(15, 15) ### score_board_reshape.round(2) np.round_(winning_chance, 2)

    a_y, a_x = first_place_yx(winning_chance)
    t_yxs = first_place_yx(t, find_all=True)
    
    # winning_chance = np.where(winning_chance==winning_chance[a_y, a_x], winning_chance, 0)
    
    winning_chances = []
    for idx, _ in enumerate(t_yxs):
        chance = winning_chance[t_yxs[idx][0], t_yxs[idx][1]].round(1)
        winning_chances.append(chance)

    print("\n=== 점수(scores) ===")
    print(score_board_reshape.astype(np.int64))
    print("\n=== 각 자리의 확률 분배(%) ===")
    print(winning_chance.astype(np.int64))
    print(f"\n=== Question_{index} ===")
    print_board(x, t, "Q")
    print("\n=== AI's Answer === ")
    print_board(x, winning_chance.round(0), "A", len(t_yxs))

    print("정답 좌표: ", end="")
    for t_yx, t_yx_chance in zip(t_yxs, winning_chances):
        print(f"{t_yx} ({t_yx_chance}%)", end=" / ")
    print(f"\n구한 좌표: [{a_y}, {a_x}] ({winning_chance[a_y, a_x].round(1)}%)", end=" / ")
    print("정답!" if [a_y, a_x] in t_yxs else "응 아니야~")

# input("테스트를 시작하려면 enter키를 눌러")

# # 손실 함수가 높아지는 이유: 답이 2개이기 때문에 softmax확률이 5:5로 분배됨
# y = network.predict(x_datas[0:1])
# c = softmax(y)
# loss = cross_entropy_error(c, t_datas[0:1])
# print(c)
# print(loss)

# # 각 pkl별 정확도 비교
# for i in range(4):
    # pkls_name = [
    #     "Adagrad lr_0.01 ln_14300 loss_7.1348",
    #     "Adagrad lr_0.01 ln_28600 loss_13.4896",
    #     "Adagrad lr_0.01 ln_42900 loss_16.0961",
    #     "Adagrad lr_0.01 ln_57200 loss_16.1181",
    #     ]
    # network.load_params(pkls_name[i]+".pkl")
    # accuracy, wrong_idxs = network.accuracy(x_datas, t_datas_real, save_wrong_idxs=True) # , multiple_answers=True
    # print(f"accuracy: {accuracy*100}%")

# # 틀린 문제 확인
# _, wrong_idxs = network.accuracy(x_datas, t_datas_real, save_wrong_idxs=True) # , multiple_answers=True
# for idx in wrong_idxs:
    # test(x_datas, t_datas_real, idx)
    # input()

# # 랜덤 테스트
# while True: # (0, 825), (825, 1650), (1650, 2255), (2255, 2860)
    # test(x_datas, t_datas_real, random.randrange(0, 2860))  
    # answer = input()
    # if answer == "1":
    #     break
