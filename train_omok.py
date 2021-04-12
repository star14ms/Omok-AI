import numpy as np
from network import DeepConvNet
from modules.common.trainer import Trainer, load_graph_datas
from modules.make_datas import make_datas, split_datas
from modules.plot import plot_loss_graph, plot_accuracy_graph
import pickle
import math
from matplotlib import pyplot
# 알람 울리기
import datetime as dt
from selenium import webdriver
# 무작위 데이터 골라 테스트
from modules.test import first_place_yx, print_board, test
import random
import time

# 학습할 데이터 만들기
# 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
x_datas, t_datas, t_datas_real = make_datas._4to5(score=1, blank_score=0)
x_train, t_train, x_test, t_test = split_datas.even_odd(x_datas, t_datas) 

optimizer = 'Momentum'
lr = 0.01
mini_batch_size = 110
params_pkl_file = None

# ================================ # 주석을 풀면 학습된 신경망 불러오기
params_pkl_file = "Momentum_lr=0.01_ln=21450_acc=100.0_params"
# ================================

# 신경망 생성
network = DeepConvNet(params_pkl_file=params_pkl_file, mini_batch_size=mini_batch_size)

# # 학습
# trainer = Trainer(epochs=5, optimizer=optimizer, optimizer_param={'lr':lr}, verbose=True, 
#     mini_batch_size=mini_batch_size, give_up={'epoch': 11},
#     network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
# trainer.train()

# # # 학습된 신경망, 그래프 데이터 저장
# network.save_params(trainer)
# trainer.save_graph_datas(network)

# 그래프 출력
graph_datas = load_graph_datas("Momentum_lr=0.01_ln=21450_acc=100.0_graphdata.pkl")
plot_loss_graph(graph_datas["train_losses"], smooth=False, ylim=2)
plot_accuracy_graph(graph_datas["train_accs"], graph_datas["test_accs"])

# 틀린 문제 확인
accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True) # , multiple_answers=True
Fnum, Qnum = len(wrong_idxs), len(t_datas)
print(f"\n총 {Qnum} 문제 중, 정답 {Qnum-Fnum}개, 오답 {Fnum}개 (정답률: {math.floor(accuracy*10000)/100}%)")

input("\n틀린 문제들을 확인하려면 enter키를 눌러")
for idx in wrong_idxs:
    test(network, x_datas, t_datas, idx)
    answer = input()
    if answer == "1":
        break

# 테스트
input("테스트를 시작하려면 enter키를 눌러")
# np.set_printoptions(precision=2, suppress=True)

# 랜덤 테스트
while True: # 4to5 (0, 825), (825, 1650), (1650, 2255), (2255, 2860)
    test(network, x_datas, t_datas, random.randrange(0, 2860))
    answer = input()
    if answer == "1":
        break


# # 각 pkl별 정확도 비교
# for i in range(4):
#     pkls_name = [
#         "Adagrad lr_0.01 ln_14300 loss_7.1348",
#         "Adagrad lr_0.01 ln_28600 loss_13.4896",
#         "Adagrad lr_0.01 ln_42900 loss_16.0961",
#         "Adagrad lr_0.01 ln_57200 loss_16.1181",
#         ]
#     network.load_params(pkls_name[i]+".pkl")
#     accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True) # , multiple_answers=True
#     print(f"accuracy: {accuracy*100}%")


# # 손실 함수가 높아지는 이유: 답이 2개이기 때문에 softmax확률이 5:5로 분배됨, 임의로 정답 라벨을 수정해버려서
# from modules.common.functions import softmax, cross_entropy_error
# Q_x, t = x_datas[9:10], t_datas[9:10]

# print_board(Q_x, t.reshape(15, 15), "Q")
# x = network.predict(Q_x)
# y = softmax(x)

# if y.ndim == 1:
#     t = t.reshape(1, t.size)
#     y = y.reshape(1, y.size)

# if t.size == y.size:
#     t = t.argmax(axis=1)
# t = 1 ### 임의로 정답 라벨을 수정해버려서 손실 함수가 높게 나옴
# print(t, y.reshape(15, 15))
# batch_size = y.shape[0]

# loss = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
# print(loss)