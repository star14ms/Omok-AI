import numpy as np
from network import DeepConvNet
from modules.common.trainer import Trainer, load_graph_datas
from modules.make_datas import make_datas, split_datas
import pickle
import math
import time
from matplotlib import pyplot
from modules.plot import plot_loss_graph, plot_accuracy_graph

# 알람 울리기, 테스트
import datetime as dt
from selenium import webdriver
from modules.test import print_board, test_pick, tests_random_pick
import random

# 학습할 데이터 만들기
# 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
x_datas, t_datas = make_datas._4to5(score=1, blank_score=0)
x_train, t_train, x_test, t_test = split_datas.even_odd(x_datas, t_datas)
mini_batch_size = 110

saved_network_pkl = None
saved_graphdata_pkl = None
train_network = False

# ================================ 주석을 풀면 신경망 불러오기
saved_network_pkl = "Momentum_lr=0.01_ln=28600_acc=99.93_params"
# ================================

# ================================ 주석을 풀면 지난 학습정보 그래프 데이터 불러오기 (그래프 출력용)
# saved_graphdata_pkl = "Momentum_lr=0.01_ln=28600_acc=99.93_graphdata"
# ================================

# ================================ # 주석을 풀면 학습 실행
# train_network = True
# ================================

# 신경망 생성
network = DeepConvNet(saved_network_pkl=saved_network_pkl, mini_batch_size=mini_batch_size)

# 학습
trainer = Trainer(epochs=10, optimizer='Momentum', optimizer_param={'lr':0.01}, 
    network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test,
    mini_batch_size=mini_batch_size)
if train_network:
    trainer.train()

# 학습된 신경망, 그래프 데이터 저장
if train_network:
    a = input("\n네트워크를 저장할거니? 예(any)/아니오(f) : ")
    if a != "f":
        network.save_params(trainer)
    else:
        print("저장 안했다^^")
    
    a = input("\n학습 정보를 저장할거니? (그래프 출력용) 예(any)/아니오(f) : ")
    if a != "f":
        trainer.save_graph_datas(network)
    else:
        print("저장 안했다^^")

# 그래프 출력
if saved_graphdata_pkl != None:
    graph_datas = load_graph_datas(saved_graphdata_pkl)
    plot_loss_graph(graph_datas["train_losses"], smooth=False)
    plot_accuracy_graph(graph_datas["train_accs"], graph_datas["test_accs"])
if train_network:
    plot_loss_graph(trainer.train_losses, smooth=False)
    plot_accuracy_graph(trainer.train_accs, trainer.test_accs)

# 정확도 구하고 틀린 문제 확인
accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True) # , multiple_answers=True
Fnum, Qnum = len(wrong_idxs), len(t_datas)
print(f"\n총 {Qnum} 문제 중, 정답 {Qnum-Fnum}개, 오답 {Fnum}개 (정답률: {math.floor(accuracy*100*100)/100}%)")

# input("\n틀린 문제들을 확인하려면 enter키를 눌러")
# for idx in wrong_idxs:
    # test_pick(network, x_datas, t_datas, idx)
    # answer = input()
    # if answer == "1":
    #     break

# 테스트
input("\n테스트를 시작하려면 enter키를 눌러")

tests_random_pick(network, x_datas, t_datas) # 4to5 (0, 825), (825, 1650), (1650, 2255), (2255, 2860)
# test_pick(network, x_datas, t_datas, 0)

# 학습할 문제들 보기
# while True:
    # idx = random.randrange(0, 2860) ### 5720 /2
    # print_board(x_train[idx:idx+1], t_train[idx:idx+1], mode="QnA")
    # answer = input()
    # if answer == "n":
    #     break

# # 각 pkl별 정확도 비교
# for i in range(4):
    # pkls_name = [
    #     "Adagrad lr_0.01 ln_14300 loss_7.1348",
    #     "Adagrad lr_0.01 ln_28600 loss_13.4896",
    #     "Adagrad lr_0.01 ln_42900 loss_16.0961",
    #     "Adagrad lr_0.01 ln_57200 loss_16.1181",
    #     ]
    # network.load_params(pkls_name[i]+".pkl")
    # accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True) # , multiple_answers=True
    # print(f"accuracy: {accuracy*100}%")

# 손실 함수가 높아지는 이유: 답이 2개이기 때문에 softmax확률이 5:5로 분배됨, 임의로 정답 라벨을 수정해버려서
# from modules.common.functions import softmax, cross_entropy_error
# x, t = x_datas[9:10], t_datas[9:10]

# y0 = network.predict(x)
# y = softmax(y0)
# loss = cross_entropy_error(y, t)

# np.set_printoptions(precision=5, suppress=True)
# print(y0.astype(np.int64).reshape(15, 15))
# print(y.astype(np.int64).reshape(15, 15))

# print_board(x, t, mode="QnA")
# print_board(x, y, mode="QnA_AI")
# print(loss)