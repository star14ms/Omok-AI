import numpy as np
from network import DeepConvNet
from modules.common.trainer import Trainer, load_graph_datas
from modules.make_datas import make_datas, split_datas
import time
from modules.plot import plot

# 알람 울리기, 테스트
import datetime as dt
from selenium import webdriver
from modules.test import print_board, test_pick, test_random_picks, test_right_or_wrong_answers
from modules.common.util import bcolors
import random

# 학습할 데이터 만들기
# 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
x_datas, t_datas = make_datas._4to5(score=1, blank_score=0)
x_train, t_train, x_test, t_test = split_datas.even_odd(x_datas, t_datas)
mini_batch_size = 110 ### 다른 미니배치 수로 학습한 신경망을 불러오면 바뀌어버림

train_network = False
saved_network_pkl = None
saved_graphdata_pkl = None

# ================================ < 주석을 풀면 학습 실행
# train_network = True 
# ================================ v 주석을 풀면 신경망 불러오기
# saved_network_pkl = "CR_CR_CR_CsumR_Smloss Momentum lr=0.01 ln=28600 acc=99.93 params"
saved_graphdata_pkl = "CR_CR_CR_CsumR_N0sloss Momentum lr=0.01 ln=28600 acc=0.06 graphdata"
# ================================ ^ 주석을 풀면 지난 학습정보 그래프 데이터 불러오기 (그래프 출력용)

# 신경망 생성
network = DeepConvNet(saved_network_pkl=saved_network_pkl, mini_batch_size=mini_batch_size)

# 학습
trainer = Trainer(epochs=10, optimizer='Momentum', optimizer_param={'lr':0.01}, 
    network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test,
    mini_batch_size=mini_batch_size)
if train_network:
    start = time.time()
    trainer.train()

    # 소요 시간 출력
    time_delta = int(time.time() - start)
    h, m, s = (time_delta // 3600), (time_delta//60 - time_delta//3600*60), (time_delta % 60)
    print(f"\n{h}h {m}m {s}s") 

# 학습된 신경망, 그래프 데이터 저장
if train_network:
    a = input("\n네트워크를 저장할거니? 예(any)/아니오(f) : ")
    if a != "f" and a != "ㄹ":
        network.save_params(trainer)
    else:
        print("저장 안했다^^")
        
    a = input("\n학습 정보를 저장할거니? (그래프 출력용) 예(any)/아니오(f) : ")
    if a != "f":
        trainer.save_graph_datas(network)
    else:
        print("저장 안했다^^")

# 그래프 출력
if train_network:
    plot.loss_graph(trainer.train_losses, smooth=False)
    plot.accuracy_graph(trainer.train_accs, trainer.test_accs)
if saved_graphdata_pkl != None:
    graph_datas = load_graph_datas(saved_graphdata_pkl)
    plot.loss_graph(graph_datas["train_losses"], smooth=False)
    plot.accuracy_graph(graph_datas["train_accs"], graph_datas["test_accs"])

np.set_printoptions(linewidth=np.inf, formatter={'all':lambda x: ( # precision=0, suppress=True, 
    bcolors.according_to(x) + (" " if x<10 else "") + str(int(x)) + bcolors.ENDC)})
bcolors.ANSI_codes()
# 정확도 구하고 맞거나 틀린 문제 확인
# accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True, verbose=True) # , multiple_answers=True
# test_right_or_wrong_answers(network, x_datas, t_datas, wrong_idxs)

# 테스트
test_random_picks(network, x_datas, t_datas) # 4to5 (0, 825), (825, 1650), (1650, 2255), (2255, 2860)
# test_pick(network, x_datas, t_datas, 0)

# # 각 pkl별 정확도 비교
# pkls_name = [
    # "Adagrad lr_0.01 ln_14300 loss_7.1348",
    # "Adagrad lr_0.01 ln_28600 loss_13.4896",
    # "Adagrad lr_0.01 ln_42900 loss_16.0961",
    # "Adagrad lr_0.01 ln_57200 loss_16.1181",
    # ]
# for i in range(4):
    # network.load_params(pkls_name[i])
    # accuracy, wrong_idxs = network.accuracy(x_datas, t_datas, save_wrong_idxs=True)
    # print(f"accuracy: {accuracy}%")

# # 손실 함수가 높아지는 이유: 답이 2개이기 때문에 softmax확률이 5:5로 분배됨, 임의로 정답 라벨을 수정해버려서
# from modules.common.functions import softmax, cross_entropy_error
# from modules.common.layers import Not0SamplingLoss
# idx, num = 0, 2
# x, t = x_datas[idx:idx+num], t_datas[idx:idx+num]

# y0 = network.predict(x)
# # y = softmax(y0)
# # loss = cross_entropy_error(y, t)

# print(x.shape, t.shape, y0.shape)

# # W = 0.01 * np.random.randn(225, 225).astype('f')
# layer = Not0SamplingLoss(not0_num=4)
# loss = layer.forward(y0, t, x)

# # print(y0.astype(np.int64).reshape(15, 15))
# # print(y.astype(np.int64).reshape(15, 15))

# # print_board(x, t, mode="QnA")
# # print_board(x, y, mode="QnA_AI")
# print(loss)
