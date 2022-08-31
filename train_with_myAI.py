import numpy as np # 보드 만들기
import random 
from datetime import datetime # 기보 날짜 기록
from modules.common.util import bcolors, rotate_2dim_array, rotate_2dim_array_idx
from modules.test import print_board

# AI code (Human Made Algorithm)
from pygame_src.AI_code import *
from pygame_src.foul_detection import isFive, num_Four, num_Three

# AI Deep Learning network
from network import DeepConvNet
from modules.common.optimizer import *
from modules.make_datas import _change_one_hot_label
import time as t # 트레이닝 돌 두는 시간 텀
from modules.common.util import time

# network = DeepConvNet(saved_network_pkl="training with myAI (1) acc_None ln_85034 Adam lr_0.01 CR_CR_CR_CR_ANR_A_Smloss params")
network = DeepConvNet(saved_network_pkl="training with myAI (1) acc_None ln_195030 Adam lr_0.01 CR_CR_CR_CR_A_Smloss params")
str_data_info = "training with myAI (1)"
lr=0.01
optimizer = Adam(lr=lr)
plot_distribution = False

################################################################ main code

print("\n--Python 오목 AI 학습! (렌주룰)--")

exit=False # 프로그램 종료

size = 15
win_num = 0
game_trial = 0
game_trial_per_ln_5000 = 0
learning_5000_num = network.learning_num // 5000
str_turns = ""

x_datas = np.empty((0, 15, 15), dtype=np.float16)
t_datas = np.empty((0, 1), dtype=int)
x_datas_left = np.empty((0, 15, 15), dtype=np.float16)
t_datas_left = np.empty((0, 1), dtype=int)
start_time = t.time()

while not exit:

    board = np.zeros([size, size], dtype=np.float16)
    whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
    turn = 0
    clean_board_num = 0
    max_turn = size * size

    game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
    game_over = False # 게임이 끝났나?

    # record = [] # 기보 기록할 곳
    x_datas_necessary = np.empty((0, 15, 15), dtype=np.float16)
    t_datas_necessary = np.empty((0, 1), dtype=int)

    while not game_end:

        # AI가 두기
        if not game_over:
            
            # 사람이 생각한 알고리즘 (백)
            x1, y1, is_necessary = AI_think_win_xy(whose_turn, board, all_molds=True, verbose=False, return_is_necessary=True)
            if whose_turn != 1:
                x, y = x1, y1
            else: # 딥러닝 신경망 AI (흑)
                x2, y2 = network.think_win_xy(board, whose_turn, verbose=False)
                x, y = x2, y2

                x_datas = np.append(x_datas, board.reshape(1, 15, 15), axis=0)
                t_datas = np.append(t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
                # if len(x_datas) > 3:
                #     x_datas = np.delete(x_datas, 0, axis=0)
                #     t_datas = np.delete(t_datas, 0, axis=0)
            
            # # 둘 곳이 명백할 때, 마지막 4수의 보드 상태와 내 AI답 저장 (학습할 문제와 답 저장)
            # if is_necessary:
            #     x_datas_necessary = np.append(x_datas_necessary, board.reshape(1, 15, 15), axis=0)
            #     t_datas_necessary = np.append(t_datas_necessary, np.array([[y1*15+x1]], dtype=int), axis=0)
            # elif whose_turn == 1:
            #     x_datas = np.append(x_datas, board.reshape(1, 15, 15), axis=0)
            #     t_datas = np.append(t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
            #     if len(x_datas) > 3:
            #         x_datas = np.delete(x_datas, 0, axis=0)
            #         t_datas = np.delete(t_datas, 0, axis=0)

            # 선택한 좌표에 돌 두기
            board[y][x] = whose_turn
            # record.append([y, x, whose_turn])
            turn += 1
        
            # 오목이 생겼으면 게임 종료 신호 키기
            if isFive(whose_turn, board, x, y, placed=True) == True:
                game_over=True
            
        # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
        if not game_over:
            whose_turn *= -1
        
            if turn >= max_turn:
                clean_board_num += 1
                turn = 0
                board = np.zeros([size, size])
            
        # 흑,백 승리 이미지 화면에 추가, 학습하기, 기보 저장
        else:
            game_trial += 1
            game_trial_per_ln_5000 += 1
            if whose_turn == 1: win_num += 1
            str_turns = str_turns + ("" if str_turns=="" else ",") + str(clean_board_num*225 + turn)

            # x_datas = np.r_[x_datas_necessary, x_datas]
            # t_datas = np.r_[t_datas_necessary, t_datas]

            # 데이터 양을 8배로 늘리기 (보드 회전(*4), 반전(*2))
            for x_data, t_data in zip(x_datas, t_datas):
                for d in range(1, 8): # -7~0 = 0~7
                    x_datas = np.append(x_datas, rotate_2dim_array(x_data, d), axis=0)
                    t_datas = np.append(t_datas, rotate_2dim_array_idx(t_data[0], d), axis=0)
                    # print_board(x_datas[-1], t_datas[-1], mode="QnA")
                    # input()

            x_datas = np.r_[x_datas_left, x_datas]
            t_datas = np.r_[t_datas_left, t_datas]
            len_x_datas = len(x_datas)

            # 데이터를 100개씩 나누어 학습
            while len(x_datas) >= 100:
                x_batch = x_datas[:100].reshape(100, 1, 15, 15)
                t_batch = _change_one_hot_label(t_datas[:100])
                grads, loss = network.gradient(x_batch, t_batch, save_loss=True)
                optimizer.update(network.params, grads)
                network.learning_num += 100

                print(time.str_hms_delta(start_time, hms=True), end=" | ")
                print(f"{network.learning_num}문제 학습" + \
                    f" | 손실 {format(loss, '.3f')}", end=" | ")
                print(f"{str_turns}수" if len_x_datas == len(x_datas) else "")
                str_turns = ""

                x_datas = x_datas[100:]
                t_datas = t_datas[100:]
            
                if (network.learning_num // 5000) != learning_5000_num:
                    print(f"-> 최근 {game_trial_per_ln_5000}경기 중 승리 {win_num}경기 (승률: {round(win_num/game_trial_per_ln_5000*100, 1)}%)")
                    learning_5000_num = network.learning_num // 5000
                    network.save_params(optimizer, lr, str_data_info=str_data_info)
                    game_trial_per_ln_5000, win_num = 0, 0
                    print()
                    
            x_datas_left = x_datas
            t_datas_left = t_datas
            x_datas = np.empty((0, 15, 15), dtype=np.float16)
            t_datas = np.empty((0, 1), dtype=int)

            # 기보 파일로 저장
            # with open('etc/GiBo_Training.txt', 'a', encoding='utf8') as file:
                # file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                # for i in range(len(record)):
                #     turn_hangul = "흑" if record[i][2] == 1 else "백"
                #     file.write(str(record[i][0]+1)+' '+str(record[i][1]+1)+' '+turn_hangul+'\n')
                # file.write("\n")

            game_end = True
