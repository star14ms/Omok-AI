import numpy as np
from datetime import datetime # 기보 날짜 기록
from modules.common.util import rotate_2dim_array, rotate_2dim_array_idx
from modules.test import print_board
from modules.plot import *

# AI code (Human Made Algorithm)
from pygame_src.AI_code import *
from pygame_src.foul_detection import isFive
from modules.yixin import Yixin, Click

# AI Deep Learning network
from network import DeepConvNet
from modules.common.optimizer import *
from modules.make_datas import _change_one_hot_label
from modules.common.util import time

# saved_network_pkl = "with_myAI (x8x26) ln_1023800 acc_None Adam lr_0.01 CR_CR_CR_CR_A_Smloss params"
saved_network_pkl = "with_Yixin (X26f256) ln_157000 acc_None Adam lr_0.01 CR_CR_CR_CR_A_Smloss params"

saved_graphdata_pkl = saved_network_pkl.split('params')[0] + "learning_info" if saved_network_pkl != None else ""
if saved_graphdata_pkl != "":
    graph_datas = load_graph_datas(saved_graphdata_pkl)
#     if graph_datas != None:
#         plot.loss_graph(graph_datas['train_losses'], smooth=False)
#         plot.loss_graph(graph_datas['train_losses'], smooth=True)
#     else:
#         graph_datas = {'train_losses': []}
#     exit()

network = DeepConvNet(saved_network_pkl=saved_network_pkl)
str_data_info = ' '.join(saved_network_pkl.split(' ')[:2])
lr=0.01
optimizer = Adam(lr=lr)

train_with_Yixin = True
learning_num_per_save = 1000
goal_learning_num = 0
verbose = False

if train_with_Yixin: Yixin.init()

################################################################ main code

print("\n--Python 오목 AI 학습! (렌주룰)--")

exit=False # 프로그램 종료
size = 15

resent_who_win = []
save_num = network.learning_num // learning_num_per_save
str_turns = ""
train_losses = []

x_datas_left = np.empty((0, 15, 15), dtype=np.float16)
t_datas_left = np.empty((0, 1), dtype=int)
start_time = time.time()

while not exit:

    board = np.zeros([size, size], dtype=np.float16)
    # print_board(board, mode="Q")
    whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
    turn = 0
    clean_board_num = 0
    max_turn = size * size

    game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
    game_over = False # 게임이 끝났나?

    # record = [] # 기보 기록할 곳
    x_datas = np.empty((0, 15, 15), dtype=np.float16)
    t_datas = np.empty((0, 1), dtype=int)
    if train_with_Yixin: Yixin.reset() ### 디버깅 끝내고 다시 풀기

    while not game_end:

        # AI가 두기
        if not game_over:
            
            if turn == 0:
                x1, y1 = 7, 7
            elif train_with_Yixin and turn >= 3: # 알파오 Yixin (백)
                x1, y1 = Yixin.think_win_xy(whose_turn, undo=True if whose_turn == 1 else False)
                # print(f"{X_line[x1]}{Y_line[y1]} Yixin" if x1!=None else None)
            if not train_with_Yixin or x1==None or turn < 3: # 사람이 생각한 알고리즘 (백)
                x1, y1 = AI_think_win_xy(whose_turn, board, all_molds=True, verbose=False)
                # print(f"{X_line[x1]}{Y_line[y1]} myAI")
                if train_with_Yixin and turn == 2: Yixin.Click_setting("plays_w")
                if train_with_Yixin and whose_turn==-1:
                    if turn == 1:
                        Click(x1, y1, board_xy=True)
                    else:
                        Yixin.Click_setting("plays_w")
                        Click(x1, y1, board_xy=True)
                        Yixin.Click_setting("plays_w")
    
            if whose_turn != 1:
                x, y = x1, y1
                if train_with_Yixin and turn >= 3:
                    Yixin.Click_setting("plays_b")
                    Yixin.Click_setting("plays_w")
                    Click(0, 0) # 다음 흑이 둘 수 미리 생각하기 (Yixin)
            else: # 딥러닝 신경망 AI (흑)
                x2, y2 = network.think_win_xy(board, whose_turn, verbose=verbose)
                x, y = x2, y2
                if train_with_Yixin: Click(x2, y2, board_xy=True)

                x_datas = np.append(x_datas, board.reshape(1, 15, 15), axis=0)
                t_datas = np.append(t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
            
            # 선택한 좌표에 돌 두기
            board[y][x] = whose_turn
            # record.append([y, x, whose_turn])
            turn += 1
            
            # 오목이 생겼으면 게임 종료 신호 키기
            if isFive(whose_turn, board, x, y, placed=True) == True:
                game_over=True
                # exit=True
            
        # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
        if not game_over:
            whose_turn *= -1
        
            if turn >= max_turn:
                clean_board_num += 1
                turn = 0
                board = np.zeros([size, size], dtype=np.float16)
            
        # 흑,백 승리 이미지 화면에 추가, 학습하기, 기보 저장
        else:
            resent_who_win.append(whose_turn)
            if len(resent_who_win) > 100: del resent_who_win[0]

            str_turns = str_turns + ("" if str_turns=="" else ",") + str(clean_board_num*225 + turn)
            if train_with_Yixin: Yixin.Click_setting("plays_w" if whose_turn == 1 else "plays_b")

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
                graph_datas['train_losses'].append(loss)
                network.learning_num += 100

                print(time.str_hms_delta(start_time), end=" | ")
                print(f"{network.learning_num}문제 학습" + \
                    f" | 손실 {format(loss, '.3f')}", end=" | ")
                print(f"{str_turns}수" if len_x_datas == len(x_datas) else "")
                str_turns = ""

                x_datas = x_datas[100:]
                t_datas = t_datas[100:]
            
                if (network.learning_num // learning_num_per_save) != save_num:
                    win_num = sum(np.array(resent_who_win)==1)
                    game_num = len(resent_who_win)
                    print(f"-> 최근 {game_num}경기 중 승리 {win_num}경기 (승률: {round(win_num/game_num*100, 1)}%)")
                    save_num = network.learning_num // learning_num_per_save

                    network.save_params(optimizer, lr, str_data_info, verbose=False)
                    save_graph_datas(graph_datas, network, optimizer, lr, str_data_info, verbose=False)
                    network.delete_pre_saved_pkl(learning_num_per_save)
                    print()
            
            if network.learning_num == goal_learning_num: exit = True
            
            x_datas_left = x_datas
            t_datas_left = t_datas

            # 기보 파일로 저장
            # with open('etc/GiBo_Training.txt', 'a', encoding='utf8') as file:
                # file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                # for i in range(len(record)):
                #     turn_hangul = "흑" if record[i][2] == 1 else "백"
                #     file.write(str(record[i][0]+1)+' '+str(record[i][1]+1)+' '+turn_hangul+'\n')
                # file.write("\n")

            game_end = True

plot.loss_graph(graph_datas['train_losses'], smooth=True)
plot.loss_graph(graph_datas['train_losses'], smooth=False)