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


class Omok_AI:
    def __init__(self, saved_network_pkl: str, str_data_info: str) -> None: 
        self.start_time = time.time()
        self.network = DeepConvNet(saved_network_pkl=saved_network_pkl)
        self.lr=0.01
        self.optimizer = Adam(lr=self.lr)
        self.learning_num_per_save = 1000
        self.goal_learning_num = 100
        self.resent_who_win = []

        self.save_num = self.network.learning_num // self.learning_num_per_save

        assert self.goal_learning_num > 0

        self.x_datas_left = np.empty((0, 15, 15), dtype=np.float16)
        self.t_datas_left = np.empty((0, 1), dtype=int)

        saved_graphdata_pkl = saved_network_pkl.split('params')[0] + "learning_info" if saved_network_pkl != None else ""
        graph_datas = {'train_losses': []}
        if saved_graphdata_pkl != "":
            graph_datas = load_graph_datas(saved_graphdata_pkl)
            if graph_datas != None:
                plot.loss_graph(graph_datas['train_losses'], smooth=False)
                plot.loss_graph(graph_datas['train_losses'], smooth=True)
        else:
            graph_datas = {'train_losses': []}

        self.graph_datas = graph_datas
        self.str_data_info = str_data_info


    def train(self, x_datas, t_datas):
        network = self.network
        lr = self.lr
        optimizer = self.optimizer
        learning_num_per_save = self.learning_num_per_save
        goal_learning_num = self.goal_learning_num
        graph_datas = self.graph_datas
        str_data_info = self.str_data_info

        # 데이터 양을 8배로 늘리기 (보드 회전(*4), 반전(*2))
        for x_data, t_data in zip(x_datas, t_datas):
            for d in range(1, 8): # -7~0 = 0~7
                x_datas = np.append(x_datas, rotate_2dim_array(x_data, d), axis=0)
                t_datas = np.append(t_datas, rotate_2dim_array_idx(t_data[0], d), axis=0)
                # print_board(x_datas[-1], t_datas[-1], mode="QnA")
                # input()
        
        x_datas = np.r_[self.x_datas_left, x_datas]
        t_datas = np.r_[self.t_datas_left, t_datas]
        len_x_datas = len(x_datas)
        
        # 데이터를 100개씩 나누어 학습
        while len(x_datas) >= 100:
            x_batch = x_datas[:100].reshape(100, 1, 15, 15)
            t_batch = _change_one_hot_label(t_datas[:100])
        
            grads, loss = network.gradient(x_batch, t_batch, save_loss=True)
            optimizer.update(network.params, grads)
            graph_datas['train_losses'].append(loss)
            network.learning_num += 100
        
            print(time.str_hms_delta(self.start_time), end=" | ")
            print(f"{network.learning_num}문제 학습" + \
                f" | 손실 {format(loss, '.3f')}", end=" | ")
            print(f"{self.str_turns}수" if len_x_datas == len(x_datas) else "")
            self.str_turns = ""
        
            x_datas = x_datas[100:]
            t_datas = t_datas[100:]
        
            if (network.learning_num // learning_num_per_save) != self.save_num:
                win_num = sum(np.array(self.resent_who_win)==1)
                game_num = len(self.resent_who_win)
                print(f"-> 최근 {game_num}경기 중 승리 {win_num}경기 (승률: {round(win_num/game_num*100, 1)}%)")
                self.save_num = network.learning_num // learning_num_per_save
        
                network.save_params(optimizer, lr, str_data_info, verbose=False)
                save_graph_datas(graph_datas, network, optimizer, lr, str_data_info, verbose=False)
                network.delete_pre_saved_pkl(learning_num_per_save)
                print()
        
        if network.learning_num == goal_learning_num: self.exit = True
        
        self.x_datas_left = x_datas
        self.t_datas_left = t_datas


class Omok:
    def __init__(self, AI: Omok_AI, train_with_Yixin: bool, verbose: bool = False) -> None:
        self.exit = False # 프로그램 종료
        self.size = 15
        self.str_turns = ""
    
        self.AI = AI
        self.train_with_Yixin = train_with_Yixin
        self.verbose = verbose


    def run(self):
        train_with_Yixin = self.train_with_Yixin

        while not self.exit:

            board = np.zeros([self.size, self.size], dtype=np.float16)
            # print_board(board, mode="Q")
            whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
            turn = 0
            clean_board_num = 0
            max_turn = self.size * self.size
        
            game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
            game_over = False # 게임이 끝났나?
        
            # record = [] # 기보 기록할 곳
            x_datas = np.empty((0, 15, 15), dtype=np.float16)
            t_datas = np.empty((0, 1), dtype=int)
            if train_with_Yixin: Yixin.reset() ### 디버깅 끝내고 다시 풀기

            self.game_start(board, whose_turn, turn, clean_board_num, max_turn, game_end, game_over, x_datas, t_datas)

        plot.loss_graph(self.AI.graph_datas['train_losses'], smooth=True)
        plot.loss_graph(self.AI.graph_datas['train_losses'], smooth=False)
            
    
    def game_start(self, board, whose_turn, turn, clean_board_num, max_turn, game_end, game_over, x_datas, t_datas):
        train_with_Yixin = self.train_with_Yixin

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
                    x2, y2 = self.AI.network.think_win_xy(board, whose_turn, verbose=self.verbose)
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
                    # self.exit=True
                
            # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
            if not game_over:
                whose_turn *= -1
            
                if turn >= max_turn:
                    clean_board_num += 1
                    turn = 0
                    board = np.zeros([self.size, self.size], dtype=np.float16)
                
            # 흑,백 승리 이미지 화면에 추가, 학습하기, 기보 저장
            else:
                self.AI.resent_who_win.append(whose_turn)
                if len(self.AI.resent_who_win) > 100: del self.AI.resent_who_win[0]
                
                self.str_turns = self.str_turns + ("" if self.str_turns=="" else ",") + str(clean_board_num*225 + turn)
                if train_with_Yixin: Yixin.Click_setting("plays_w" if whose_turn == 1 else "plays_b")
                
                self.AI.train(x_datas, t_datas)

                # 기보 파일로 저장
                # with open('etc/GiBo_Training.txt', 'a', encoding='utf8') as file:
                    # file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                    # for i in range(len(record)):
                    #     turn_hangul = "흑" if record[i][2] == 1 else "백"
                    #     file.write(str(record[i][0]+1)+' '+str(record[i][1]+1)+' '+turn_hangul+'\n')
                    # file.write("\n")
        
                game_end = True


if __name__ == '__main__':
    saved_network_pkl = "with_Yixin (X26f256) ln_157000 acc_None Adam lr_0.01 CR_CR_CR_CR_A_Smloss params"
    # saved_network_pkl = None
    str_data_info = 'with_Yixin (X26f256)'
    train_with_Yixin = True
    verbose = False
    
    print("\n--Python 오목 AI 학습! (렌주룰)--")
    
    AI = Omok_AI(saved_network_pkl, str_data_info)

    if train_with_Yixin: Yixin.init()
    
    omok = Omok(AI, train_with_Yixin, verbose)
    omok.run()
    