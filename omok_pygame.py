import numpy as np # 보드 만들기
import pygame # 게임 화면 만들기
from pygame.mixer import Sound

from datetime import datetime # 기보 날짜 기록
from modules.plot import *

# AI code (Human Made Algorithm)
from pygame_src.AI_code import *
from pygame_src.foul_detection import isFive, num_Four, num_Three
from modules.yixin import Yixin, Click

# AI Deep Learning network
from omok_tool import Omok_AI


def img_load(filename, size):
    return pygame.transform.scale(pygame.image.load(filename), size)


class Omok_Pygame():
    def __init__(self, AI: Omok_AI, training_mode: bool, train_with_Yixin: bool, verbose: bool, mute: bool) -> None:
        self.AI = AI
        self.training_mode = training_mode
        self.train_with_Yixin = train_with_Yixin
        self.verbose = verbose
        self.mute = mute

        pygame.init()
        
        window_length = 250*3
        window_high = 250*3
        self.window_high = window_high
        self.window_num = 0
        self.size = 15 # 바둑판 좌표 범위
        self.dis = 47 # 바둑판 이미지에서 격자 사이의 거리

        self.screen = pygame.display.set_mode((window_length, window_high))
        pygame.display.set_caption("이걸 보다니 정말 대단해!") # 제목

        self.load_source(window_length, window_high)


    def load_source(self, winow_length, window_high):
        self._load_images(winow_length, window_high)
        self._load_sonuds()
        self._load_texts()


    def _load_images(self, winow_length, window_high, path='pygame_src\img'):

        # my_rect1 = Rect(0,0,window_num,window_high)
        endmark_size = (int(300*2.5),300)
        stone_w = int(45*5/6)
        menu_w = 250*2
        menu_h = 250*1

        images = {}
        images['game_board'] = img_load(f"{path}\game_board.png", (winow_length, window_high))
        images['win_black'] = img_load(f"{path}\win_black.png", endmark_size)
        images['win_white'] = img_load(f"{path}\win_white.png", endmark_size)
        images['select'] = img_load(f"{path}\select2.png", (stone_w,stone_w))
        images['last_sign1'] = img_load(f"{path}\last_sign1.png", (stone_w,stone_w))
        # images[last_sign2] = img_load(f"{path}\last_sign2.png", (stone_w,stone_w))
        images['black_stone'] = img_load(f"{path}\wblack_stone.png", (stone_w,stone_w))
        images['white_stone'] = img_load(f"{path}\white_stone.png", (stone_w,stone_w))
        # images[your_rect1] = Rect(window_length-window_num,0,window_length,window_high)
        images['play_button'] = img_load(f"{path}\play_button.png", (menu_w+14,menu_h))
        images['play_button2'] = img_load(f"{path}\play_button2.png", (menu_w,menu_h))
        images['selected_button'] = img_load(f"{path}\selected_button.png", (menu_w+14,menu_h))
        images['selected_button2'] = img_load(f"{path}\selected_button2.png", (menu_w,menu_h))
        images['mute_img'] = img_load(f"{path}\mute.png", (40, 40))
        self.images = images


    def _load_sonuds(self, path='pygame_src\sound'):
        pygame.mixer.music.load(r'pygame_src\bgm\딥마인드챌린지 알파고 경기 시작전 브금1.wav') # 대전 BGM

        sounds = {}
        sounds['넘기는효과음'] = Sound(f"{path}\넘기는효과음.wav") # 게임 모드 선택 중
        sounds['othree'] = Sound(f"{path}\othree.wav") # 게임 시작!
        sounds['바둑알 놓기'] = Sound(f"{path}\바둑알 놓기.wav")
        sounds['바둑알 꺼내기'] = Sound(f"{path}\바둑알 꺼내기.wav")
        sounds['피싱'] = Sound(f"{path}\피싱.wav") # 최후의 수!
        sounds['디제이스탑'] = Sound(f"{path}\디제이스탑.wav") # 거기 두면 안 돼! 금수야! 
        sounds['BOING'] = Sound(f"{path}\[효과음]BOING.wav") # 응 졌어
        sounds['AI_lose'] = Sound(f'{path}\알파고 쉣낏.wav') # AI를 이겼을 때
        self.sounds = sounds


    def _load_texts(self):
        # pygame.font.init() # pygame.init()에서 자동으로 실행됨
        HANNA = r"pygame_src\BMHANNA_11yrs_ttf.ttf"
        fonts = {}
        fonts[80] = pygame.font.Font(HANNA, 80)
        fonts[70] = pygame.font.Font(HANNA, 70)
        fonts[40] = pygame.font.Font(HANNA, 40)

        texts = {}
        texts['threethree_text'] = fonts[80].render('응~ 쌍삼~', True, (255, 0, 0)) # True의 의미 : 글자 우둘투둘해지는거 막기 (안티 에일리어싱 여부)
        texts['fourfour_text'] = fonts[80].render('응~ 사사~', True, (255, 0, 0))
        texts['six_text'] = fonts[80].render('응~ 육목~', True, (255, 0, 0))
        texts['foul_lose'] = fonts[70].render('그렇게 두고 싶으면 그냥 둬', True, (255, 0, 0))
        texts['AI_vs_AI_mode'] = fonts[40].render('AI vs AI 모드', True, (255, 0, 0))
        texts['AI_is_training_text'] = fonts[40].render('학습 중...', True, (255, 0, 0))
        self.texts = texts


    def run(self):
        mute = self.mute
        size = self.size
        screen = self.screen
        window_num = self.window_num
        self.dis = self.dis
        window_high = self.window_high

        images = self.images
        sounds = self.sounds
        texts = self.texts

        print("\n--Python 오목! (렌주룰)--")
        
        self.exit = False # 프로그램 종료
        first_trial = True
        
        self.resent_who_win = []
        
        while not self.exit:
            pygame.display.set_caption("오목이 좋아, 볼록이 좋아? 오목!")
        
            self.whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
            self.turn = 0
            self.clean_board_num = 0
            self.final_turn = None # 승패가 결정난 턴 (수순 다시보기 할 때 활용)
            self.max_turn = size * size
        
            self.game_selected = False # 게임 모드를 선택했나?
            self.game_mode = "Human_vs_AI" # 게임 모드
            AI_mode = "Deep_Learning"
        
            self.game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
            self.game_over = False # 게임이 끝났나?
            self.game_review = False # 수순 다시보기 모드인가?
        
            self.record = [] # 기보 기록할 곳
            self.x_datas = np.empty((0, 15, 15), dtype=np.float16)
            self.t_datas = np.empty((0, 1), dtype=int)
            x_datas_necessary = np.empty((0, 15, 15), dtype=np.float16)
            t_datas_necessary = np.empty((0, 1), dtype=int)
        
            self.black_foul = False # 금수를 뒀나?
            before_foul = False # 한 수 전에 금수를 뒀나?
            stubborn_foul = "No" # 방향키를 움직이지 않고 또 금수를 두었나? (그랬을 때 금수 종류) (금수자리를 연타했나)
            foul_n_mok = 0 # 연속 금수 횟수
        
            x = 7 # 커서 좌표
            y = 7
            self.y_win = 375-19 ## 18.75 -> 19 # 커서 실제 위치
            self.x_win = 625-18-250
        
            self.board = np.zeros([size, size], dtype=np.float16)
            
            # print("\n게임 모드 선택")
            self.select_game_mode_phase()
        
            if not self.training_mode or first_trial:
                first_trial = False
                pygame.mixer.music.play(-1) if not mute else () # -1 : 반복 재생
        
            # print("게임 시작!")
            # print(difference_score_board(self.whose_turn, size, self.board), "\n") #print
            while not self.game_end:
                screen.blit(images.get('game_board'),(window_num, 0)) ## screen.fill(0) : 검은 화면
                
                # 트레이닝 모드일 때
                if self.training_mode:
                    self.training_phase()
                    continue


                # AI가 두기
                if self.game_mode=="AI_vs_AI" or (self.game_mode=="Human_vs_AI" and self.whose_turn == 1 and not self.game_over):
                    
                    # 알고리즘 AI
                    if AI_mode == "Human_Made_Algorithms":
                        x, y = AI_think_win_xy(self.whose_turn, self.board, all_molds=True, verbose=True)
                    # 딥러닝 신경망 AI
                    else:
                        x, y = self.AI.network.think_win_xy(self.board, self.whose_turn, verbose=self.verbose) ### DeepConvNet -> network
        
                    # 선택한 좌표에 돌 두기
                    self.place(x, y)
                    self.turn_end_phase(x, y)
                    self.screen_bilt()
                    pygame.display.update()
                        

                # 입력 받기
                for event in pygame.event.get():

                    # 키보드를 누르고 땔 때
                    if event.type == pygame.KEYDOWN:
        
                        # Enter, Space 키
                        if event.key == pygame.K_SPACE and not self.training_mode and not self.game_over: # 돌 두기
        
                            # 플레이어가 두기
                            if self.game_mode=="Human_vs_Human" or (self.game_mode=="Human_vs_AI" and self.whose_turn == -1):
                                
                                # 이미 돌이 놓여 있으면 다시
                                if self.board[y][x] == -1 or self.board[y][x] == 1: 
                                    print("돌이 그 자리에 이미 놓임")
                                    Sound.play(sounds.get('디제이스탑'))
                                    continue
                                
                                # 흑 차례엔 흑돌, 백 차례엔 백돌 두기
                                board_temp = self.board.copy()
                                board_temp[y][x] = self.whose_turn
                                
                                # 오목 생겼나 확인
                                five = isFive(self.whose_turn, board_temp, x, y, placed=True)
                                
                                # 오목이 아닌데, 흑이면 금수 확인
                                if not five and self.whose_turn == 1:
                                    
                                    # 장목(6목 이상), 3-3, 4-4 금수인지 확인
                                    if stubborn_foul=="6목" or five == None: # black_고집 센 : 금수자리를 연타하는 경우 연산 생략
                                        print("흑은 장목을 둘 수 없음")
                                        self.black_foul = True
                                        stubborn_foul = "6목"
                                        screen.blit(texts.get('six_text'),(235, 660))
                                        if before_foul:
                                            foul_n_mok += 1
                                    elif stubborn_foul=="4-4" or num_Four(self.whose_turn, self.board, x, y, placed=True) >= 2:
                                        print("흑은 사사에 둘 수 없음")
                                        self.black_foul = True
                                        stubborn_foul = "4-4"
                                        screen.blit(texts.get('fourfour_text'),(235, 660))
                                        if before_foul:
                                            foul_n_mok += 1
                                    elif stubborn_foul=="3-3" or num_Three(self.whose_turn, self.board, x, y, placed=True) >= 2:
                                        print("흑은 삼삼에 둘 수 없음")
                                        self.black_foul = True
                                        stubborn_foul = "3-3"
                                        screen.blit(texts.get('threethree_text'),(235, 660))
                                        if before_foul:
                                               foul_n_mok += 1
                                     
                                    # 금수를 두면 무르고 다시 (연속 금수 10번까지 봐주기)
                                    if self.black_foul:
                                        if foul_n_mok < 10:
                                            Sound.play(sounds.get('디제이스탑'))
                                            before_foul = True
                                            self.black_foul = False
        
                                            # 바둑알, 커서 위치 표시, 마지막 돌 표시 화면에 추가
                                            self.screen_bilt_board()
                                            screen.blit(images.get('select'),(self.x_win,self.y_win))
                                            self.screen_bilt_last_stone([self.last_stone_xy[1], self.last_stone_xy[0]])
                                            pygame.display.update()
                                            continue
                                        else:
                                            print("그렇게 두고 싶으면 그냥 둬\n흑 반칙패!")
                                            screen.blit(images.get('game_board'),(window_num, 0))
                                            screen.blit(texts.get('foul_lose'),(5, 670))
                                            pygame.display.set_caption("이건 몰랐지?ㅋㅋ")
                                            self.game_over=True
                                
                                if self.whose_turn == 1:
                                    before_foul = False
                                    foul_n_mok = 0

                                # 돌 위치 확정
                                self.place(x, y)
                                self.turn_end_phase(x, y)


                        elif event.key == pygame.K_SPACE and self.game_over: # 금수 연타했을 때 패배 창 제대로 못 보고 넘어가기 방지
                            continue
                        elif event.key == pygame.K_RETURN and self.game_over: # 게임 종료
                                self.game_end=True
                                self.game_over=False
                        elif event.key == pygame.K_t: # 트레이닝 모드 on/off
                            if not self.training_mode:
                                self.training_mode = True
                            else:
                                self.training_mode = False
                        elif event.key == pygame.K_m: # 음소거
                            if not mute:
                                mute = True
                                pygame.mixer.music.pause()
                            else:
                                mute = False
                                pygame.mixer.music.unpause()
        
                        # ↑ ↓ → ← 방향키
                        elif event.key == pygame.K_UP: 
                            if not self.game_review:
                                if self.y_win-self.dis > 0:
                                    self.y_win -= self.dis
                                    y -= 1
                            stubborn_foul = "No"
                        elif event.key == pygame.K_DOWN:
                            if not self.game_review:
                                if self.y_win+self.dis < window_high-self.dis:
                                    self.y_win += self.dis
                                    y += 1
                                stubborn_foul = "No"
                        elif event.key == pygame.K_LEFT:
                            if not self.game_review:
                                if self.x_win-self.dis > window_num:
                                    self.x_win -= self.dis
                                    x -= 1
                                stubborn_foul = "No"
                            
                            elif self.turn > 0:
                                Sound.play(sounds.get('바둑알 꺼내기'))
                                self.turn -= 1
                                self.board[self.record[self.turn][0], self.record[self.turn][1]] = 0
                                self.last_stone_xy = [self.record[self.turn-1][0], self.record[self.turn-1][1]] # self.last_stone_xy : 돌의 마지막 위치
                        elif event.key == pygame.K_RIGHT:
                            if not self.game_review:
                                if self.x_win+self.dis < window_high + window_num - self.dis:
                                    self.x_win += self.dis
                                    x += 1
                                stubborn_foul = "No"
                            
                            elif self.turn < self.final_turn:
                                Sound.play(sounds.get('바둑알 놓기'))
                                self.turn += 1
                                self.board[self.record[self.turn-1][0], self.record[self.turn-1][1]] = self.record[self.turn-1][2]
                                self.last_stone_xy = [self.record[self.turn-1][0], self.record[self.turn-1][1]]
                        
                        # 기타 키
                        elif event.key == pygame.K_F1: # 바둑돌 지우기
                            Sound.play(sounds.get('바둑알 꺼내기'))
                            self.board[y][x]=0
                        elif event.key == pygame.K_F2: # 검은 바둑돌
                            self.board[y][x]=1
                            self.last_stone_xy = [y,x]
                            Sound.play(sounds.get('바둑알 놓기'))
                        elif event.key == pygame.K_F3: # 흰 바둑돌
                            self.board[y][x]=-1
                            self.last_stone_xy = [y,x]
                            Sound.play(sounds.get('바둑알 놓기'))
                        elif event.key == pygame.K_F4: # 커서 현재 위치 출력
                            print("커서 위치:", x, y) # 컴퓨터: 0부터 셈, 사람: 1부터 셈 -> +1 
                        elif event.key == pygame.K_F5: # 바둑판 비우기
                            Sound.play(sounds.get('바둑알 꺼내기'))
                            self.board = np.zeros([size, size])
                        elif event.key == pygame.K_F6: # 현재 턴 출력
                            print(str(self.turn)+"턴째")
                        elif event.key == pygame.K_F7: # 기보 출력
                            print(self.record)
                        elif event.key == pygame.K_F8: # AI vs AI mode on/off
                            if self.game_mode != "AI_vs_AI":
                                game_mode_origin = self.game_mode
                                self.game_mode = "AI_vs_AI"
                            else:
                                self.game_mode = game_mode_origin
                        elif event.key == pygame.K_F9: # AI mode Human_Made_Algorithms / Deep_Learning
                            if AI_mode != "Human_Made_Algorithms":
                                AI_mode = "Human_Made_Algorithms"
                            else:
                                AI_mode = "Deep_Learning"
                        elif event.key == pygame.K_ESCAPE: # 창 닫기
                            self.exit=True
                            self.game_end=True             
        
                        # 화면 업데이트
                        self.screen_bilt()
                        pygame.display.update()
        
                    # 창 닫기(X) 버튼을 클릭했을 때
                    elif event.type == pygame.QUIT:
                        self.exit=True
                        self.game_end=True
                    
                
        print("\nGood Bye")
        pygame.quit()


    def place(self, x: int, y: int):
        self.board[y][x] = self.whose_turn
        
        self.record.append([y, x, self.whose_turn])
        self.last_stone_xy = [y, x]
        self.turn += 1
        
        self.x_win = 28 + self.dis*x # 커서 이동
        self.y_win = 27 + self.dis*y


    def select_game_mode_phase(self):
        screen = self.screen
        sounds = self.sounds
        images = self.images

        if not self.training_mode:
            screen.blit(images.get('game_board'),(self.window_num, 0)) # 바둑판 이미지 추가
            screen.blit(images.get('play_button'),(125, 100))
            screen.blit(images.get('selected_button2'),(125, 400))
            pygame.display.update()
        elif self.train_with_Yixin:
            Yixin.reset()
            screen.blit(images.get('game_board'),(self.window_num, 0)) # 바둑판 이미지 추가
            pygame.display.update()


        while not self.game_selected and not self.training_mode:
            for event in pygame.event.get():
        
                if event.type == pygame.QUIT:
                    self.exit=True
                    self.game_selected = True
                    self.game_end=True
                
                elif event.type == pygame.KEYDOWN:
                    
                    if event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                        Sound.play(sounds.get('넘기는효과음'))
                        if self.game_mode=="Human_vs_AI":
                            self.game_mode = "Human_vs_Human"
                        else:
                            self.game_mode = "Human_vs_AI"
        
                    elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                        if self.game_mode=="Human_vs_AI":
                            pygame.display.set_caption("...인간 주제에? 미쳤습니까 휴먼?")
                            print('\nAI: "후후... 날 이겨보겠다고?"')
                        else:
                            pygame.display.set_caption("나랑 같이...오목 할래?")
                        Sound.play(sounds.get('othree'))
                        self.game_selected = True
        
                    elif event.key == pygame.K_t:
                        if not self.training_mode:
                            self.training_mode = True
                        else:
                            self.training_mode = False
        
                    elif event.key == pygame.K_ESCAPE:
                        self.exit=True
                        self.game_selected = True
                        self.game_end=True
        
                    if not self.game_selected:
                        if self.game_mode=="Human_vs_AI":
                            screen.blit(images.get('play_button'),(125, 100))
                            screen.blit(images.get('selected_button2'),(125, 400))
                        else:
                            screen.blit(images.get('selected_button'),(125, 100))
                            screen.blit(images.get('play_button2'),(125, 400))
                    else:
                        screen.blit(images.get('game_board'),(self.window_num, 0))
                        screen.blit(images.get('select'),(self.x_win,self.y_win))
                    pygame.display.update()


    def training_phase(self):
        mute = self.mute
        size = self.size
        screen = self.screen
        self.dis = self.dis

        images = self.images
        sounds = self.sounds
        texts = self.texts

        # AI가 두기
        if not self.game_over:
        
            if self.turn == 0:
                x1, y1 = 7, 7
            elif self.train_with_Yixin and self.turn >= 3: # 알파오 Yixin (백)
                x1, y1 = Yixin.think_win_xy(self.whose_turn, undo=True if self.whose_turn == 1 else False)
                # print(f"{X_line[x1]}{Y_line[y1]} Yixin" if x1!=None else None)
            if not self.train_with_Yixin or x1==None or self.turn < 3: # 알고리즘 AI (백)
                x1, y1 = AI_think_win_xy(self.whose_turn, self.board, all_molds=True, verbose=False)
                # print(f"{X_line[x1]}{Y_line[y1]} myAI")
                if self.train_with_Yixin and self.turn == 2: Yixin.Click_setting("plays_w")
                if self.train_with_Yixin and self.whose_turn==-1:
                    if self.turn == 1:
                        Click(x1, y1, board_xy=True)
                    else:
                        Yixin.Click_setting("plays_w")
                        Click(x1, y1, board_xy=True)
                        Yixin.Click_setting("plays_w")
    
            if self.whose_turn != 1:
                x, y = x1, y1
                if self.train_with_Yixin and self.turn >= 3:
                    Yixin.Click_setting("plays_b")
                    Yixin.Click_setting("plays_w")
                    Click(0, 0) # 다음 흑이 둘 수 미리 생각하기 (Yixin)
            else: # 딥러닝 신경망 AI (흑)
                x2, y2 = self.AI.network.think_win_xy(self.board, self.whose_turn, verbose=self.verbose)
                x, y = x2, y2
                if self.train_with_Yixin: Click(x2, y2, board_xy=True)
            
                self.x_datas = np.append(self.x_datas, self.board.reshape(1, 15, 15), axis=0)
                self.t_datas = np.append(self.t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
                # if len(self.x_datas) > 3:
                    # self.x_datas = np.delete(self.x_datas, 0, axis=0)
                    # self.t_datas = np.delete(self.t_datas, 0, axis=0)
            
            # # 둘 곳이 명백할 때, 마지막 4수의 보드 상태와 내 AI답 저장 (학습할 문제와 답 저장)
            # if is_necessary:
                # x_datas_necessary = np.append(x_datas_necessary, self.board.reshape(1, 15, 15), axis=0)
                # t_datas_necessary = np.append(t_datas_necessary, np.array([[y1*15+x1]], dtype=int), axis=0)
            # elif self.whose_turn == 1:
                # self.x_datas = np.append(self.x_datas, self.board.reshape(1, 15, 15), axis=0)
                # self.t_datas = np.append(self.t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
                # if len(self.x_datas) > 3:
                #     self.x_datas = np.delete(self.x_datas, 0, axis=0)
                #     self.t_datas = np.delete(self.t_datas, 0, axis=0)
        
            # 선택한 좌표에 돌 두기
            self.place(x, y)
        
            # 오목이 생겼으면 게임 종료 신호 키기
            if isFive(self.whose_turn, self.board, x, y, placed=True) == True:
                pygame.display.set_caption("나에게 복종하라 로봇.")
                self.game_over=True
            
            # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
            if not self.game_over:
                # time.sleep(0.08) ## 바둑돌 소리 겹치지 않게 -> AI계산 시간이 길어지면서 필요없어짐
                Sound.play(sounds.get('바둑알 놓기')) if not mute else ()
                self.whose_turn *= -1
        
                if self.turn < self.max_turn: 
                    self.last_stone_xy = [y, x] # 마지막 놓은 자리 표시
                else:
                    self.clean_board_num += 1
                    self.turn = 0
                    self.board = np.zeros([size, size])
            
        # 바둑알, 커서 위치 표시, 마지막 돌 표시, 학습 모드 화면에 표시
        if not self.exit:
            if mute:
                screen.blit(images.get('mute_img'),(355, 705))
            self.screen_bilt_board()
            screen.blit(images.get('select'),(self.x_win,self.y_win))
            screen.blit(texts.get('AI_is_training_text'),(10, 705))
            if self.turn != 0: # or event.key == pygame.K_F2 or event.key == pygame.K_F3
                self.screen_bilt_last_stone([self.last_stone_xy[1], self.last_stone_xy[0]])
            if self.game_mode=="AI_vs_AI":
                screen.blit(texts.get('AI_vs_AI_mode'),(520, 705))
        
        # 흑,백 승리 이미지 화면에 추가, 학습하기, 기보 저장
        if self.game_over:
            self.resent_who_win.append(self.whose_turn)
            if len(self.resent_who_win) > 100: del self.resent_who_win[0]
            print(str(self.clean_board_num*225 + self.turn) + '수')

            if self.whose_turn == 1 and not self.black_foul: # 흑 승리/백 승리 표시
                screen.blit(images.get('win_black'),(0,250))
                if self.train_with_Yixin: Yixin.Click_setting("plays_w")
            else:
                screen.blit(images.get('win_white'),(0,250))
                if self.train_with_Yixin: Yixin.Click_setting("plays_b")
            pygame.display.update()
        
            # self.x_datas = np.r_[x_datas_necessary, self.x_datas]
            # self.t_datas = np.r_[t_datas_necessary, self.t_datas]

            self.AI.train(self.x_datas, self.t_datas)
        
            # 기보 파일로 저장
            with open('etc/GiBo_Training.txt', 'a', encoding='utf8') as file:
                file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                for i in range(len(self.record)):
                    turn_hangul = "흑" if self.record[i][2] == 1 else "백"
                    file.write(str(self.record[i][0]+1)+' '+str(self.record[i][1]+1)+' '+turn_hangul+'\n')
                file.write("\n")
            
            self.game_end = True
        
        # 화면 업데이트
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                self.exit=True
                self.game_end=True 

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_m: # 음소거
                    if not mute:
                        mute = True
                        pygame.mixer.music.pause()
                    else:
                        mute = False
                        pygame.mixer.music.unpause()


    def turn_end_phase(self, x: int, y: int):
        # print(difference_score_board(self.whose_turn, size, self.board), "\n") #print

        # 오목이 생겼으면 게임 종료 신호 키기
        if isFive(self.whose_turn, self.board, x, y, placed=True):
            self.game_over=True
    
        # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
        if not self.game_over:
            # time.sleep(0.08) ## 바둑돌 소리 겹치지 않게 -> AI계산 시간이 길어지면서 필요없어짐
            Sound.play(self.sounds.get('바둑알 놓기'))
            self.whose_turn *= -1
            
            if self.turn < self.max_turn: ### <= -> <
                self.last_stone_xy = [y, x] # 마지막 놓은 자리 표시
            else:
                self.turn = 0
                self.board = np.zeros([self.size, self.size])
        else:
            pygame.mixer.music.stop()
    
            if self.game_mode=="Human_vs_Human":
                pygame.display.set_caption("게임 종료!")
                Sound.play(self.sounds.get('피싱'))
            elif self.game_mode=="Human_vs_AI" and self.whose_turn == 1:
                pygame.display.set_caption("나에게 복종하라 인간.")
                Sound.play(self.sounds.get('BOING'))
            elif self.game_mode=="Human_vs_AI" and self.whose_turn == -1 and not self.black_foul:
                pygame.display.set_caption("다시봤습니다 휴먼!")
                Sound.play(self.sounds.get('피싱'))
    
            if self.whose_turn == 1 and not self.black_foul:
                # if self.game_mode=="Human_vs_AI":
                    # Sound.play(AI_lose)
                print("흑 승리!")
            elif not self.black_foul:
                print("백 승리!")
        

    def screen_bilt_board(self): # 바둑알 표시하기
        size = self.size
        screen = self.screen
        black_stone = self.images.get('black_stone')
        white_stone = self.images.get('white_stone')

        for a in range(size):
            for b in range(size):
                if self.board[a][b]!=0 and self.board[a][b]==1:
                    screen.blit(black_stone,(625-18+(b-7)*self.dis-250,375-19+(a-7)*self.dis)) ## 18.75 -> 19 # 소수면 콘솔창에 경고 알림 뜸
                if self.board[a][b]!=0 and self.board[a][b]==-1:
                    screen.blit(white_stone,(625-18+(b-7)*self.dis-250,375-19+(a-7)*self.dis)) ## 18.75 -> 19


    def screen_bilt_last_stone(self, xy): # 마지막 돌 위치 표시하기
        dest = (625-18+(xy[0]-7)*self.dis-250, 375-19+(xy[1]-7)*self.dis)
        self.screen.blit(self.images.get('last_sign1'), dest) ## 18.75 -> 19


    def screen_bilt(self):
        screen = self.screen
        images = self.images
        texts = self.texts

        # 바둑알, 커서 위치 표시, 마지막 돌 표시, AI vs AI 모드 화면에 추가
        if not self.exit:
            if mute:
                screen.blit(images.get('mute_img'),(355, 705))
            self.screen_bilt_board()
            if self.training_mode:
                screen.blit(texts.get('AI_is_training_text'),(10, 705))
            if not self.game_review:
                screen.blit(images.get('select'),(self.x_win,self.y_win))
            if self.turn != 0: # or event.key == pygame.K_F2 or event.key == pygame.K_F3
                self.screen_bilt_last_stone([self.last_stone_xy[1], self.last_stone_xy[0]])
            if self.game_mode=="AI_vs_AI":
                screen.blit(texts.get('AI_vs_AI_mode'),(520, 705))

            # 흑,백 승리 이미지 화면에 추가, 수순 다시보기 모드로 전환, 기보 저장
            if self.game_over and not self.game_review:
                self.game_review = True
                self.final_turn = self.turn
                if self.whose_turn == 1 and not self.black_foul: # 흑 승리/백 승리 표시
                    screen.blit(images.get('win_black'),(0,250))
                else:
                    screen.blit(images.get('win_white'),(0,250))
        
                # 기보 파일로 저장
                with open('etc/GiBo.txt', 'a', encoding='utf8') as file:
                    file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                    for i in range(len(self.record)):
                        turn_hangul = "흑" if self.record[i][2] == 1 else "백"
                        file.write(str(self.record[i][0]+1)+' '+str(self.record[i][1]+1)+' '+turn_hangul+'\n')
                    file.write("\n")


if __name__ == '__main__':
    # saved_network_pkl = "with_Yixin (X26f256) ln_157000 acc_None Adam lr_0.01 CR_CR_CR_CR_A_Smloss params"
    # str_data_info = ' '.join(saved_network_pkl.split(' ')[:2])
    saved_network_pkl = None
    str_data_info = 'str_data_info'
    
    training_mode = False
    train_with_Yixin = False
    verbose = False
    mute = False
    AI = Omok_AI(saved_network_pkl, str_data_info)
    
    if training_mode and train_with_Yixin: Yixin.init()
    
    omok_pygame = Omok_Pygame(AI, training_mode, train_with_Yixin, verbose, mute)
    omok_pygame.run()

    # plot.loss_graph(AI.graph_datas['train_losses'], smooth=True)
    # plot.loss_graph(AI.graph_datas['train_losses'], smooth=False)
