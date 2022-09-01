import numpy as np # 보드 만들기
import pygame # 게임 화면 만들기
from datetime import datetime # 기보 날짜 기록
from modules.plot import *

# AI code (Human Made Algorithm)
from pygame_src.AI_code import *
from pygame_src.foul_detection import isFive, num_Four, num_Three
from modules.yixin import Yixin, Click

# AI Deep Learning network
from omok_tool import Omok_AI


class Omok_Pygame():
    def __init__(self, training_mode: bool, mute: bool) -> None:
        self.training_mode = training_mode
        self.mute = mute

        pygame.init()
        
        window_length=250*3
        window_high=250*3
        self.window_high = window_high
        self.window_num=0
        self.screen=pygame.display.set_mode((window_length,window_high))
        pygame.display.set_caption("이걸 보다니 정말 대단해!") # 제목
        
        board_img=pygame.image.load("pygame_src\img\game_board.png")
        self.board_img=pygame.transform.scale(board_img,(window_high,window_high))
        self.size=15 # 바둑판 좌표 범위
        self.dis=47 # 바둑판 이미지에서 격자 사이의 거리
        
        win_black=pygame.image.load("pygame_src\img\win_black.png")
        self.win_black=pygame.transform.scale(win_black,(int(300*2.5),300))
        
        win_white=pygame.image.load("pygame_src\img\win_white.png")
        self.win_white=pygame.transform.scale(win_white,(int(300*2.5),300))
        
        select=pygame.image.load("pygame_src\img\select2.png")
        self.select=pygame.transform.scale(select,(int(45*5/6),int(45*5/6)))
        
        last_sign1=pygame.image.load("pygame_src\img\last_sign1.png")
        self.last_sign1=pygame.transform.scale(last_sign1,(int(45*5/6),int(45*5/6)))
        
        # last_sign2=pygame.image.load("pygame_src\img\last_sign2.png")
        # last_sign2=pygame.transform.scale(last_sign2,(int(45*5/6),int(45*5/6)))
        
        black_stone=pygame.image.load("pygame_src\img\wblack_stone.png")
        self.black_stone=pygame.transform.scale(black_stone,(int(45*5/6),int(45*5/6)))
        
        white_stone=pygame.image.load("pygame_src\img\white_stone.png")
        self.white_stone=pygame.transform.scale(white_stone,(int(45*5/6),int(45*5/6)))
        
        # my_rect1 = pygame.Rect(0,0,window_num,window_high)
        # your_rect1 =  pygame.Rect(window_length-window_num,0,window_length,window_high)
        
        play_button=pygame.image.load("pygame_src\img\play_button.png")
        self.play_button=pygame.transform.scale(play_button,(250*2+14,250*1))
        
        play_button2=pygame.image.load("pygame_src\img\play_button2.png")
        self.play_button2=pygame.transform.scale(play_button2,(250*2,250*1))
        
        selected_button=pygame.image.load("pygame_src\img\selected_button.png")
        self.selected_button=pygame.transform.scale(selected_button,(250*2+14,250*1))
        
        selected_button2=pygame.image.load("pygame_src\img\selected_button2.png")
        self.selected_button2=pygame.transform.scale(selected_button2,(250*2,250*1))
        
        mute_img=pygame.image.load("pygame_src\img\mute.png")
        self.mute_img=pygame.transform.scale(mute_img,(40, 40))
        
        pygame.mixer.music.load(r'pygame_src\bgm\딥마인드챌린지 알파고 경기 시작전 브금1.wav') # 대전 BGM
        self.selecting_sound = pygame.mixer.Sound("pygame_src\sound\넘기는효과음.wav") # 게임 모드 선택 중
        self.sound1 = pygame.mixer.Sound("pygame_src\sound\othree.wav") # 게임 시작!
        self.sound2 = pygame.mixer.Sound("pygame_src\sound\바둑알 놓기.wav")
        self.sound3 = pygame.mixer.Sound("pygame_src\sound\바둑알 꺼내기.wav")
        self.sound4 = pygame.mixer.Sound("pygame_src\sound\피싱.wav") # 최후의 수!
        self.black_foul_sound = pygame.mixer.Sound("pygame_src\sound\디제이스탑.wav") # 거기 두면 안 돼! 금수야! 
        self.lose_sound = pygame.mixer.Sound("pygame_src\sound\[효과음]BOING.wav") # 응 졌어
        self.AI_lose = pygame.mixer.Sound('pygame_src\sound\알파고 쉣낏.wav') # AI를 이겼을 때
        
        # pygame.font.init() # pygame.init()에서 자동으로 실행됨
        font = r"pygame_src\BMHANNA_11yrs_ttf.ttf"
        myfont = pygame.font.Font(font, 80)
        self.threethree_text = myfont.render('응~ 쌍삼~', True, (255, 0, 0)) # True의 의미 : 글자 우둘투둘해지는거 막기 (안티 에일리어싱 여부)
        self.fourfour_text = myfont.render('응~ 사사~', True, (255, 0, 0))
        self.six_text = myfont.render('응~ 육목~', True, (255, 0, 0))
        
        myfont2 = pygame.font.Font(font, 70)
        self.foul_lose = myfont2.render('그렇게 두고 싶으면 그냥 둬', True, (255, 0, 0))
        
        myfont3 = pygame.font.Font(font, 40)
        self.AI_vs_AI_mode = myfont3.render('AI vs AI 모드', True, (255, 0, 0))
        self.AI_is_training_text = myfont3.render('학습 중...', True, (255, 0, 0))


    def make_board(self, board): # 바둑알 표시하기
        size = self.size
        screen = self.screen
        black_stone = self.black_stone
        white_stone = self.white_stone
        dis = self.dis

        for a in range(size):
            for b in range(size):
                if board[a][b]!=0 and board[a][b]==1:
                    screen.blit(black_stone,(625-18+(b-7)*dis-250,375-19+(a-7)*dis)) ## 18.75 -> 19 # 소수면 콘솔창에 경고 알림 뜸
                if board[a][b]!=0 and board[a][b]==-1:
                    screen.blit(white_stone,(625-18+(b-7)*dis-250,375-19+(a-7)*dis)) ## 18.75 -> 19


    def last_stone(self, xy): # 마지막 돌 위치 표시하기
        dis = self.dis
        dest = (625-18+(xy[0]-7)*dis-250, 375-19+(xy[1]-7)*dis)
        self.screen.blit(self.last_sign1, dest) ## 18.75 -> 19


    def run(self):
        training_mode = self.training_mode
        mute = self.mute

        size = self.size
        screen = self.screen
        board_img = self.board_img
        window_num = self.window_num
        play_button = self.play_button
        selected_button2 = self.selected_button2
        selecting_sound = self.selecting_sound
        sound1 = self.sound1
        selected_button = self.selected_button
        play_button2 = self.play_button2
        select = self.select
        dis = self.dis
        sound2 = self.sound2
        mute_img = self.mute_img
        AI_is_training_text = self.AI_is_training_text
        AI_vs_AI_mode = self.AI_vs_AI_mode
        win_black = self.win_black
        win_white = self.win_white
        black_foul_sound = self.black_foul_sound
        six_text = self.six_text
        fourfour_text = self.fourfour_text
        threethree_text = self.threethree_text
        foul_lose = self.foul_lose
        sound4 = self.sound4
        lose_sound = self.lose_sound
        window_high = self.window_high
        sound3 = self.sound3


        print("\n--Python 오목! (렌주룰)--")
        
        exit=False # 프로그램 종료
        first_trial = True
        
        resent_who_win = []
        str_turns = ""
        
        while not exit:
            pygame.display.set_caption("오목이 좋아, 볼록이 좋아? 오목!")
        
            whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
            turn = 0
            clean_board_num = 0
            final_turn = None # 승패가 결정난 턴 (수순 다시보기 할 때 활용)
            max_turn = size * size
        
            game_selected = False # 게임 모드를 선택했나?
            game_mode = "Human_vs_AI" # 게임 모드
            AI_mode = "Deep_Learning"
        
            game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
            game_over = False # 게임이 끝났나?
            game_review = False # 수순 다시보기 모드인가?
        
            record = [] # 기보 기록할 곳
            x_datas = np.empty((0, 15, 15), dtype=np.float16)
            t_datas = np.empty((0, 1), dtype=int)
            x_datas_necessary = np.empty((0, 15, 15), dtype=np.float16)
            t_datas_necessary = np.empty((0, 1), dtype=int)
        
            black_foul = False # 금수를 뒀나?
            before_foul = False # 한 수 전에 금수를 뒀나?
            stubborn_foul = "No" # 방향키를 움직이지 않고 또 금수를 두었나? (그랬을 때 금수 종류) (금수자리를 연타했나)
            foul_n_mok = 0 # 연속 금수 횟수
        
            x=7 # 커서 좌표
            y=7
            y_win=375-19 ## 18.75 -> 19 # 커서 실제 위치
            x_win=625-18-250
        
            board = np.zeros([size, size], dtype=np.float16)
            
            if not training_mode:
                screen.blit(board_img,(window_num, 0)) # 바둑판 이미지 추가
                screen.blit(play_button,(125, 100))
                screen.blit(selected_button2,(125, 400))
                pygame.display.update()
            elif train_with_Yixin:
                Yixin.reset()
                screen.blit(board_img,(window_num, 0)) # 바둑판 이미지 추가
                pygame.display.update()
            
            # print("\n게임 모드 선택")
            while not game_selected and not training_mode:
                for event in pygame.event.get():
        
                    if event.type == pygame.QUIT:
                        exit=True
                        game_selected = True
                        game_end=True
                    
                    elif event.type == pygame.KEYDOWN:
                        
                        if event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                            pygame.mixer.Sound.play(selecting_sound)
                            if game_mode=="Human_vs_AI":
                                game_mode = "Human_vs_Human"
                            else:
                                game_mode = "Human_vs_AI"
        
                        elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                            # if game_mode = "Human_vs_AI":
                            #     print("준비중")
                            #     pygame.mixer.Sound.play(black_foul_sound)
                            #     continue
                            if game_mode=="Human_vs_AI":
                                pygame.display.set_caption("...인간 주제에? 미쳤습니까 휴먼?")
                                print('\nAI: "후후... 날 이겨보겠다고?"')
                            else:
                                pygame.display.set_caption("나랑 같이...오목 할래?")
                            pygame.mixer.Sound.play(sound1)
                            game_selected = True
        
                        elif event.key == pygame.K_t:
                            if not training_mode:
                                training_mode = True
                            else:
                                training_mode = False
        
                        elif event.key == pygame.K_ESCAPE:
                            exit=True
                            game_selected = True
                            game_end=True
        
                        if not game_selected:
                            if game_mode=="Human_vs_AI":
                                screen.blit(play_button,(125, 100))
                                screen.blit(selected_button2,(125, 400))
                            else:
                                screen.blit(selected_button,(125, 100))
                                screen.blit(play_button2,(125, 400))
                        else:
                            screen.blit(board_img,(window_num, 0))
                            screen.blit(select,(x_win,y_win))
                        pygame.display.update()
        
            if not training_mode or first_trial:
                first_trial = False
                pygame.mixer.music.play(-1) if not mute else () # -1 : 반복 재생
        
            # print("게임 시작!")
            # print(difference_score_board(whose_turn, size, board), "\n") #print
            while not game_end:
                screen.blit(board_img,(window_num, 0)) ## screen.fill(0) : 검은 화면
                
                # 트레이닝 모드일 때
                if training_mode: # and time.time()-waiting_time > 0.1
                    # waiting_time = time.time()
        
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
                            x2, y2 = AI.network.think_win_xy(board, whose_turn, verbose=verbose)
                            x, y = x2, y2
                            if train_with_Yixin: Click(x2, y2, board_xy=True)
            
                            x_datas = np.append(x_datas, board.reshape(1, 15, 15), axis=0)
                            t_datas = np.append(t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
                            # if len(x_datas) > 3:
                                # x_datas = np.delete(x_datas, 0, axis=0)
                                # t_datas = np.delete(t_datas, 0, axis=0)
                        
                        # # 둘 곳이 명백할 때, 마지막 4수의 보드 상태와 내 AI답 저장 (학습할 문제와 답 저장)
                        # if is_necessary:
                            # x_datas_necessary = np.append(x_datas_necessary, board.reshape(1, 15, 15), axis=0)
                            # t_datas_necessary = np.append(t_datas_necessary, np.array([[y1*15+x1]], dtype=int), axis=0)
                        # elif whose_turn == 1:
                            # x_datas = np.append(x_datas, board.reshape(1, 15, 15), axis=0)
                            # t_datas = np.append(t_datas, np.array([[y1*15+x1]], dtype=int), axis=0)
                            # if len(x_datas) > 3:
                            #     x_datas = np.delete(x_datas, 0, axis=0)
                            #     t_datas = np.delete(t_datas, 0, axis=0)
        
                        # 선택한 좌표에 돌 두기
                        board[y][x] = whose_turn
                        
                        record.append([y, x, whose_turn])
                        last_stone_xy = [y, x]
                        turn += 1
                    
                        x_win = 28 + dis*x # 커서 이동
                        y_win = 27 + dis*y
                    
                        # 오목이 생겼으면 게임 종료 신호 키기
                        if isFive(whose_turn, board, x, y, placed=True) == True:
                            pygame.display.set_caption("나에게 복종하라 로봇.")
                            game_over=True
                        
                        # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
                        if not game_over:
                            # time.sleep(0.08) ## 바둑돌 소리 겹치지 않게 -> AI계산 시간이 길어지면서 필요없어짐
                            pygame.mixer.Sound.play(sound2) if not mute else ()
                            whose_turn *= -1
                    
                            if turn < max_turn: 
                                last_stone_xy = [y, x] # 마지막 놓은 자리 표시
                            else:
                                clean_board_num += 1
                                turn = 0
                                board = np.zeros([size, size])
                        
                    # 바둑알, 커서 위치 표시, 마지막 돌 표시, 학습 모드 화면에 표시
                    if not exit:
                        if mute:
                            screen.blit(mute_img,(355, 705))
                        self.make_board(board)
                        screen.blit(select,(x_win,y_win))
                        screen.blit(AI_is_training_text,(10, 705))
                        if turn != 0: # or event.key == pygame.K_F2 or event.key == pygame.K_F3
                            self.last_stone([last_stone_xy[1], last_stone_xy[0]])
                        if game_mode=="AI_vs_AI":
                            screen.blit(AI_vs_AI_mode,(520, 705))
                    
                    # 흑,백 승리 이미지 화면에 추가, 학습하기, 기보 저장
                    if game_over:
                        resent_who_win.append(whose_turn)
                        if len(resent_who_win) > 100: del resent_who_win[0]
                        str_turns = str_turns + ("" if str_turns=="" else ",") + str(clean_board_num*225 + turn)
                        
                        if whose_turn == 1 and not black_foul: # 흑 승리/백 승리 표시
                            screen.blit(win_black,(0,250))
                            if train_with_Yixin: Yixin.Click_setting("plays_w")
                        else:
                            screen.blit(win_white,(0,250))
                            if train_with_Yixin: Yixin.Click_setting("plays_b")
                        pygame.display.update()
        
                        # x_datas = np.r_[x_datas_necessary, x_datas]
                        # t_datas = np.r_[t_datas_necessary, t_datas]
        
                        Omok_AI.train(x_datas, t_datas)
        
                        # 기보 파일로 저장
                        with open('etc/GiBo_Training.txt', 'a', encoding='utf8') as file:
                            file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                            for i in range(len(record)):
                                turn_hangul = "흑" if record[i][2] == 1 else "백"
                                file.write(str(record[i][0]+1)+' '+str(record[i][1]+1)+' '+turn_hangul+'\n')
                            file.write("\n")
                        
                        game_end = True
        
                    # 화면 업데이트
                    pygame.display.update()
        
                    for event in pygame.event.get():
                        if event.type==pygame.QUIT:
                            exit=True
                            game_end=True 

                        if event.type == pygame.KEYDOWN:

                            if event.key == pygame.K_m: # 음소거
                                if not mute:
                                    mute = True
                                    pygame.mixer.music.pause()
                                else:
                                    mute = False
                                    pygame.mixer.music.unpause()
                    continue

                # 입력 받기
                for event in pygame.event.get():
        
                    # 키보드를 누르고 땔 때
                    if event.type == pygame.KEYDOWN:
        
                        # Enter, Space 키
                        if event.key == pygame.K_SPACE and not training_mode and not game_over: # 돌 두기
        
                            # 플레이어가 두기
                            if game_mode=="Human_vs_Human" or (game_mode=="Human_vs_AI" and whose_turn == -1):
                                
                                # 이미 돌이 놓여 있으면 다시
                                if board[y][x] == -1 or board[y][x] == 1: 
                                    print("돌이 그 자리에 이미 놓임")
                                    pygame.mixer.Sound.play(black_foul_sound)
                                    continue
                                
                                # 흑 차례엔 흑돌, 백 차례엔 백돌 두기
                                if whose_turn == 1: 
                                    board[y][x] = 1
                                else:
                                    board[y][x] = -1
                                
                                # 오목 생겼나 확인
                                five = isFive(whose_turn, board, x, y, placed=True)
                                
                                # 오목이 생겼으면 게임 종료 신호 키기, 아니면 무르기
                                if five == True:
                                    if game_mode=="Human_vs_AI":
                                        pygame.display.set_caption("다시봤습니다 휴먼!")
                                    else:
                                        pygame.display.set_caption("게임 종료!")
                                    
                                    game_over=True
                                
                                # 오목이 아닌데, 흑이면 금수 확인
                                elif whose_turn == 1:
                                    
                                    # 장목(6목 이상), 3-3, 4-4 금수인지 확인
                                    if stubborn_foul=="6목" or five == None: # black_고집 센 : 금수자리를 연타하는 경우 연산 생략
                                        print("흑은 장목을 둘 수 없음")
                                        black_foul = True
                                        stubborn_foul = "6목"
                                        screen.blit(six_text,(235, 660))
                                        if before_foul:
                                            foul_n_mok += 1
                                    elif stubborn_foul=="4-4" or num_Four(whose_turn, board, x, y, placed=True) >= 2:
                                        print("흑은 사사에 둘 수 없음")
                                        black_foul = True
                                        stubborn_foul = "4-4"
                                        screen.blit(fourfour_text,(235, 660))
                                        if before_foul:
                                            foul_n_mok += 1
                                    elif stubborn_foul=="3-3" or num_Three(whose_turn, board, x, y, placed=True) >= 2:
                                        print("흑은 삼삼에 둘 수 없음")
                                        black_foul = True
                                        stubborn_foul = "3-3"
                                        screen.blit(threethree_text,(235, 660))
                                        if before_foul:
                                               foul_n_mok += 1
                                     
                                    # 금수를 두면 무르고 다시 (연속 금수 10번까지 봐주기)
                                    if black_foul:
                                        if foul_n_mok < 10:
                                            pygame.mixer.Sound.play(black_foul_sound)
                                            before_foul = True
                                            black_foul = False
                                            board[y][x] = 0
        
                                            # 바둑알, 커서 위치 표시, 마지막 돌 표시 화면에 추가
                                            self.make_board(board)
                                            screen.blit(select,(x_win,y_win))
                                            self.last_stone([last_stone_xy[1], last_stone_xy[0]])
                                            pygame.display.update()
                                            continue
                                        else:
                                            print("그렇게 두고 싶으면 그냥 둬\n흑 반칙패!")
                                            screen.blit(board_img,(window_num, 0))
                                            screen.blit(foul_lose,(5, 670))
                                            pygame.display.set_caption("이건 몰랐지?ㅋㅋ")
                                            game_over=True
                                
                                # 돌 위치 확정
                                record.append([y, x, whose_turn]) # 기보 기록
                                last_stone_xy = [y, x] # 마지막 돌 위치 기록
                                turn += 1 # 턴 추가
                                
                                if whose_turn == 1:
                                    before_foul = False
                                    foul_n_mok = 0
                                
                                # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
                                if not game_over:
                                    pygame.mixer.Sound.play(sound2) 
                                    whose_turn *= -1
        
                                    if turn < max_turn: ### <= -> <
                                        last_stone_xy = [y,x] # 마지막 놓은 자리 표시
                                    else:
                                        turn = 0
                                        board = np.zeros([size, size])
                                else:
                                    pygame.mixer.music.stop()
                                    pygame.mixer.Sound.play(sound4)
            
                                    if whose_turn == 1 and not black_foul:
                                        # if game_mode=="Human_vs_AI":
                                            # pygame.mixer.Sound.play(AI_lose)
                                        print("흑 승리!")
                                    else:
                                        if not black_foul:
                                            print("백 승리!")
                                # print(difference_score_board(whose_turn, size, board), "\n") #print
        
                            # AI가 두기
                            if game_mode=="AI_vs_AI" or (game_mode=="Human_vs_AI" and whose_turn == 1 and not game_over):
                                
                                # 사람이 생각한 알고리즘
                                if AI_mode == "Human_Made_Algorithms":
                                    x, y = AI_think_win_xy(whose_turn, board, all_molds=True, verbose=True)
                                # 딥러닝 신경망 AI
                                else:
                                    x, y = AI.network.think_win_xy(board, whose_turn, verbose=verbose) ### DeepConvNet -> network
        
                                # 선택한 좌표에 돌 두기
                                board[y][x] = whose_turn
        
                                record.append([y, x, whose_turn])
                                last_stone_xy = [y, x]
                                turn += 1
        
                                x_win = 28 + dis*x # 커서 이동
                                y_win = 27 + dis*y
        
                                # 오목이 생겼으면 게임 종료 신호 키기
                                if isFive(whose_turn, board, x, y, placed=True) == True:
                                    pygame.display.set_caption("나에게 복종하라 인간.")
                                    game_over=True
        
                                # 승부가 결정나지 않았으면 턴 교체, 바둑판이 가득 차면 초기화
                                if not game_over:
                                    # time.sleep(0.08) ## 바둑돌 소리 겹치지 않게 -> AI계산 시간이 길어지면서 필요없어짐
                                    pygame.mixer.Sound.play(sound2)
                                    whose_turn *= -1
        
                                    if turn < max_turn: 
                                        last_stone_xy = [y, x] # 마지막 놓은 자리 표시
                                    else:
                                        turn = 0
                                        board = np.zeros([size, size])
                                else:
                                    pygame.mixer.music.stop()
                                    pygame.mixer.Sound.play(lose_sound)
                                    if not black_foul:
                                        print("백 승리!")
                                # print(difference_score_board(whose_turn, size, board), "\n") #print
                        
                        elif event.key == pygame.K_SPACE and game_over: # 금수 연타했을 때 패배 창 제대로 못 보고 넘어가기 방지
                            continue
                        elif event.key == pygame.K_RETURN and game_over: # 게임 종료
                                game_end=True
                                game_over=False
                        elif event.key == pygame.K_t: # 트레이닝 모드 on/off
                            if not training_mode:
                                training_mode = True
                            else:
                                training_mode = False
                        elif event.key == pygame.K_m: # 음소거
                            if not mute:
                                mute = True
                                pygame.mixer.music.pause()
                            else:
                                mute = False
                                pygame.mixer.music.unpause()
        
                        # ↑ ↓ → ← 방향키
                        elif event.key == pygame.K_UP: 
                            if not game_review:
                                if y_win-dis > 0:
                                    y_win -= dis
                                    y -= 1
                            stubborn_foul = "No"
                        elif event.key == pygame.K_DOWN:
                            if not game_review:
                                if y_win+dis < window_high-dis:
                                    y_win += dis
                                    y += 1
                                stubborn_foul = "No"
                        elif event.key == pygame.K_LEFT:
                            if not game_review:
                                if x_win-dis > window_num:
                                    x_win -= dis
                                    x -= 1
                                stubborn_foul = "No"
                            
                            elif turn > 0:
                                pygame.mixer.Sound.play(sound3)
                                turn -= 1
                                board[record[turn][0], record[turn][1]] = 0
                                last_stone_xy = [record[turn-1][0], record[turn-1][1]] # last_stone_xy : 돌의 마지막 위치
                        elif event.key == pygame.K_RIGHT:
                            if not game_review:
                                if x_win+dis < window_high + window_num - dis:
                                    x_win += dis
                                    x += 1
                                stubborn_foul = "No"
                            
                            elif turn < final_turn:
                                pygame.mixer.Sound.play(sound2)
                                turn += 1
                                board[record[turn-1][0], record[turn-1][1]] = record[turn-1][2]
                                last_stone_xy = [record[turn-1][0], record[turn-1][1]]
                        
                        # 기타 키
                        elif event.key == pygame.K_F1: # 바둑돌 지우기
                            pygame.mixer.Sound.play(sound3)
                            board[y][x]=0
                        elif event.key == pygame.K_F2: # 검은 바둑돌
                            board[y][x]=1
                            last_stone_xy = [y,x]
                            pygame.mixer.Sound.play(sound2)
                        elif event.key == pygame.K_F3: # 흰 바둑돌
                            board[y][x]=-1
                            last_stone_xy = [y,x]
                            pygame.mixer.Sound.play(sound2)
                        elif event.key == pygame.K_F4: # 커서 현재 위치 출력
                            print("커서 위치:", x, y) # 컴퓨터: 0부터 셈, 사람: 1부터 셈 -> +1 
                        elif event.key == pygame.K_F5: # 바둑판 비우기
                            pygame.mixer.Sound.play(sound3)
                            board = np.zeros([size, size])
                        elif event.key == pygame.K_F6: # 현재 턴 출력
                            print(str(turn)+"턴째")
                        elif event.key == pygame.K_F7: # 기보 출력
                            print(record)
                        elif event.key == pygame.K_F8: # AI vs AI mode on/off
                            if game_mode != "AI_vs_AI":
                                game_mode_origin = game_mode
                                game_mode = "AI_vs_AI"
                            else:
                                game_mode = game_mode_origin
                        elif event.key == pygame.K_F9: # AI mode Human_Made_Algorithms / Deep_Learning
                            if AI_mode != "Human_Made_Algorithms":
                                AI_mode = "Human_Made_Algorithms"
                            else:
                                AI_mode = "Deep_Learning"
                        elif event.key == pygame.K_ESCAPE: # 창 닫기
                            exit=True
                            game_end=True             
        
                        # 바둑알, 커서 위치 표시, 마지막 돌 표시, AI vs AI 모드 화면에 추가
                        if not exit:
                            if mute:
                                screen.blit(mute_img,(355, 705))
                            self.make_board(board)
                            if training_mode:
                                screen.blit(AI_is_training_text,(10, 705))
                            if not game_review:
                                screen.blit(select,(x_win,y_win))
                            if turn != 0: # or event.key == pygame.K_F2 or event.key == pygame.K_F3
                                self.last_stone([last_stone_xy[1], last_stone_xy[0]])
                            if game_mode=="AI_vs_AI":
                                screen.blit(AI_vs_AI_mode,(520, 705))
        
                        # 흑,백 승리 이미지 화면에 추가, 수순 다시보기 모드로 전환, 기보 저장
                        if game_over and not game_review:
                            game_review = True
                            final_turn = turn
                            if whose_turn == 1 and not black_foul: # 흑 승리/백 승리 표시
                                screen.blit(win_black,(0,250))
                            else:
                                screen.blit(win_white,(0,250))
        
                            # 기보 파일로 저장
                            with open('etc/GiBo.txt', 'a', encoding='utf8') as file:
                                file.write(datetime.today().strftime("%Y/%m/%d %H:%M:%S") + "\n") # YYYY/mm/dd HH:MM:SS 형태로 출력
                                for i in range(len(record)):
                                    turn_hangul = "흑" if record[i][2] == 1 else "백"
                                    file.write(str(record[i][0]+1)+' '+str(record[i][1]+1)+' '+turn_hangul+'\n')
                                file.write("\n")
        
                        # 화면 업데이트
                        pygame.display.update()
        
                    # 창 닫기(X) 버튼을 클릭했을 때
                    elif event.type == pygame.QUIT:
                        exit=True
                        game_end=True
          
        print("\nGood Bye")
        pygame.quit()
        
        plot.loss_graph(AI.graph_datas['train_losses'], smooth=True)
        plot.loss_graph(AI.graph_datas['train_losses'], smooth=False)


if __name__ == '__main__':
    saved_network_pkl = "with_Yixin (X26f256) ln_157000 acc_None Adam lr_0.01 CR_CR_CR_CR_A_Smloss params"
    str_data_info = ' '.join(saved_network_pkl.split(' ')[:2])
    
    training_mode = False
    train_with_Yixin = False
    verbose = False
    mute = False
    AI = Omok_AI(saved_network_pkl, str_data_info)
    
    if training_mode and train_with_Yixin: Yixin.init()
    
    omok_pygame = Omok_Pygame(training_mode=training_mode, mute=mute)
    omok_pygame.run()
    