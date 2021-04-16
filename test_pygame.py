import numpy as np # 보드 만들기
import pygame # 게임 화면 만들기
import random 
from datetime import datetime # 기보 날짜 기록
from modules.common.util import bcolors

# AI code (Human Made)
from pygame_src.AI_code import *
from pygame_src.foul_detection import isFive, num_Four, num_Three

# AI Deep Learning network
from network import DeepConvNet

network = DeepConvNet(saved_network_pkl="CR_CR_CR_CsumR_Smloss Momentum lr=0.01 ln=28600 acc=99.93 params")

################################################################ pygame code

pygame.init()

window_length=250*3
window_high=250*3
window_num=0
screen=pygame.display.set_mode((window_length,window_high))
pygame.display.set_caption("이걸 보다니 정말 대단해!") # 제목

board_img=pygame.image.load("pygame_src\img\game_board.png")
board_img=pygame.transform.scale(board_img,(window_high,window_high))
size=15 # 바둑판 좌표 범위
dis=47 # 바둑판 이미지에서 격자 사이의 거리

win_black=pygame.image.load("pygame_src\img\win_black.png")
win_black=pygame.transform.scale(win_black,(int(300*2.5),300))

win_white=pygame.image.load("pygame_src\img\win_white.png")
win_white=pygame.transform.scale(win_white,(int(300*2.5),300))

select=pygame.image.load("pygame_src\img\select2.png")
select=pygame.transform.scale(select,(int(45*5/6),int(45*5/6)))

last_sign1=pygame.image.load("pygame_src\img\last_sign1.png")
last_sign1=pygame.transform.scale(last_sign1,(int(45*5/6),int(45*5/6)))

# last_sign2=pygame.image.load("pygame_src\img\last_sign2.png")
# last_sign2=pygame.transform.scale(last_sign2,(int(45*5/6),int(45*5/6)))

black_stone=pygame.image.load("pygame_src\img\wblack_stone.png")
black_stone=pygame.transform.scale(black_stone,(int(45*5/6),int(45*5/6)))

white_stone=pygame.image.load("pygame_src\img\white_stone.png")
white_stone=pygame.transform.scale(white_stone,(int(45*5/6),int(45*5/6)))

# my_rect1 = pygame.Rect(0,0,window_num,window_high)
# your_rect1 =  pygame.Rect(window_length-window_num,0,window_length,window_high)

play_button=pygame.image.load("pygame_src\img\play_button.png")
play_button=pygame.transform.scale(play_button,(250*2+14,250*1))

play_button2=pygame.image.load("pygame_src\img\play_button2.png")
play_button2=pygame.transform.scale(play_button2,(250*2,250*1))

selected_button=pygame.image.load("pygame_src\img\selected_button.png")
selected_button=pygame.transform.scale(selected_button,(250*2+14,250*1))

selected_button2=pygame.image.load("pygame_src\img\selected_button2.png")
selected_button2=pygame.transform.scale(selected_button2,(250*2,250*1))

pygame.mixer.music.load(r'pygame_src\bgm\딥마인드챌린지 알파고 경기 시작전 브금1.wav') # 대전 BGM
selecting_sound = pygame.mixer.Sound("pygame_src\sound\넘기는효과음.wav") # 게임 모드 선택 중
sound1 = pygame.mixer.Sound("pygame_src\sound\othree.wav") # 게임 시작!
sound2 = pygame.mixer.Sound("pygame_src\sound\바둑알 놓기.wav")
sound3 = pygame.mixer.Sound("pygame_src\sound\바둑알 꺼내기.wav")
sound4 = pygame.mixer.Sound("pygame_src\sound\피싱.wav") # 최후의 수!
black_foul_sound = pygame.mixer.Sound("pygame_src\sound\디제이스탑.wav") # 거기 두면 안 돼! 금수야! 
lose_sound = pygame.mixer.Sound("pygame_src\sound\[효과음]BOING.wav") # 응 졌어
AI_lose = pygame.mixer.Sound('pygame_src\sound\알파고 쉣낏.wav') # AI를 이겼을 때

# pygame.font.init() # pygame.init()에서 자동으로 실행됨
font = r"pygame_src\BMHANNA_11yrs_ttf.ttf"
myfont = pygame.font.Font(font, 80)
threethree_text = myfont.render('응~ 쌍삼~', True, (255, 0, 0)) # True의 의미 : 글자 우둘투둘해지는거 막기 (안티 에일리어싱 여부)
fourfour_text = myfont.render('응~ 사사~', True, (255, 0, 0))
six_text = myfont.render('응~ 육목~', True, (255, 0, 0))

myfont2 = pygame.font.Font(font, 70)
foul_lose = myfont2.render('그렇게 두고 싶으면 그냥 둬', True, (255, 0, 0))

myfont3 = pygame.font.Font(font, 40)
AI_vs_AI_mode = myfont3.render('AI vs AI 모드', True, (255, 0, 0))

def make_board(board): # 바둑알 표시하기
    for a in range(size):
        for b in range(size):
            if board[a][b]!=0 and board[a][b]==1:
                screen.blit(black_stone,(625-18+(b-7)*dis-250,375-19+(a-7)*dis)) ## 18.75 -> 19 # 소수면 콘솔창에 경고 알림 뜸
            if board[a][b]!=0 and board[a][b]==-1:
                screen.blit(white_stone,(625-18+(b-7)*dis-250,375-19+(a-7)*dis)) ## 18.75 -> 19
         
def last_stone(xy): # 마지막 돌 위치 표시하기
    screen.blit(last_sign1,(625-18+(xy[0]-7)*dis-250,375-19+(xy[1]-7)*dis)) ## 18.75 -> 19

################################################################ main code

print("\n--Python 오목! (렌주룰)--")

exit=False # 프로그램 종료

while not exit:
    pygame.display.set_caption("오목이 좋아, 볼록이 좋아? 오목!")

    whose_turn = 1 # 누구 턴인지 알려줌 (1: 흑, -1: 백)
    turn = 0
    final_turn = None # 승패가 결정난 턴 (수순 다시보기 할 때 활용)
    max_turn = size * size

    game_selected = False # 게임 모드를 선택했나?
    game_mode = "Human_vs_AI" # 게임 모드
    AI_mode = "Deep_Learning"

    game_end = False # 게임 후 수순 다시보기 모드까지 끝났나?
    black_win = None # 흑,백 승패 여부
    game_over = False # 게임이 끝났나?
    game_review = False # 수순 다시보기 모드인가?

    record = [] # 기보 기록할 곳

    black_foul = False # 금수를 뒀나?
    before_foul = False # 한 수 전에 금수를 뒀나?
    stubborn_foul = False # 방향키를 움직이지 않고 또 금수를 두었나? (금수자리를 연타했나)
    foul_n_mok = 0 # 연속 금수 횟수
    threethree_foul = False
    fourfour_foul = False
    six_foul = False

    x=7 # 커서 좌표
    y=7
    y_win=375-19 ## 18.75 -> 19 # 커서 실제 위치
    x_win=625-18-250

    board = np.zeros([size, size]) # 컴퓨터가 이용할 바둑판
    screen.blit(board_img,(window_num, 0)) # 바둑판 이미지 추가
    screen.blit(play_button,(125, 100))
    screen.blit(selected_button2,(125, 400))
    pygame.display.update()
    
    print("\n게임 모드 선택")
    while not game_selected:
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

    pygame.mixer.music.play(-1) # -1 : 반복 재생
    print("\n게임 시작!")
    # print(difference_score_board(whose_turn, size, board), "\n") #print
    while not game_end:
        screen.blit(board_img,(window_num, 0)) ## screen.fill(0) : 검은 화면

        # 입력 받기
        for event in pygame.event.get():

            # 창 닫기(X) 버튼을 클릭했을 때
            if event.type == pygame.QUIT:
                exit=True
                game_end=True

            # 키보드를 누르고 땔 때
            elif event.type == pygame.KEYDOWN:

                # ↑ ↓ → ← 방향키
                if event.key == pygame.K_UP: 
                    if not game_review:
                        if y_win-dis > 0:
                            y_win -= dis
                            y -= 1
                    balck_stubborn = False
                elif event.key == pygame.K_DOWN:
                    if not game_review:
                        if y_win+dis < window_high-dis:
                            y_win += dis
                            y += 1
                        balck_stubborn = False
                elif event.key == pygame.K_LEFT:
                    if not game_review:
                        if x_win-dis > window_num:
                            x_win -= dis
                            x -= 1
                        balck_stubborn = False
                    
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
                        balck_stubborn = False
                    
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
                elif event.key == pygame.K_F9: # AI vs AI mode on/off
                    if AI_mode != "Human_made_Algorithms":
                        AI_mode = "Human_made_Algorithms"
                    else:
                        AI_mode = "Deep_Learning"
                elif event.key == pygame.K_ESCAPE: # 창 닫기
                    exit=True
                    game_end=True

                # Enter, Space 키
                elif event.key == pygame.K_RETURN and game_over: # 게임 종료
                        game_end=True
                        game_over=False
                elif event.key == pygame.K_SPACE and game_over: # 금수 연타했을 때 패배 창 제대로 못 보는거 방지
                    continue
                elif event.key == pygame.K_SPACE and not game_over: # 돌 두기

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
                        five = isFive(whose_turn, size, board, x, y, placed=True)
                        
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
                            if stubborn_foul or five == None: # black_고집 센 : 금수자리를 연타하는 경우 연산 생략
                                print("흑은 장목을 둘 수 없음")
                                black_foul = True
                                screen.blit(six_text,(235, 660))
                                if before_foul:
                                    foul_n_mok += 1
                            elif stubborn_foul or num_Four(whose_turn, size, board, x, y, placed=True) >= 2:
                                print("흑은 사사에 둘 수 없음")
                                black_foul = True
                                screen.blit(fourfour_text,(235, 660))
                                if before_foul:
                                    foul_n_mok += 1
                            elif stubborn_foul or num_Three(whose_turn, size, board, x, y, placed=True) >= 2:
                                print("흑은 삼삼에 둘 수 없음")
                                black_foul = True
                                screen.blit(threethree_text,(235, 660))
                                if before_foul:
                                       foul_n_mok += 1
                             
                            # 금수를 두면 무르고 다시 (연속 금수 10번까지 봐주기)
                            if black_foul: 
                                if foul_n_mok < 10:
                                    balck_stubborn = True
                                    pygame.mixer.Sound.play(black_foul_sound)
                                    before_foul = True
                                    black_foul = False
                                    board[y][x] = 0

                                    # 바둑알, 커서 위치 표시, 마지막 돌 표시 화면에 추가
                                    make_board(board)
                                    screen.blit(select,(x_win,y_win))
                                    last_stone([last_stone_xy[1], last_stone_xy[0]])
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
                                black_win=True
                                # if game_mode=="Human_vs_AI":
                                    # pygame.mixer.Sound.play(AI_lose)
                                print("흑 승리!")
                            else:
                                black_win=False
                                if not black_foul:
                                    print("백 승리!")
                        # print(difference_score_board(whose_turn, size, board), "\n") #print

                    # AI가 두기
                    if game_mode=="AI_vs_AI" or (game_mode=="Human_vs_AI" and whose_turn == 1 and not game_over):
                        
                        # 사람이 생각한 알고리즘
                        if AI_mode == "Human_made_Algorithms":
                            x, y = AI_think_win_xy(whose_turn, size, board)
                        # 딥러닝 신경망 AI
                        else:
                            x, y = network.think_win_xy(board) ### DeepConvNet -> network

                        # 선택한 좌표에 돌 두기
                        board[y][x] = whose_turn

                        record.append([y, x, whose_turn])
                        last_stone_xy = [y, x]
                        turn += 1

                        x_win = 28 + dis*x # 커서 이동
                        y_win = 27 + dis*y

                        # 오목이 생겼으면 게임 종료 신호 키기
                        if isFive(whose_turn, size, board, x, y, placed=True) == True:
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

                # 바둑알, 커서 위치 표시, 마지막 돌 표시, AI vs AI 모드 화면에 추가
                if not exit:
                    make_board(board)
                    if not game_review:
                        screen.blit(select,(x_win,y_win))
                    if turn != 0: # or event.key == pygame.K_F2 or event.key == pygame.K_F3
                        last_stone([last_stone_xy[1], last_stone_xy[0]])
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
                
print("\nGood Bye")
pygame.quit()