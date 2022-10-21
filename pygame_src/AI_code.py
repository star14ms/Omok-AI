import numpy as np
import random # 점수가 같은 좌표들 중 하나 고르기
from pygame_src.foul_detection import isFive, num_Four, num_Three, isFoul
from modules.common.util import bcolors
from modules.test import print_board

def AI_think_win_xy(whose_turn, board, all_molds, mok_value=1.2, verbose=False, return_is_necessary=False):
    board = board.copy()
    size = board.shape[0]
    is_necessary = True
    whose_think = whose_turn

    # 무조건 둬야 하는 좌표 감지 (우선순위 1~4위)
    self_5_xy = canFive(whose_think, whose_turn, board)     # 1.자신의 5자리
    opon_5_xy = canFive(whose_think, whose_turn*-1, board)  # 2.상대의 5자리
    self_4_xys = canFour(whose_think, whose_turn, board)    # 3.자신의 열린4 자리 (최대 2곳)
    opon_4_xys = canFour(whose_think, whose_turn*-1, board) # 4.상대의 열린4 자리 (최대 2곳)
    
    # self_5_xy, opon_5_xy, self_4_xys, opon_4_xys = [None], [None], [None], [None]
    # 가장 높은 가치의 좌표 감지
    scores = difference_score_board(whose_turn, board, mok_value)
    if all_molds and len(np.where(board != 0)[0]) == 2: ### ([x1, x2], [y1, y2])
        xy_most_high_list = xy_random_mold(board)
    else:
        xy_most_high_list = xy_most_high_value(board, scores)
    expect_xy = xy_most_high_list[0]
    
    # 우선 순위가 가장 높은 좌표를 선택
    if self_5_xy[0] != None: # 1위
        x, y = self_5_xy[0], self_5_xy[1]
    elif opon_5_xy[0] != None: # 2위
        x, y = opon_5_xy[0], opon_5_xy[1]
    elif self_4_xys[0] != None: # 3위
        
        x1, y1 = self_4_xys[0][0], self_4_xys[0][1] ### value_board는 [y, x] 형태

        # 떨어진 3일 때
        if len(self_4_xys) == 1:
            x, y = x1, y1
        else: # 열린 3일 때

            # 열린 3은 열린 4를 만드는 곳이 2곳임 (비교 필요)
            x2, y2 = self_4_xys[1][0], self_4_xys[1][1]

            # 기대 점수가 가장 높은 좌표와 같은 좌표를 선택 # 우선순위 1위
            if expect_xy == [x1, y1]:
                x, y = x1, y1
            elif expect_xy == [x2, y2]:
                x, y = x2, y2
            else: # 돌들의 평균 위치에 더 가까운 좌표를 선택 (주변에 돌이 더 많은 쪽) # 우선순위 2위
                selected_xy = select_xy_more_close([x1, y1], [x2, y2], board, scores)
                if selected_xy:
                    x, y = selected_xy[0], selected_xy[1]
                else: # 중앙에 더 가까운 좌표를 선택 # 우선순위 3위
                    selected_xy = select_xy_more_center([x1, y1], [x2, y2], scores)
                    if selected_xy:
                        x, y = selected_xy[0], selected_xy[1]
                    else:
                        x, y = x1, y1
    elif opon_4_xys[0] != None: # 4위
        
        x1, y1 = opon_4_xys[0][0], opon_4_xys[0][1]
        
        if len(opon_4_xys) == 1: ### white -> black 복붙
            x, y = x1, y1
        else:
            x2, y2 = opon_4_xys[1][0], opon_4_xys[1][1]
            
            if expect_xy == [x1, y1]:
                x, y = x1, y1
            elif expect_xy == [x2, y2]:
                x, y = x2, y2
            else:
                selected_xy = select_xy_more_close([x1, y1], [x2, y2], board, scores)
                if selected_xy:
                    x, y = selected_xy[0], selected_xy[1]
                else:
                    selected_xy = select_xy_more_center([x1, y1], [x2, y2], scores)
                    if selected_xy:
                        x, y = selected_xy[0], selected_xy[1]
                    else:
                        x, y = x1, y1
    else: # 우선순위 5 ~ 7위
        xy_selected = False

        # 5. 4-4, 4-3, 3-3자리 선택 (우선순위 5위) 
        for y_a in range(size):
            for x_a in range(size):
                
                if board[y_a][x_a] == 0: ### 돌이 이미 두어진 곳은 검사할 필요 없는 데다, 검사 후 돌이 사라짐
                    self_4 = num_Four(whose_turn, board, x_a, y_a, placed=False) # 자신의 4 개수
                    opon_4 = num_Four(whose_turn*-1, board, x_a, y_a, placed=False) # 자신의 3 개수
                    self_3 = num_Three(whose_turn, board, x_a, y_a, placed=False) # 상대의 4 개수
                    opon_3 = num_Three(whose_turn*-1, board, x_a, y_a, placed=False) # 상대의 3 개수
                    # print(isFive(whose_turn, board, x_a, y_a, placed=False),
                    #     num_Four(whose_turn, board, x_a, y_a, placed=False),
                    #     num_Three(whose_turn, board, x_a, y_a, placed=False))
                    # print(x_a+1, y_a+1, self_4, self_3, opon_4, opon_3)
                    ### opon_4 <-> self_3, whose_turn*-1 <-> whose_turn 바뀜
                    if whose_turn == -1 and self_4 >= 2: # 백의 4-4 공격
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    elif whose_turn*-1 == -1 and opon_4 >= 2 and ( # 흑이 백의 4-4, 방어
                        isFive(whose_turn, board, x_a, y_a, placed=False) != None and
                        num_Four(whose_turn, board, x_a, y_a, placed=False) < 2 and
                        num_Three(whose_turn, board, x_a, y_a, placed=False) < 2):
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    
                    elif whose_turn == 1 and self_4 == 1 and self_3 == 1: # 흑의 4-3 공격
                        x, y = x_a, y_a
                        xy_selected = True
                        break                                   
                    elif whose_turn == -1 and self_4 == 1 and self_3 == 1: # 백의 4-3 공격
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    elif whose_turn*-1 == -1 and opon_4 == 1 and opon_3 == 1 and ( # 흑이 백의 4-3 방어
                        isFive(whose_turn, board, x_a, y_a, placed=False) != None and
                        num_Four(whose_turn, board, x_a, y_a, placed=False) < 2 and
                        num_Three(whose_turn, board, x_a, y_a, placed=False) < 2):
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    elif whose_turn*-1 == 1 and opon_4 == 1 and opon_3 == 1: # 백이 흑의 4-3 방어
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    
                    elif whose_turn == -1 and self_3 >= 2: # 백의 3-3 공격
                        x, y = x_a, y_a
                        xy_selected = True
                        break
                    elif whose_turn*-1 == -1 and opon_3 >= 2 and ( # 흑이 백의 3-3 방어 ### self_3 -> opon_3
                        isFive(whose_turn, board, x_a, y_a, placed=False) != None and
                        num_Four(whose_turn, board, x_a, y_a, placed=False) < 2 and
                        num_Three(whose_turn, board, x_a, y_a, placed=False) < 2):
                        x, y = x_a, y_a
                        xy_selected = True
                        break
            
            if xy_selected: 
                break

        # 6. 가장 높은 가치를 가진 좌표를 선택 (우선순위 6위)
        if not xy_selected and board[expect_xy[1], expect_xy[0]] == 0:
            x, y = expect_xy[0], expect_xy[1]
            xy_selected = True
            is_necessary = False
            # if (whose_turn == -1) or (isFive(whose_turn, board, expect_xy[0], expect_xy[1], placed=False) != None and
            #     num_Four(whose_turn, board, expect_xy[0], expect_xy[1], placed=False) < 2 and
            #     num_Three(whose_turn, board, expect_xy[0], expect_xy[1], placed=False) < 2):
            #     x, y = expect_xy[0], expect_xy[1]

        # 7. 둘 곳이 마땅히 없을 때 빈공간을 선택 (우선순위 7위)
        if not xy_selected:
            is_necessary = False
            for y_b in range(size):
                for x_b in range(size):
                    if board[y_b][x_b] == 0:
                        x, y = x_b, y_b
                        xy_selected = True
                        break
                if xy_selected: break ### (x == x_b) and (y == y_b) x 맨 오른쪽 끝에선 맞을 수밖에 없음

    # 연결 기대 점수판, 기대점수 1위, 최종 우선순위 1위 좌표 출력
    if verbose:
        # formatter = {'all':lambda _x: ( # 세자리 출력(100) 방지
            # bcolors.according_to_score(_x) + str(int(np.minimum(_x, 99))).rjust(2) + bcolors.ENDC)}
        formatter = {'all':lambda _x: str(int(np.minimum(_x, 99))).rjust(2)}
        np.set_printoptions(linewidth=np.inf, formatter=formatter)
        scores_nomalized = ( (scores - np.min(scores)) / (scores.ptp() + 1e-7) * 100)
        # print(scores[5, 6], scores[5, 8], scores[8, 6], scores[8, 8])
        print("\n"+"="*40+"\n")
        str_self_opon = "흑/백" if whose_turn==1 else "백/흑"
        print(f"4목 {str_self_opon} {self_5_xy}/{opon_5_xy}, 3목 {str_self_opon} {self_4_xys}/{opon_4_xys}")
        print(scores_nomalized, "\n")
        # print_board(board, scores_nomalized, mode="QnAI")
    
        if all_molds and len(np.where(board != 0)[0]) == 2:
            if whose_turn*-1 in (board[7, 6], board[7, 8], board[6, 7], board[8, 7]):
                print("직접 주형 랜덤 선택")
            elif whose_turn*-1 in (board[6, 6], board[6, 8], board[8, 6], board[8, 8]):
                print("간접 주형 랜덤 선택")
        elif len(xy_most_high_list[1]) > 1:
            print("기대점수 공동 1위:", end=" ")
            for xy in xy_most_high_list[1]:
                print("["+str(xy[0]+1) +","+ str(xy[1]+1)+"]", end=" ")
            print("랜덤 선택")
    
        print("기대점수 1위: x="+str(expect_xy[0]) + " y="+str(expect_xy[1]), end=", ")
        print(f"{round(scores[expect_xy[1], expect_xy[0]], 3)}점")
        print("우선순위 1위: x="+str(x) + " y="+str(y), end=", ")
        print(f"{round(scores[y][x], 3)}점\n")
        # print("="*40)

    if not return_is_necessary:
        return x, y
    else:
        return x, y, is_necessary

################################################################ AI code1 (무조건 둬야하는 수 찾기)

# 5목 만드는 좌표가 있으면 줌 (흑 차례고 금수일 땐 주지 않음, 범위: 바둑판 전체)
def canFive(whose_think, whose_turn, board):
    size = board.shape[0]
    # 가로 감지
    for y in range(size):
        for x in range(size - 4):
            # 연속된 5칸을 잡아 그 중 4칸이 자기 돌로 차 있으면
            line = board[y, x:x+5]
            if sum(line) == whose_turn * 4:
                #  흑 금수는 제외하고, 나머지 한 칸 반환
                for i in range(5):
                    if board[y][x+i] == 0: ### 먼저 빈자리를 찾고, 그곳을 검사해야 함
                        if whose_think == 1 and isFoul(x+i, y, board, whose_think) != "No":
                            continue ### x -> x+i, 백이 둘 때, 흑 장목검사는 다른 라인들을 해야 함
                        return [x+i, y]

    # 세로 감지
    for y in range(size - 4):
        for x in range(size):

            line = board[y:y+5, x]
            if sum(line) == whose_turn * 4:

                for i in range(5):
                    if board[y+i][x] == 0:
                        if whose_think == 1 and isFoul(x, y+i, board, whose_think) != "No":
                            continue
                        return [x, y+i]
    
    # 대각선 감지
    line = [0, 0, 0, 0, 0] # 대각선 감지할 때 이용
    for y in range(size - 4):
        for x in range(size - 4):

            # \ 검사
            for i in range(5):
                line[i] = board[y+i][x+i]
            if sum(line) == whose_turn * 4:

                for i in range(5):
                    if board[y+i][x+i] == 0:
                        if whose_think == 1 and isFoul(x+i, y+i, board, whose_think) != "No":
                            continue
                        return [x+i, y+i]

            # / 검사
            for i in range(5):
                line[i] = board[y+4-i][x+i]
            if sum(line) == whose_turn * 4:

                for i in range(5):
                    if board[y+4-i][x+i] == 0:
                        if whose_think == 1 and isFoul(x+i, y+4-i, board, whose_think) != "No":
                            continue
                        return [x+i, y+4-i]
    
    return [None]

# 열린 4목 만드는 좌표가 있으면 줌 (흑 차례고 금수일 땐 주지 않음, 범위: 바둑판 전체)
def canFour(whose_think, whose_turn, board):
    size = board.shape[0]
    canFour_xy_list = []

    # 가로 감지
    for y in range(size):
        for x in range(size - 3):
            # 연속된 4칸을 잡아 그 중 3칸이 자기 돌로 차 있으면
            line = board[y, x:x+4]
            if sum(line) == whose_turn * 3:
                # 나머지 한 칸을 채웠을 때 열린 4가 되면
                if x-1 > -1 and x+4 < size:
                    if board[y][x-1] == 0 and board[y][x+4] == 0:
                        # 흑 금수는 제외하고, 나머지 한 칸 반환
                        for i in range(4):
                            if board[y][x+i] == 0:
                                if whose_think == 1 and isFoul(x+i, y, board, whose_think) != "No":
                                    continue
                                canFour_xy_list.append([x+i, y])
    
    if len(canFour_xy_list) == 2: # 같은 라인에서 4칸 차이나는 두 좌표가 생기면 바로 반환
        return canFour_xy_list 
    horizontal_four_num = len(canFour_xy_list)

    # 세로 감지
    for y in range(size - 3):
        for x in range(size):

            line = board[y:y+4, x]
            if sum(line) == whose_turn * 3:

                if y-1 > -1 and y+4 < size:
                    if board[y-1][x] == 0 and board[y+4][x] == 0:

                        for i in range(4):
                            if board[y+i][x] == 0:
                                if whose_think == 1 and isFoul(x, y+i, board, whose_think) != "No":
                                    continue
                                canFour_xy_list.append([x, y+i])
    
    if len(canFour_xy_list) == horizontal_four_num + 2:
        return canFour_xy_list[-2:]
    vertical_four_num = len(canFour_xy_list)
        
    # 대각선 \ 감지
    line = [0, 0, 0, 0] # 대각선 감지할 때 이용
    for y in range(size - 3):
        for x in range(size - 3):
            
            for n in range(4):
                line[n] = board[y+n][x+n]
            if sum(line) == whose_turn * 3:

                if x-1 > -1 and x+4 < size and y-1 > -1 and y+4 < size:
                    if board[y-1][x-1] == 0 and board[y+4][x+4] == 0:

                        for i in range(4):
                            if board[y+i][x+i] == 0:
                                if whose_think == 1 and isFoul(x+i, y+i, board, whose_think) != "No":
                                    continue
                                canFour_xy_list.append([x+i, y+i])
    
    if len(canFour_xy_list) == vertical_four_num + 2:
        return canFour_xy_list[-2:]
    diagonal_four_num1 = len(canFour_xy_list)
    
    # 대각선 / 감지        
    for y in range(size - 3):
        for x in range(size - 3):        
        
            for n in range(4):
                line[n] = board[y+n][x+3-n]
            if sum(line) == whose_turn * 3: ### 4 -> 3 복붙 주의

                if x+3+1 < size and x+3-4 > -1 and y-1 > -1 and y+4 < size: ### x+1 > size -> x+1 < size
                    if board[y-1][x+3+1] == 0 and board[y+4][x+3-4] == 0: ### x+1, x-4 -> x+3+1, x+3-4 (현재 x의 +3이 기준)
                        
                        for i in range(4):
                            if board[y+i][x+3-i] == 0:
                                if whose_think == 1 and isFoul(x+3-i, y+i, board, whose_think) != "No":
                                    continue
                                canFour_xy_list.append([x+3-i, y+i])
    
    if len(canFour_xy_list) == diagonal_four_num1 + 2:
        return canFour_xy_list[-2:]
    
    if len(canFour_xy_list) == 0:
        canFour_xy_list.append(None)
    return canFour_xy_list

################################################################ AI code2 (각 좌표의 가치 보드 만들기)

# 각 좌표의 가치를 보드로 줌 (현재 보드의 상태 뿐만 아니라, 각 좌표마다 돌을 뒀다 가정하고도 계산 가능) (placed : 돌을 두기 전/후 구별)
def whose_score_board(whose_turn, board, placed, mok_value):
    size = board.shape[0]
    whose_omok_score_board = np.zeros([size, size])
    # mok_value: 1목 당 제곱할 인자
    
    for y in range(size):
        for x in range(size):
            # 돌이 이미 놓인 자리는 0점
            if board[y][x] != 0:
                continue
            # 각 좌표에 돌을 놓아볼 때 금수 검사
            if placed:
                board[y][x] = whose_turn ### == -> =

                # 흑 금수 자리면 -2점 #** -1점으로 했더니 16진수로 표기될 때 있음
                if num_Four(1, board, x, y, placed=True) >= 2:
                    whose_omok_score_board[y][x] = -2
                    board[y][x] = 0 ### continue 전에도 바둑돌을 다시 물러야 함
                    continue
                if num_Three(1, board, x, y, placed=True) >= 2:
                    whose_omok_score_board[y][x] = -2
                    board[y][x] = 0
                    continue
                if isFive(1, board, x, y, placed=True) == None: 
                    whose_omok_score_board[y][x] = -2
                    board[y][x] = 0
                    continue

            value = 1
            
            # ㅡ 가로 검사
            for x_r in range(x-4, x+1, +1):
                if x_r > -1 and x_r+4 < size:
                    block = False ### 상대에게 막혔으면 그 범위는 0점
                    n_mok = 0
                    # 범위 5칸을 잡아 자기 돌의 개수를 셈 (상대편 돌이 있으면 넘어감)
                    line = board[y, x_r:x_r+5]
                    for k in range(5):
                        if line[k] == whose_turn*-1:
                            block = True
                            break
                        elif line[k] == whose_turn:
                            n_mok += 1
                    # 돌의 개수에 따라 일정 점수를 더함
                    if not block:
                        value *= mok_value**n_mok # 2**(-5+n_mok)*100 # 2**(n_mok)
                    # if x == 8 and y == 7: print(2**(-5+n_mok)*100, 2)
            
            # ㅣ 세로 검사
            for y_d in range(y-4, y+1, +1):
                if y_d > -1 and y_d+4 < size:
                    block = False
                    n_mok = 0
                    
                    line = board[y_d:y_d+5, x]
                    for k in range(5):
                        if line[k] == whose_turn*-1:
                            block = True
                            break
                        elif line[k] == whose_turn:
                            n_mok += 1
                    
                    if not block:
                        value *= mok_value**n_mok
                    # if x == 8 and y == 7: print(2**(-5+n_mok)*100, 2)
            line = [0, 0, 0, 0, 0] # 대각선 검사할 때 이용
            
            # \ 대각선 검사
            x_r, y_d = x-4, y-4
            for i in range(5):
                if x_r > -1 and x_r+4 < size and y_d > -1 and y_d+4 < size:
                    block = False
                    n_mok = 0
                    
                    for k in range(5):
                        line[k] = board[y_d+k][x_r+k]
                        if line[k] == whose_turn*-1:
                            block = True
                            break
                        elif line[k] == whose_turn:
                            n_mok += 1
                    
                    if not block:
                        value *= mok_value**n_mok
                    # if x == 8 and y == 7: print(2**(-5+n_mok)*100, 3)
                x_r += 1 ### 점수 좌우 대칭x 이유 ->  else: continue에도 x_r, y_d += 1을 추가해야함, 애초에 continue가 필요 없음
                y_d += 1
            
            # / 대각선 검사
            x_r, y_u = x-4, y+4
            for i in range(5):
                if x_r > -1 and x_r+4 < size and y_u < size and y_u-4 > -1:
                    block = False
                    n_mok = 0
                    
                    for k in range(5):
                        line[k] = board[y_u-k][x_r+k]
                        if line[k] == whose_turn*-1:
                            block = True
                            break
                        elif line[k] == whose_turn:
                            n_mok += 1
                    
                    if not block:
                        value *= mok_value**n_mok
                    # if x == 8 and y == 7: print(2**(-5+n_mok)*100, 4)
                x_r += 1
                y_u -= 1
            
            whose_omok_score_board[y][x] = round(value, 1)
            board[y][x] = 0
    
    return whose_omok_score_board

# 각 좌표에 돌을 두었을 때 가치 변화량을 보드로 줌
def whose_difference_score_board(whose_turn, board, mok_value):
    size = board.shape[0]
    # 돌을 두기 전/후의 점수 보드 만들기
    before_placing_score_board = whose_score_board(whose_turn, board, placed=False, mok_value=mok_value) # 돌을 두기 전
    after_placing_score_board = whose_score_board(whose_turn, board, placed=True, mok_value=mok_value) # 돌을 둔 후
    # print(before_placing_score_board, "\n")
    # print(after_placing_score_board, "\n")
    
    # 돌을 두기 전/후의 점수 차이 보드 만들기
    whose_difference_score_board = np.zeros([size, size])
    for y in range(size):
        for x in range(size): # 둔 후 가치 - 두기 전 가치
            whose_difference_score_board[y][x] = after_placing_score_board[y][x] - before_placing_score_board[y][x] 
    return whose_difference_score_board

# 흑/백 양쪽의 가치 변화량 보드를 합산한 보드를 줌 (각 좌표의 최종 가치 보드) # 신 버전 #++ whose_turn 불필요
def difference_score_board(whose_turn, board, mok_value):
    size = board.shape[0]
    # 각자의 점수 보드 만들기
    oneself_score_board = whose_difference_score_board(whose_turn, board, mok_value)
    opponent_score_board = whose_difference_score_board(whose_turn*-1, board, mok_value)
    # print(f"\n{'흑 점수 변화량' if whose_turn == 1 else '백 점수 변화량'}")
    # print(oneself_score_board, "\n")
    # print(f"{'백 점수 변화량' if whose_turn == 1 else '흑 점수 변화량'}")
    # print(opponent_score_board, "\n")
    # print("각 좌표의 최종 가치")

    # 합산한 보드 반환
    total_score_board = np.zeros([size, size]) ### np.zeros([size, size], "\n") 이렇게 하면 음수일 때 오류남
    for y in range(size):
        for x in range(size):
            total_score_board[y][x] = oneself_score_board[y][x] + opponent_score_board[y][x]
    return total_score_board

# 보드에서 제일 높은 점수를 가지는 좌표를 줌
def xy_most_high_value(board, scores):
    size = board.shape[0]
    xy_most_high = [[0, 0]] # 기대점수 1위 좌표(들)
    value_most_high = 0 # 1위 점수
    
    # 바둑판의 모든 좌표를 훑어서 기대점수 1위 좌표 찾기
    for focus_y in range(size):
        for focus_x in range(size):
            
            # (1위 점수 < 현재 좌표의 점수)일 때, 현재 좌표를 1위로 (1.더 높은 점수)
            if value_most_high < scores[focus_y][focus_x]:
                
                value_most_high = scores[focus_y][focus_x]
                xy_most_high = [[focus_x, focus_y]]
            
            # (1위 점수 = 현재 좌표의 점수)일 때
            elif value_most_high == scores[focus_y][focus_x]:
                
                selected_xy = select_xy_more_close([focus_x, focus_y], xy_most_high[0], board, scores)
                
                if selected_xy == 0: # 바둑판이 비어있을 때 중앙 반환
                    return[[7,7],[[7,7]]]
                elif selected_xy == 1: # 간접주형 사라지는거 방지 (초반 직접/간접주형 모두 가능)
                    xy_most_high.append([focus_x, focus_y])

                # 현재 좌표가 돌들의 평균 위치에 더 가까우면 현재 좌표를 1위로 (간접주형 사라짐) (2.주변에 돌이 더 많은 쪽)
                elif selected_xy == [focus_x, focus_y]: 
                    xy_most_high = [[focus_x, focus_y]]
                    
                # 평균 좌표까지의 거리가 같으면 중앙에 더 가까운 쪽을 1위로 (3.중앙에 가까운 쪽)
                elif selected_xy == None:
                    select_xy = select_xy_more_center([focus_x, focus_y], xy_most_high[0], scores)

                    if select_xy == [focus_x, focus_y]:
                        xy_most_high = [[focus_x, focus_y]]
                    
                    # 중앙까지의 거리가 같으면 현재 좌표를 1위 리스트에 추가 (4.랜덤으로 뽑기)
                    elif select_xy == None:
                        xy_most_high.append([focus_x, focus_y])

    # 공동 1위가 있을 때 랜덤으로 하나 고르기
    ran_num = random.randrange(0, len(xy_most_high))
    xy_win = xy_most_high[ran_num]

    return [xy_win, xy_most_high]

# 랜덤 3번째 수 주형 선택
def xy_random_mold(board):
    xy_molds = [[7+i, 7+j] for i in range(-2, 3, 1) for j in range(-2, 3, 1)]
    xy_mold = None
    while xy_mold == None or board[xy_mold[1], xy_mold[0]] != 0:
        ran_num = random.randrange(0, len(xy_molds))
        xy_mold = xy_molds[ran_num]

    return [xy_mold, [xy_mold]]

################################################################ AI code3 (상대 3을 막을 때, 두 좌표중 하나를 선택)

# 두 좌표 중 돌들의 평균 위치에 더 가까운 좌표를 내보냄
def select_xy_more_close(xy1, xy2, board, scores):
    size = board.shape[0]
    sum_x, sum_y = 0, 0 # 모든 돌의 x, y좌표값의 합
    num_stones = 0 # 바둑판에 놓인 돌 개수
    
    for focus2_y in range(size): ### focus -> focus2 새로운 변수
        for focus2_x in range(size):
            if board[focus2_y][focus2_x] == -1 or board[focus2_y][focus2_x] == 1: 
                sum_x += focus2_x
                sum_y += focus2_y
                num_stones += 1 ### value_board로 돌의 유무를 확인하면 반올림 0이 생겼을 때 돌인줄 알음
    
    if num_stones == 0:
        return 0
    elif (num_stones == 1 and scores[7][7] == 0): ## or num_stones == 3 (돌 두개 막기)
        return 1
    avrg_x, avrg_y = round(sum_x/num_stones, 2), round(sum_y/num_stones, 2) # 전체 바둑돌의 평균 좌표
    
    if (avrg_x-xy1[0])**2 + (avrg_y-xy1[1])**2 < (avrg_x-xy2[0])**2 + (avrg_y-xy2[1])**2:
        return xy1
    elif (avrg_x-xy1[0])**2 + (avrg_y-xy1[1])**2 > (avrg_x-xy2[0])**2 + (avrg_y-xy2[1])**2:
        return xy2
    else:
        return None

# 두 좌표 중 중앙에 더 가까운 좌표를 내보냄
def select_xy_more_center(xy1, xy2, scores, size=15):
    
    if ((size//2+1)-xy1[0])**2 + ((size//2+1)-xy1[1])**2 < ((size//2+1)-xy2[0])**2 + ((size//2+1)-xy2[1])**2:
        return xy1
    elif ((size//2+1)-xy1[0])**2 + ((size//2+1)-xy1[1])**2 > ((size//2+1)-xy2[0])**2 + ((size//2+1)-xy2[1])**2:
        return xy2
    else:
        return None
        
