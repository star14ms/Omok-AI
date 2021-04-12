
################################################################ 금수 감지 code

# 5목, 흑 장목 판정 (장목: 6목 이상)
def isFive(whose_turn, size, board, x, y, placed):
    if not placed: board[y][x] = whose_turn

    # ㅡ 가로로 이어진 돌 수
    num1 = 1 # 방금 둔 1개부터 세기 시작
    for x_l in range(x-1, x-6, -1): ### x -> x-1 # 6목도 감지하기 위해 (x-6)+1까지 셈
        if (x_l == -1): break
        if board[y][x_l] == whose_turn: ### 1 -> l
            num1 += 1
        else:
            break
    for x_r in range(x+1, x+6, +1): ### x -> x+1
        if (x_r == size): break
        if board[y][x_r] == whose_turn:
            num1 += 1
        else:
            break
    if num1 == 5:
        if not placed: board[y][x] = 0
        return True

    # ㅣ 세로로 이어진 돌 수
    num2 = 1
    for y_u in range(y-1, y-6, -1):  ### x-5 -> x-6(장목 검사) -> y-6 (복붙 주의)
        if (y_u == -1): break
        if board[y_u][x] == whose_turn:
            num2 += 1
        else:
            break
    for y_d in range(y+1, y+6, +1):
        if (y_d == size): break
        if board[y_d][x] == whose_turn:
            num2 += 1
        else:
            break
    if num2 == 5:
        if not placed: board[y][x] = 0
        return True

    # \ 대각선으로 이어진 돌 수 
    num3 = 1
    x_l = x
    y_u = y ### x -> y
    for i in range(5):
        if (x_l-1 == -1) or (y_u-1 == -1): break ### or -> and (while 안에 있었을 때)
        x_l -= 1
        y_u -= 1
        if board[y_u][x_l] == whose_turn:
            num3 += 1
        else: 
            break
    x_r = x
    y_d = y
    for i in range(5):
        if (x_r+1 == size) or (y_d+1 == size): break ### != -> == (while을 나오면서)
        x_r += 1
        y_d += 1
        if board[y_d][x_r] == whose_turn:
            num3 += 1
        else:
            break
    if num3 == 5:
        if not placed: board[y][x] = 0
        return True

    # / 대각선으로 이어진 돌 수
    num4 = 1
    x_l = x
    y_d = y
    for i in range(5):
        if (x_l-1 == -1) or (y_d+1 == size): break
        x_l -= 1
        y_d += 1
        if board[y_d][x_l] == whose_turn:
            num4 += 1
        else:
            break
    x_r = x
    y_u = y
    for i in range(5):
        if (x_r+1 == size) or (y_u-1 == -1): break
        x_r += 1
        y_u -= 1
        if board[y_u][x_r] == whose_turn:
            num4 += 1
        else:
            break
    if num4 == 5:
        if not placed: board[y][x] = 0
        return True
    
    if not placed: board[y][x] = 0

    if num1 > 5 or num2 > 5 or num3 > 5 or num4 > 5:
        if whose_turn == -1: ### 1 -> -1
            return True
        else:
            return None # 흑 6목 감지
    else:
        return False

# 4 개수 (4: 다음 차례에 5를 만들 수 있는 곳) (placed : xy에 돌이 두어져 있나, 안 두어져 있다면 둬보고 검사)
def num_Four(whose_turn, size, board, x, y, placed):
    four = 0
    if not placed: board[y][x] = whose_turn # 돌 두어보기

    # ㅡ 가로 4 검사
    one_pass = False # 열린 4는 두번 세지기 때문에 연속으로 나오면 패스
    for x_r in range(x-4, x+1, +1): ### x -> x+1
        if x_r > -1 and x_r+4 < size:
            line = board[y, x_r:x_r+5]

            if sum(line) == whose_turn*4:
                if one_pass == False and ( 
                    (x_r-1 > -1 and board[y][x_r-1] != whose_turn) and ### 아직 5가 아니여야 함
                    (x_r+5 < size and board[y][x_r+5] != whose_turn)):
                    four += 1
                    one_pass = True
            else:
                one_pass = False

    # ㅣ 세로 4 검사
    one_pass = False
    for y_d in range(y-4, y+1, +1):
        if y_d > -1 and y_d+4 < size:
            line = board[y_d:y_d+5, x] ### [y, y_d:y_d+5] -> [y_d:y_d+5, x]

            if sum(line) == whose_turn*4:
                if one_pass == False and (
                    (y_d-1 > -1 and board[y_d-1][x] != whose_turn) and 
                    (y_d+5 < size and board[y_d+5][x] != whose_turn)):
                    four += 1
                    one_pass = True
            else:
                one_pass = False
    
    line = [0, 0, 0, 0, 0] # 대각선 검사할 때 이용
    
    # \ 대각선 4 검사
    one_pass = False
    x_r = x-4
    y_d = y-4
    for i in range(5):
        if x_r > -1 and x_r+4 < size and y_d > -1 and y_d+4 < size:
            for k in range(5):
                line[k] = board[y_d+k][x_r+k]

            if sum(line) == whose_turn*4: ### line.sum() -> sum(line)
                if one_pass == False and (
                    (x_r-1 > -1 and y_d-1 > -1 and board[y_d-1][x_r-1] != whose_turn) and 
                    (x_r+5 < size and y_d+5 < size and board[y_d+5][x_r+5] != whose_turn)):
                    four += 1
                    one_pass = True
            else:
                one_pass = False
        
        x_r += 1
        y_d += 1
    
    # / 대각선 4 검사
    one_pass = False
    x_r = x-4
    y_u = y+4
    for i in range(5):
        if x_r > -1 and x_r+4 < size and y_u < size and y_u-4 > -1: ### (y_u < size), (y_u+4 > -1) <-> (y_u < -1) and (y_u+4 > size) 
            for k in range(5):
                line[k] = board[y_u-k][x_r+k]

            if sum(line) == whose_turn*4:
                if one_pass == False and (
                    (x_r-1 > -1 and y_u+1 < size and board[y_u+1][x_r-1] != whose_turn) and 
                    (x_r+5 < size and y_u-5 > -1 and board[y_u-5][x_r+5] != whose_turn)):
                    four += 1
                    one_pass = True
            else:
                one_pass = False
        
        x_r += 1
        y_u -= 1

    if not placed: board[y][x] = 0 # 돌 원상태로
    return four

# 3 개수 (3: 다음 차례에 열린 4를 만들 수 있는 곳)
def num_Three(whose_turn, size, board, x, y, placed):
    three = 0
    if not placed: board[y][x] = whose_turn

    # ㅡ 가로 3 검사
    for x_r in range(x-3, x+1, +1): ### x -> x+1
        if x_r > -1 and x_r+3 < size:
            line = board[y, x_r:x_r+4]
            # 범위 4칸 중 3칸에 돌이 있을 때
            if sum(line) == whose_turn*3:
                if (x_r-1 > -1) and (x_r+4 < size):
                    # 4칸 양쪽이 열려 있고, 거짓금수가 아니면 3 한번 세기
                    if (board[y][x_r-1] == 0) and (board[y][x_r+4] == 0):
                        if (whose_turn == 1) and (x_r-2 > -1) and (x_r+5 < size): # 🟨,⬛ = x_r
                            if ((board[y][x_r-2]==whose_turn) and (x_r+6 < size) and (board[y][x_r+6]==whose_turn) or # ⚫🟡(🟨⚫⚫⚫)🟡🟡⚫
                                (board[y][x_r+5]==whose_turn) and (x_r-3 > -1) and (board[y][x_r-3]==whose_turn)):    # ⚫🟡🟡(⬛⚫⚫🟡)🟡⚫
                                continue # 양방향 장목
                            if (board[y][x_r]==0) and (board[y][x_r-2]==whose_turn) and (board[y][x_r+5]==whose_turn*-1):
                                continue # 한방향 장목 # ⚫🟡(🟨⚫⚫⚫)🟡⚪
                            if (board[y][x_r+3]==0) and (board[y][x_r-2]==whose_turn*-1) and (board[y][x_r+5]==whose_turn):
                                continue # 한방향 장목 # ⚪🟡(⬛⚫⚫🟡)🟡⚫
                        three += 1
                        break # 열린 3은 두번 세지기 때문에 라인 당 한번만 세기

    # ㅣ 세로 3 검사
    for y_d in range(y-3, y+1, +1):
        if y_d > -1 and y_d+3 < size:
            line = board[y_d:y_d+4, x]

            if sum(line) == whose_turn*3:
                if (y_d-1 > -1) and (y_d+4 < size):

                    if (board[y_d-1][x] == 0) and (board[y_d+4][x] == 0):
                        if (whose_turn == 1) and (y_d-2 > -1 and y_d+5 < size):
                            if ((board[y_d-2][x]==whose_turn) and (y_d+6 < size) and (board[y_d+6][x]==whose_turn) or
                                (board[y_d+5][x]==whose_turn) and (y_d-3 > -1) and (board[y_d-3][x]==whose_turn)):
                                continue
                            if (board[y_d][x]==0) and (board[y_d-2][x]==whose_turn) and (board[y_d+5][x]==whose_turn*-1): 
                                continue
                            if (board[y_d+3][x]==0) and (board[y_d-2][x]==whose_turn*-1) and (board[y_d+5][x]==whose_turn):
                                continue
                        three += 1
                        break

    line = [0, 0, 0, 0] # 대각선 검사할 때 이용

    # \ 대각선 3 검사
    x_r = x-3 ### -4 -> -3 (복붙주의)
    y_d = y-3
    for i in range(4):
        if x_r > -1 and x_r+3 < size and y_d > -1 and y_d+3 < size:
            for k in range(4):
                line[k] = board[y_d+k][x_r+k]

            if sum(line) == whose_turn*3:
                if (x_r-1 > -1) and (y_d-1 > -1) and (x_r+4 < size) and (y_d+4 < size):
                    
                    if (board[y_d-1][x_r] == 0) and (board[y_d+4][x_r] == 0):
                        if (whose_turn == 1) and (x_r-2 > -1) and (y_d-2 > -1) and (x_r+5 < size) and (y_d+5 < size):
                            if ((board[y_d-2][x_r-2]==whose_turn) and (x_r+6 < size) and (y_d+6 < size) and (board[y_d+6][x_r+6]==whose_turn) or
                                (board[y_d+5][x_r+5]==whose_turn) and (x_r-3 > -1) and (y_d-3 > -1) and (board[y_d-3][x_r-3]==whose_turn)):
                                continue
                            if (board[y_d][x_r]==0) and (board[y_d-2][x_r-2]==whose_turn) and (board[y_d+5][x_r+5]==whose_turn*-1): 
                                continue
                            if (board[y_d+3][x_r+3]==0) and (board[y_d-2][x_r-2]==whose_turn*-1) and (board[y_d+5][x_r+5]==whose_turn):
                                continue
                        three += 1
                        break
        x_r += 1
        y_d += 1

    # / 대각선 3 검사
    x_r = x-3
    y_u = y+3
    for i in range(4):
        if x_r > -1 and x_r+3 < size and y_u+1 < size and y_u-3 > -1: ### (y_u-1 > -1), (y_u+3 < size) -> (y_u+1 < size), (y_u-3 > -1)
            for k in range(4):
                line[k] = board[y_u-k][x_r+k]

            if sum(line) == whose_turn*3:
                if (x_r-1 > -1) and (x_r+4 < size) and (y_u+1 < size) and (y_u-4 > -1): ### y_u-1, y_u+4 -> y_u+1, y_u-4
                    
                    if (board[y_u+1][x_r-1] == 0) and (board[y_u-4][x_r+4] == 0):
                        if (whose_turn == 1) and (x_r-2 > -1) and (y_u+2 < size) and (x_r+5 < size) and (y_u-5 > -1):
                            if ((board[y_u+2][x_r-2]==whose_turn) and (x_r+6 < size) and (y_u-6 > -1) and (board[y_u-6][x_r+6]==whose_turn) or
                                (board[y_u-5][x_r+5]==whose_turn) and (x_r-3 > -1) and (y_u+3 < size) and (board[y_u+3][x_r-3]==whose_turn)):
                                continue
                            if (board[y_u][x_r]==0) and (board[y_u+2][x_r-2]==whose_turn) and (board[y_u-5][x_r+5]==whose_turn*-1): 
                                continue
                            if (board[y_u-3][x_r+3]==0) and (board[y_u+2][x_r-2]==whose_turn*-1) and (board[y_u-5][x_r+5]==whose_turn):
                                continue
                        three += 1
                        break
        x_r += 1
        y_u -= 1

    if not placed: board[y][x] = 0
    return three
