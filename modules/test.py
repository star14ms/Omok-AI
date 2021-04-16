import numpy as np
import sys
sys.path.append(".")
from modules.common.functions import softmax, cross_entropy_error
from modules.common.util import bcolors
import random
import math

def first_place_yx(array, find_all=False):
    if not find_all: # return y, x
        return array.argmax()//15, array.argmax()%15
    
    arr2dim = array.reshape(15, 15)
    first_place_indexs = np.argwhere(arr2dim == np.amax(arr2dim)).tolist()
    first_place_yx_list = []
    # print(first_place_indexs)

    for fp_index in first_place_indexs:
        if fp_index not in first_place_yx_list:
            first_place_yx_list.append(fp_index)

    return first_place_yx_list # return [[y1, x1], [y2, x2], ...]

def print_board(*args, mode="QnAI", num_answers=1):
    
    if len(args) == 1:
        Q = args[0]
        if mode!="Q":
            mode = "Q"
            print("배열을 하나 받으면 Q모드 출력만 가능")
    
    elif len(args) == 2:
        Q, A = args[0], args[1]

        if mode!="QnAI" and mode!="QnA":
            print("배열을 두개 받으면 QnA 또는 QnAI모드 출력만 가능")
            if 99 < A.sum() < 101: ### A를 먼저 선언해야 함
                mode = "QnAI"
            else:
                mode = "QnA"

    elif len(args) == 3:
        Q, A, T = args[0], args[1], args[2]
        if 99 < T.sum() < 101:
            A, T = T, A

        if mode!="QnA_AI":
            print("배열을 세개 받으면 QnA_AI모드 출력 가능")

    else:
        print("Error: *args에는 0~1개의 배열만 받을 수 있음")
        return
    
    # print(mode)
    Q = Q.reshape(15, 15)
    t_x, t_y = -1, -1
    a_x, a_y = -1, -1
    
    if (mode=="QnA" or mode=="QnAI" or mode=="QnA_AI"):
        if A.shape[0] == 1:
            A = A.reshape(15, 15)
        a_y, a_x = A.argmax()//15, A.argmax()%15
    
    if mode=="QnA_AI":
        if T.shape[0] == 1:
            T = T.reshape(15, 15)
        t_y, t_x = T.argmax()//15, T.argmax()%15
    
    if "AI" in mode and A.sum().round() != 100:
        A = A*100 # 백분율
    
    len_x, len_y = Q.shape[-1], Q.shape[-2]

    for y in range(len_y): ### y, len_x 
        for x in range(len_x): ### x, len_y
            
            if Q[y][x] == 0: # or (mode=="QnAI" and x==a_x and y==a_y)
                if mode=="Q":
                    print("🟤", end="")
                elif mode=="QnA":
                    print("{}".format("🟤" if A[y][x] == 0 else "🟣"), end="")
                else:
                    chance = A[y][x]*num_answers
                    if 0.5 <= chance < 5: ### 0 <
                        print("{}".format("🔴" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟥"), end="")
                    elif 5 <= chance < 20:
                        print("{}".format("🟠" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟧"), end="")
                    elif 20 <= chance < 50:
                        print("{}".format("🟡" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟨"), end="")
                    elif 50 <= chance < 80:
                        print("{}".format("🟢" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟩"), end="")
                    elif 80 <= chance < 95:
                        print("{}".format("🔵" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟦"), end="")
                    elif 95 <= chance:
                        print("{}".format("🟣" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟪"), end="")
                    else:
                        print("{}".format("🟤" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "🟫"), end="")
 
            elif Q[y][x] == 1:
                print("{}".format("⚫" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "⬛"), end="") 
            elif Q[y][x] == -1:
                print("{}".format("⚪" if (x!=a_x or y!=a_y) and (x!=t_x or y!=t_y) else "⬜"), end="") 
            else:
                print("\nprint_question_board() 오류!")
        print()

    # if mode == "QnAI":
    #     print("\n🟤< 0.5% <🔴< 5% <🟠< 20% <🟡< 50% <🟢< 80% <🔵< 95% <🟣\n")

# 문제를 하나 골라 테스트 (index: 테스트 할 문제 위치)
def test_pick(network, x_data, t_data, index):
    x = x_data[index:index+1]
    t = t_data[index:index+1]

    scores = network.predict(x)
    a_x, a_y = scores.argmax()%15, scores.argmax()//15

    scores_nomalized = ( (scores - np.min(scores)) / scores.ptp() * 100).reshape(15, 15)

    winning_odds = (softmax(scores)*100).reshape(15, 15) ### x.round(2) np.round_(x, 2)
    # winning_odds = np.where(winning_odds==winning_odds[a_y, a_x], winning_odds, 0)

    t_x, t_y = t.argmax()%15, t.argmax()//15
    # t_yxs = first_place_yx(t, find_all=True) # 답 여러개일 때
    # t_yx_chances = []
    # for idx, _ in enumerate(t_yxs):
    #     chance = math.floor(winning_odds[t_yxs[idx][0], t_yxs[idx][1]]*100)/100
    #     t_yx_chances.append(chance)
    
    print("\n=== 점수(scores) ===")
    np.set_printoptions(linewidth=np.inf, formatter={'all':lambda x: ( # 세자리 출력(100) 방지
    bcolors.according_to_score(x) + str(int(np.minimum(x, 99))).rjust(2) + bcolors.ENDC)})
    print(scores_nomalized) # .astype(int)

    print("\n=== 각 자리의 확률 분배(%) ===")
    np.set_printoptions(linewidth=np.inf, formatter={'all':lambda x: (
    bcolors.according_to_chance(x) + str(int(np.minimum(x, 99))).rjust(2) + bcolors.ENDC)})
    print(winning_odds)

    print(f"\n=== Q-{index} AI's Answer ===")
    print_board(x, winning_odds, t, mode="QnA_AI") # len(t_yxs) 답 여러개일 때

    print(f"정답 좌표: x={str(t_x).rjust(2)},y={str(t_y).rjust(2)} ({ math.floor(winning_odds[t_y, t_x]*100)/100 }%)", end=" / ")
    # print("정답 좌표: ", end="") # 답 여러개일 때
    # for t_yx, t_yx_chance in zip(t_yxs, t_yx_chances):
    #     print(f"{t_yx} ({t_yx_chance}%)", end=" / ")
    print(f"\n구한 좌표: x={str(a_x).rjust(2)},y={str(a_y).rjust(2)} ({ math.floor(winning_odds[a_y, a_x]*100)/100 }%)", end=" / ")
    print("정답!" if [a_y, a_x] == [t_y, t_x] else "응 아니야~")
    # print("정답!" if [a_y, a_x] in t_yxs else "응 아니야~")

# 멈추고 싶을 때까지 랜덤 문제 테스트 (scope_l부터 scope_r '전'까지의 범위에서 테스트)
def test_random_picks(network, x_datas, t_datas, scope_l=0, scope_r=None):
    answer = input("\n문제를 랜덤으로 풀어볼거면 enter키를 눌러 / 그만하려면(2)\n")
    if answer == "2":
        return
    if scope_r == None:
        scope_r = t_datas.shape[0]
    while True:
        test_pick(network, x_datas, t_datas, random.randrange(scope_l, scope_r))
        answer = input()
        if answer == "2":
            return

# 맞거나 틀린 문제 골라 테스트
def test_right_or_wrong_answers(network, x_datas, t_datas, wrong_idxs):
    
    answer = input("\n맞은 문제 확인(1) / 틀린 문제 확인(any) / 그만하려면(2): ")
    if answer == "1":
        TrueR_FalseW = True
    elif answer == "2":
        return
    else:
        TrueR_FalseW = False

    print()
    len_Q = len(t_datas) 
    len_W = len(wrong_idxs)
    num_test = 0

    for idx in range(len(t_datas)):
        if (idx not in wrong_idxs) == TrueR_FalseW:
            print("="*50)
            test_pick(network, x_datas, t_datas, idx)

            num_test += 1
            if TrueR_FalseW:
                print(f"({num_test}/{len_Q-len_W})")
            else:
                print(f"({num_test}/{len_W})")
    
            answer = input()
            if answer == "2":
                return

    print(f"{'맞은' if TrueR_FalseW else '틀린'} 문제들을 모두 확인했어")
