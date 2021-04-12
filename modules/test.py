import numpy as np
from modules.common.functions import softmax, cross_entropy_error

def first_place_yx(array2dim, find_all=False):
    if not find_all: # return y, x
        return array2dim.argmax()//15, array2dim.argmax()%15

    first_place_indexs = np.argwhere(array2dim == np.amax(array2dim)).tolist()
    first_place_yx_list = []
    # print(first_place_indexs)

    for fp_index in first_place_indexs:
        if fp_index not in first_place_yx_list:
            first_place_yx_list.append(fp_index)

    return first_place_yx_list # return [[y1, x1], [y2, x2], ...]

def print_board(*args, mode="QnA_AI", num_answers=1):
    
    if len(args) == 1:
        Q = args[0]
        if mode!="Q":
            mode = "Q"
            print("배열을 하나 받으면 Q모드 출력만 가능")
    elif len(args) == 2:
        Q, A = args[0], args[1]
        if A.shape[0] == 1:
            A = A.reshape(15, 15)
        if mode!="QnA_AI" and mode!="QnA":
            print("배열을 두개 받으면 QnA 또는 QnA_AI모드 출력만 가능")
            if 99 < A.sum() < 101:
                mode = "QnA_AI"
            else:
                mode = "QnA"
    else:
        print("Error: *args에는 0~1개의 배열만 받을 수 있음")
        return

    len_x, len_y = Q.shape[-1], Q.shape[-2]
    if Q.ndim == 4:
        Q = Q[0][0]
    if mode=="QnA_AI":
        a_y, a_x = A.argmax()//15, A.argmax()%15

    for y in range(len_y): ### y, len_x 
        for x in range(len_x): ### x, len_y
    
            if Q[y][x] == 0:
                if mode=="Q":
                    print("🟤", end="")
                elif mode=="QnA":
                    if A[y][x] == 0:
                        print("🟤", end="")
                    else:
                        print("🟣", end="")
                else:
                    chance = A[y][x]*num_answers
                    if 0 < chance < 5:
                        print("🔴", end="") if x!=a_x or y!=a_y else print("🟥", end="")
                    elif 5 <= chance < 20:
                        print("🟠", end="") if x!=a_x or y!=a_y else print("🟧", end="") 
                    elif 20 <= chance < 50:
                        print("🟡", end="") if x!=a_x or y!=a_y else print("🟨", end="")
                    elif 50 <= chance < 80:
                        print("🟢", end="") if x!=a_x or y!=a_y else print("🟩", end="")
                    elif 80 <= chance < 95:
                        print("🔵", end="") if x!=a_x or y!=a_y else print("🟦", end="")
                    elif 95 <= chance:
                        print("🟣", end="") if x!=a_x or y!=a_y else print("🟪", end="")
                    else:
                        print("🟤", end="") if x!=a_x or y!=a_y else print("🟫", end="")
 
            elif Q[y][x] == 1:
                print("⚫", end="") if mode=="QnA" or mode=="Q" or (x!=a_x or y!=a_y) else print("⬛", end="")
            elif Q[y][x] == -1:
                print("⚪", end="") if mode=="QnA" or mode=="Q" or (x!=a_x or y!=a_y) else print("⬜", end="")
            else:
                print("\nprint_question_board() 오류!")
        print()

    if mode == "QnA_AI":
        print("\n🟤 < 0.5% < 🔴 < 5% < 🟠 < 20% < 🟡 < 50% < 🟢 < 80% < 🔵 < 95% < 🟣\n")

def test(network, x_data, t_data, index):
    x = x_data[index:index+1]
    t = t_data[index:index+1].reshape(15, 15)

    score_board = network.predict(x)
    a_y, a_x = score_board_reshape.argmax()//15, score_board_reshape.argmax()%15

    winning_chance = (softmax(score_board) * 100).reshape(15, 15) ### score_board_reshape.round(2) np.round_(winning_chance, 2)
    # winning_chance = np.where(winning_chance==winning_chance[a_y, a_x], winning_chance, 0)
    
    t_yxs = first_place_yx(t, find_all=True)
    
    winning_chances = []
    for idx, _ in enumerate(t_yxs):
        chance = winning_chance[t_yxs[idx][0], t_yxs[idx][1]].round(1)
        winning_chances.append(chance)

    print("\n=== 점수(scores) ===")
    print(score_board_reshape.astype(np.int64).reshape(15, 15))
    print("\n=== 각 자리의 확률 분배(%) ===")
    print(winning_chance.astype(np.int64))
    print(f"\n=== Question_{index} ===")
    print_board(x, t, mode="QnA")
    print("\n=== AI's Answer ===")
    print_board(x, winning_chance.round(0), mode="QnA_AI", num_answers=len(t_yxs))

    print("정답 좌표: ", end="")
    for t_yx, t_yx_chance in zip(t_yxs, winning_chances):
        print(f"{t_yx} ({t_yx_chance}%)", end=" / ")
    print(f"\n구한 좌표: [{a_y}, {a_x}] ({winning_chance[a_y, a_x].round(1)}%)", end=" / ")
    print("정답!" if [a_y, a_x] in t_yxs else "응 아니야~")
