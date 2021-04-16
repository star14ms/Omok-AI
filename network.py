# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('modules')  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from modules.test import print_board
import math # save_params에서 acc에 floor사용, accuracy에서 ceil 사용

class DeepConvNet:
    
    def __init__(self, 
        layers_info = [
            'conv',   'Relu',
            'conv',   'Relu',
            'conv',   'Relu',
            'convSum','Relu',
            'softmaxLoss'
        ], params_ = [(1, 15, 15),
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},

        ], dropout_ration=0.5, pooling_params={"pool_h": 2, "pool_w": 2, "stride": 2}, 
           weight_decay_lambda=0, mini_batch_size=110, saved_network_pkl=None):

        layers_info = [layer.lower() for layer in layers_info]
        self.network_infos = {}

        # __init__() 매개변수 (신경망 정보 출력용)
        self.layers_info = layers_info
        self.params_ = params_
        self.dropout_ration = dropout_ration
        self.pooling_params = pooling_params
        self.weight_decay_lambda = weight_decay_lambda
        self.mini_batch_size = mini_batch_size
        self.saved_network_pkl = saved_network_pkl

        # class에서 사용하는 매개변수
        self.learning_num = 0 # 문제를 학습한 횟수
        self.layers = [] # 레이어 생성
        self.params = {} # 레이어의 가중치와 편향들 저장
        self.layer_idxs_used_params = [] # 가중치와 편향을 사용하는 레이어들의 위치
        self.loaded_params = {} ### 저장해놨다가 네트워크 생성 후 다시 self.params에 업데이트 

        # pkl 파일을 입력받았다면 불러오기
        if saved_network_pkl != None:
            if self.load_params(saved_network_pkl) == False:
                print("네트워크 불러오기 실패...")
                return ImportError
            # else:
                # self.saved_network_pkl = "CR_CR_CR_CsumR_Smloss Momentum lr=0.01 ln=28600 acc=99.93 params"
                # self.layers_info = [layer.lower() for layer in self.layers_info]
                # layers_info = self.layers_info

        self.network_name = ""
        for layer in self.layers_info:
            if layer == "conv":
                self.network_name = self.network_name + "C"
            elif layer == "convsum":
                self.network_name = self.network_name + "Csum"
            elif layer == "relu":
                self.network_name = self.network_name + "R_"
            elif layer == "softmaxloss":
                self.network_name = self.network_name + "Smloss"
        # print(self.network_name)

        # 각 층의 뉴런 사이에 연결된 노드 수 저장
        node_nums = np.array([0 for i in range(len(params_))])
        conv_params = []
        conv_Sum_Cannels = False
        feature_map_size = params_[0][1] if type(params_[0]) == tuple else params_[0]
        idx = 0
        for layer_idx, layer in enumerate(layers_info):
            
            if 'conv' in layer or 'convolution' in layer:
    
                if type(params_[idx]) == tuple: ### 'tuple'
                    if type(params_[idx+1]) == dict:
                        node_nums[idx] = params_[idx][0] * (params_[idx+1]['filter_size'] ** 2)
                    else:
                        print(f"error : node num idx {idx}에서 오류1")
                    conv_params.append(params_[idx+1])
                elif type(params_[idx]) == dict:
                    if type(params_[idx+1]) == dict:
                        node_nums[idx] = params_[idx]['filter_num'] * (params_[idx+1]['filter_size'] ** 2)
                    else:
                        print(f"error : node num idx {idx}에서 오류2")
                    conv_params.append(params_[idx+1])
                else:
                    print(f"error : node num idx {idx}에서 오류3")
                
                # 특징 맵 크기 구하기
                feature_map_size = (( feature_map_size + 2*params_[idx+1]['pad'] - params_[idx+1]['filter_size'] ) / params_[idx+1]['stride']) + 1
                
                if ( feature_map_size + 2*params_[idx+1]['pad'] - params_[idx+1]['filter_size'] ) % params_[idx+1]['stride'] != 0:
                    print("Error: 합성곱 계층 출력 크기가 정수가 아님")
                idx += 1
                
                # 채널들을 합치는 합성곱 계층인지 여부를 저장 (다음 노드 수가 달라짐: /채널 수)
                if 'sum' in layer:
                    conv_Sum_Cannels = True
                else:
                    conv_Sum_Cannels = False
                    
            elif layer == 'pool' or layer == 'pooling':
                if (feature_map_size - pooling_params["pool_h"]) % pooling_params["stride"] != 0:
                    print("Error: 풀링 계층 출력 크기가 정수가 아님")
                feature_map_size = (feature_map_size - pooling_params["pool_h"]) / pooling_params["stride"] + 1

            elif layer == 'affine':

                if type(params_[idx]) == dict:
                    if type(params_[idx+1]) == int:
                        if conv_Sum_Cannels:
                            node_nums[idx] = params_[idx]['filter_num'] * (feature_map_size**2)
                            conv_Sum_Cannels = False
                        else:
                            node_nums[idx] = feature_map_size**2

                        node_nums[idx+1] = params_[idx+1] ### affine은 앞뒤 노드 수가 모두 필요함
                    else:
                        print(f"error : node num idx {idx}에서 오류4")
                    conv_params.append(params_[idx])

                elif type(params_[idx]) == int:
                    if type(params_[idx+1]) == int:
                        node_nums[idx] = params_[idx]
                        node_nums[idx+1] = params_[idx+1]
                    else:
                        print(f"error : node num idx {idx}에서 오류5")

                elif type(params_[idx]) == tuple:
                    if type(params_[idx+1]) == int:
                        node_nums[idx] = params_[idx][1] * params_[idx][2]
                        node_nums[idx+1] = params_[idx+1]
                    else:
                        print(f"error : node num idx {idx}에서 오류6")
                else:
                    print(f"error : node num idx {idx}에서 오류7")

                feature_map_size = node_nums[idx]
                idx += 1

        if node_nums[-1] == 0: ### 가중치 초깃값 설정에서 0으로 나누기 방지
            node_nums = np.delete(node_nums, -1)
        # node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, 50, 10])
        # print(node_nums)

        # 가중치 초깃값 설정
        if 'relu' in layers_info:
            weight_init_scales = np.sqrt(2.0 / node_nums) # He 초깃값
        elif ('sigmoid' in layers_info) or ('tanh' in layers_info):
            weight_init_scales = np.sqrt(1.0 / node_nums) # Xavier 초깃값
        else:
            print("\nError: There is no activation function. (relu or sigmoid or tanh)\n")
            weight_init_scales = 0.01
    
        pre_channel_num = params_[0][0] if type(params_[0]) == tuple else params_[0] ### params_[0][1]
        
        idx = 0
        for layer_idx, layer in enumerate(layers_info):

            if 'conv' in layer or 'convolution' in layer:
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(  
                    conv_params[idx]['filter_num'], pre_channel_num, 
                    conv_params[idx]['filter_size'], conv_params[idx]['filter_size']  )
                self.params['b' + str(idx+1)] = np.zeros(conv_params[idx]['filter_num'])
                pre_channel_num = conv_params[idx]['filter_num']

                if 'sum' not in layer:
                    self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                    conv_params[idx]['stride'], conv_params[idx]['pad'], reshape_2dim=False))
                else:
                    self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                    conv_params[idx]['stride'], conv_params[idx]['pad'], reshape_2dim=True))
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'affine':
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn( node_nums[idx], node_nums[idx+1] )
                self.params['b' + str(idx+1)] = np.zeros(node_nums[idx+1])
                self.layers.append(Affine(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)]))
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'norm' or layer == 'batchnorm':
                self.params['gamma' + str(idx)] = np.ones(node_nums[idx])
                self.params['beta' + str(idx)] = np.zeros(node_nums[idx])
                self.layers.append(BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)]))
                
            elif layer == 'relu':
                self.layers.append(Relu())
            elif layer == 'sigmoid':
                self.layers.append(Sigmoid())
            elif layer == 'pool' or layer == 'pooling':
                self.layers.append(Pooling(pool_h=pooling_params["pool_h"], pool_w=pooling_params["pool_w"], stride=pooling_params["stride"]))
            elif layer == 'dropout' or layer == 'drop':
                self.layers.append(Dropout(dropout_ration))
            elif 'softmaxloss' in layers_info or 'softloss' in layers_info:
                self.last_layer = SoftmaxWithLoss()
            elif 'sigmoidloss' in layers_info or 'sigloss' in layers_info:
                self.last_layer = SigmoidWithLoss()
            elif 'squaredloss' in layers_info or 'loss' in layers_info:
                self.last_layer = SquaredLoss()
            else:
                print(f"\nError: Undefined layer.({layer})\n")
                return False
        # print(layers_info)
        # print(self.layers)

        if saved_network_pkl != None:
            self.params = self.loaded_params ### 네트워크를 self.params에 옮기지 않았음 
            for i, layer_idx in enumerate(self.layer_idxs_used_params): ### layers, layer_idxs_used_params 먼저 로드해야 함
                self.layers[layer_idx].W = self.params['W' + str(i+1)]
                self.layers[layer_idx].b = self.params['b' + str(i+1)]

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            # print(layer)
            if isinstance(layer, Dropout) or isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)

        weight_decay = 0
        for i, _ in enumerate(self.layer_idxs_used_params):
            W = self.params['W' + str(i+1)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t, save_wrong_idxs=False, multiple_answers=False, verbose=False):
        batch_size = self.mini_batch_size
        # if len(t) % batch_size != 0:
        #     print(f"\n문제 수가 미니배치 수로 나누어 떨어지지 않음\n(뒤에 남는 {len(t) % batch_size}문제는 풀지 못함)")
        if verbose:
            print(f"\n=== 정답률 측정 (총 {x.shape[0]}문제) ===\n")
        if t.ndim != 1 : 
            t = np.argmax(t, axis=1)
        # if not multiple_answers:
            # if t.ndim != 1 : t = np.argmax(t, axis=1)
        # else:
            # t_yxs = []
            # max_t_num = 0
            # t_copy[...] = t

            # for t_one in t:
            #     t_yx = np.argwhere(t_one == np.amax(t_one)).flatten()
            #     t_yxs.append(t_yx)
            #     max_t_num = max(max_t_num, len(t_yx))

            # for i in range(max_t_num):
            #     t_yxs1 = t_yxs[...][0]

        acc = 0.0
        wrong_idxs = [] # 풀어보지 못한 나머지 문제들을 모두 맞았다고 처리
        # right_idxs = [] # 풀어보지 못한 나머지 문제들을 모두 틀렸다고 처리

        for i in range( math.ceil(x.shape[0] / batch_size) ):
            tx = x[i*batch_size:(i+1)*batch_size if (i+1)*batch_size <= x.shape[0] else x.shape[0]]
            tt = t[i*batch_size:(i+1)*batch_size if (i+1)*batch_size <= x.shape[0] else x.shape[0]]
            if verbose:
                print("문제", i*batch_size, "~", (i+1)*batch_size if (i+1)*batch_size <= x.shape[0] else x.shape[0], "푸는 중")
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

            # if not multiple_answers:
                # tt = t[i*batch_size:(i+1)*batch_size]
                # acc += np.sum(y == tt)
            # else:
                # tt_yxs = t_yxs[i*batch_size:(i+1)*batch_size]
                # # print(y[0], tt_yxs[0])
                # acc += np.sum(y in tt_yxs)

            if save_wrong_idxs:
                batch_idx = np.where(y != tt)
                wrong_idxs = wrong_idxs + (i*batch_size+batch_idx[0]).tolist()
                # ridxs = (i*batch_size+(np.where(y == tt))[0]).tolist()
                # right_idxs = right_idxs + ridxs
                # print(ridxs)
        
        if save_wrong_idxs:
            # print(len(right_idx, len(wrong_idxs)))
            # print(right_idxs)
            if verbose:
                Fnum, Qnum = len(wrong_idxs), x.shape[0]
                print(f"\n=== 총 {Qnum}문제, 정답 {Qnum-Fnum}개, 오답 {Fnum}개 (정답률: {math.floor(acc/x.shape[0]*100*100)/100}%) ===")
            return acc / x.shape[0], wrong_idxs
        else:
            return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            # print(layer)
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW + self.weight_decay_lambda * self.params['W' + str(i+1)]
            grads['b' + str(i+1)] = self.layers[layer_idx].db
            if layer_idx+1 < len(self.layers) and type(self.layers[layer_idx+1]) == BatchNormalization:
                # print(i+1, layer_idx+1)
                grads['gamma' + str(i+1)] = self.layers[layer_idx+1].dgamma
                grads['beta' + str(i+1)] = self.layers[layer_idx+1].dbeta

        return grads

    def save_params(self, trainer, save_inside_dir=True, pkl_dir="saved_pkls/network"):
        
        self.network_infos["layers_info"]         = self.layers_info
        self.network_infos["params_"]             = self.params_
        self.network_infos["dropout_ration"]      = self.dropout_ration
        self.network_infos["pooling_params"]      = self.pooling_params
        self.network_infos["weight_decay_lambda"] = self.weight_decay_lambda
        self.network_infos["mini_batch_size"]     = self.mini_batch_size
        self.network_infos["saved_network_pkl"]   = self.saved_network_pkl
        self.network_infos["learning_num"]        = self.learning_num

        self.network_infos["params"] = self.params
        # self.network_infos["layers"] = self.layers # pkl파일 용량이 너무 커짐 (이미 layers_info가 있으니 똑같이 다신 만들 수 있음)
        # self.network_infos["layer_idxs_used_params"] = self.layer_idxs_used_params  ### 새로운 레이어들이 또 append됨
        # print(self.saved_network_pkl)
        if len(trainer.test_accs) == 0:
            if self.saved_network_pkl == None:
                acc = "None"
            else: ### self. 빼먹음
                acc = self.saved_network_pkl.split(" ")[-2].lstrip("acc=") # " " -> "_" 같이 수정해주어야 함
        else:
            acc = math.floor(trainer.test_accs[-1]*100*100)/100
        
        file_name = f"{self.network_name} {trainer.optimizer.__class__.__name__} lr={trainer.optimizer.lr} ln={self.learning_num} acc={acc} params.pkl"
        file_path = file_name
        
        if save_inside_dir: ### if save_inside_dir: 안에 선언이 있으면 선언되지 않을 수 있음
            file_path = f"{pkl_dir}/{self.network_name}/" + file_path

        if not os.path.exists(f"{pkl_dir}/{self.network_name}"):
            os.makedirs(f"{pkl_dir}/{self.network_name}")
            print(f"{pkl_dir}/{self.network_name} 폴더 생성")

        # main process
        # params = {}
        # for key, val in self.params.items():
        #     params[key] = val
        # self.network_infos["params"] = params

        with open(file_path, 'wb') as f:
            pickle.dump(self.network_infos, f)

        print(f"\n네트워크 저장 성공!\n({file_name})")

    def load_params(self, file_name="params.pkl", pkl_dir="saved_pkls/network"):

        if (".pkl" not in file_name) or (file_name[-4:] != ".pkl"):
            file_name = f"{file_name}.pkl"
        file_path = file_name

        netwrok_name = file_path.split(" ")[0] # " " -> "_" 같이 수정해주어야 함
        file_path = f"{netwrok_name}/{file_path}"

        if (file_path.split("/")[:-1] != f"{pkl_dir}"):
            file_path = f"{pkl_dir}/{file_path}"

        if not os.path.exists(f"{pkl_dir}/{netwrok_name}"):
            os.makedirs(f"{pkl_dir}/{netwrok_name}")
            print(f"Error: {pkl_dir}/{netwrok_name} 폴더가 없어서 생성")

        if not os.path.exists(file_path):
            file_path = file_name
            if not os.path.exists(file_path):
                file_path = f"{pkl_dir}/{file_name}"
                if not os.path.exists(file_path):
                    print(f"Error: {file_name} 파일이 없음")
                    return False

        # main process
        with open(file_path, 'rb') as f:
            network_infos = pickle.load(f)

        self.layers_info         = network_infos["layers_info"]
        self.params_             = network_infos["params_"] 
        self.dropout_ration      = network_infos["dropout_ration"] 
        self.pooling_params      = network_infos["pooling_params"] 
        self.weight_decay_lambda = network_infos["weight_decay_lambda"]    
        self.saved_network_pkl   = network_infos["saved_network_pkl"] 
        self.learning_num        = network_infos["learning_num"]

        self.loaded_params = network_infos["params"]
        # self.layers = network_infos["layers"] # pkl파일 용량이 너무 커짐
        # self.layer_idxs_used_params = network_infos["layer_idxs_used_params"] ### 새로운 레이어들이 또 append됨
        
        # print(network_infos["layers_info"], network_infos["learning_num"], network_infos["saved_network_pkl"] )

        # params = network_infos["params"]
        # for key, val in params.items():
        #     self.params[key] = val
        
        if self.mini_batch_size != network_infos["mini_batch_size"]:
            print("\nError: '불러올 신경망의 미니배치 수'와 '처음 선언한 미니배치 수'가 일치하지 않는다", end=" ")
            print("(앞으로 풀 문제의 수를 나누어 떨어지게 하는 것이 좋음)")
            answer = input("\n어느 것을 선택하시겠습니까? 선언한 수(any)/불러온 수(1): ")
            if answer == "1":
                self.mini_batch_size = network_infos["mini_batch_size"] 
            print("미니배치 수", self.mini_batch_size, "선택")

        print(f"\n네트워크 불러오기 성공!\n({file_name})")
        
        # 파일 이름에서 문제를 학습한 횟수 가져오기 (self.learning_num)
        # name_splited = file_path.split("_")
        # load_ln_success = False
        # for name in name_splited:
            # if "ln=" in name:
            #     ln = name.lstrip("ln=")
            #     load_ln_success = True
        # if not load_ln_success:
        #     print("Error: 누적 학습 횟수 가져오기 실패 (.pkl 파일 이름에 누적 학습 횟수(ln_)가 없음)")
        # else:
            # try:
            #     self.learning_num = int(ln)
            # except:
            #     print("Error: 누적 학습 횟수 가져오기 실패 (.pkl 파일 이름에서 누적 학습 정보(ln_)가 정수형이 아님")

        return True

    def think_win_xy(self, board):
        scores = self.predict(board.reshape(1, 1, 15, 15))
        x, y = scores.argmax()%15, scores.argmax()//15
        winning_odds = (softmax(scores) * 100).reshape(15, 15)

        print("\n=== 점수(scores) ===")
        print(scores.astype(np.int64).reshape(15, 15))
        print("\n=== 각 자리의 확률 분배(%) ===")
        print(winning_odds.astype(np.int64).reshape(15, 15))
        print(f"\n=== Question ===")
        print_board(board, mode="Q")
        print("\n=== AI's Answer ===")
        print_board(board, winning_odds, mode="QnA_AI")
        print(f"구한 좌표: [{x}, {y}] ({winning_odds[y, x].round(1)}%)")

        return x, y