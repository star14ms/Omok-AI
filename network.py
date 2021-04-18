# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('modules')
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from modules.test import print_board
import math # save_params에서 acc에 floor사용, accuracy에서 ceil 사용
from modules.common.util import bcolors, __get_logger

logger = __get_logger()
# logger.critical("hello")

class DeepConvNet:
    
    def __init__(self, 
        layers_info = [
            'Conv', 'Relu',
            'Conv', 'Relu',
            'Conv', 'Relu',
            'ConvSum', 'Relu',
            'SoftmaxLoss'
        ], params_ = [(1, 15, 15),
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
        ], dropout_ration=0.5, pooling_params={"pool_h": 2, "pool_w": 2, "stride": 2}, 
           weight_decay_lambda=0, mini_batch_size=100, saved_network_pkl=None, not0_size=4):
        
        layers_info = [layer.lower() for layer in layers_info]
        self.network_infos = {} # pkl파일에 저장할 신경망 정보
        # print(layers_info)
        # 신경망 정보에 넣을 내용
        self.layers_info = layers_info
        self.params_ = params_
        self.dropout_ration = dropout_ration
        self.pooling_params = pooling_params
        self.weight_decay_lambda = weight_decay_lambda
        self.mini_batch_size = mini_batch_size ### 다른 미니배치 수로 학습한 신경망을 불러오면 바뀌어버림
        self.saved_network_pkl = saved_network_pkl
        self.learning_num = 0 # 문제를 학습한 횟수
        self.params = {} # 레이어의 가중치와 편향들 저장
        
        self.not0_size = not0_size # 오답 샘플 개수

        self.loaded_params = {} ### 저장해놨다가 네트워크 생성 후 다시 self.params에 업데이트 
        self.layer_idxs_used_params = [] # 가중치와 편향을 사용하는 레이어들의 위치
        self.layers = [] # 레이어 저장
        
        # pkl 파일을 입력받았다면 불러오기
        if saved_network_pkl != None:
            if self.load_params(saved_network_pkl) == False:
                print("네트워크 불러오기 실패...")
                return ImportError
            else:
                self.layers_info = [layer.lower() for layer in self.layers_info]
                layers_info = self.layers_info # 불러온 레이어 데이터로 레이어를 만들어야함
                params_ = self.params_
                self.saved_network_pkl = saved_network_pkl

        # 네트워크 정보 문자로 나타내기 (네트워크를 저장할 때 파일 이름으로 쓰임)
        layers_info_str = ""
        for layer in self.layers_info:
            if layer == "conv":
                layers_info_str = layers_info_str + "_C"
            elif layer == "convsum":
                layers_info_str = layers_info_str + "_Csum"
            elif layer == "affine":
                layers_info_str = layers_info_str + "_A"
            elif layer == "relu":
                layers_info_str = layers_info_str + "R"
            elif "norm" in layer:
                layers_info_str = layers_info_str + "N"
            elif "dropout" in layer:
                layers_info_str = layers_info_str + "D"
            elif layer == "softmaxloss" or layer == "smloss":
                layers_info_str = layers_info_str + "_Smloss"
            elif layer == "not0samplingloss" or "not0samploss":
                layers_info_str = layers_info_str + "_N0sloss"
        self.network_dir = layers_info_str.lstrip("_")

        # 매개변수를 사용하는 층의 뉴런 하나당 앞 층과 연결된 노드 수, 출력 크기 계산
        node_nums = np.array([0 for i in range(len(params_)-1)])
        out_sizes = np.array([0 for i in range(len(params_)-1)])
        pre_sum_cannels = False
        idx = 0 # 매개변수를 사용하는 층의 위치
        feature_map_size = params_[0][1] if type(params_[0]) == tuple else params_[0]
        for layer_idx, layer in enumerate(layers_info):
            
            if 'conv' in layer or 'convolution' in layer:
                if type(params_[idx+1]) == dict:

                    if type(params_[idx]) == tuple: ### 'tuple'
                        node_nums[idx] = params_[idx+1]['filter_size'] ** 2

                    elif type(params_[idx]) == dict:
                        node_nums[idx] = params_[idx+1]['filter_size'] ** 2 * params_[idx]['filter_num']

                    else:
                        logger.warning(f"node num idx {idx}에서 오류\nnode_nums={node_nums}" + \
                        f"\nparams_의 {type(params_[idx])} 타입은 이전 노드 수를 어떻게 구해야 할 지 모르겠다")
                else:
                    logger.warning(f"node num idx {idx}에서 오류\nnode_nums={node_nums}" + \
                    f"\nError: params_의 conv위치에는 합성곱 계층 정보(dic)가 들어가야 한다.")
                
                # 채널들을 합치는 합성곱 계층이었나 여부를 저장 (다음 노드 수가 달라짐: /채널 수)
                pre_sum_cannels = True if 'sum' in layer else False
                
                # 특징 맵 크기 구하기
                if ( feature_map_size + 2*params_[idx+1]['pad'] - params_[idx+1]['filter_size'] ) % params_[idx+1]['stride'] != 0:
                    logger.warning(
                        f"합성곱 계층 출력 크기가 정수가 아님" + \
                    "({feature_map_size} + 2*{params_[idx+1]['pad']} - {params_[idx+1]['filter_size']}) / {params_[idx+1]['stride']}")

                feature_map_size = (( feature_map_size + 2*params_[idx+1]['pad'] - params_[idx+1]['filter_size'] ) / params_[idx+1]['stride']) + 1
                out_sizes[idx] = feature_map_size**2 * (params_[idx+1]['filter_num'] if 'sum' not in layer else 1)
                idx += 1
                    
            elif layer == 'pool' or layer == 'poolin0g':
                if (feature_map_size - self.pooling_params["pool_h"]) % self.pooling_params["stride"] != 0:
                    logger.warning("풀링 계층 출력 크기가 정수가 아님")
                feature_map_size = (feature_map_size - self.pooling_params["pool_h"]) / self.pooling_params["stride"] + 1

            elif layer == 'affine':
                if type(params_[idx+1]) == int:

                    if type(params_[idx]) == dict:
                        node_nums[idx] = (feature_map_size**2) * (params_[idx]['filter_num'] if not pre_sum_cannels else 1)
                        pre_sum_cannels = False

                    elif type(params_[idx]) == int:
                        node_nums[idx] = params_[idx]

                    elif type(params_[idx]) == tuple:
                        node_nums[idx] = params_[idx][1] * params_[idx][2]
                    else:
                        logger.warning(f"node num idx {idx}에서 오류\nnode_nums={node_nums}" + \
                        f"Error: params_의 {type(params_[idx])} 타입은 이전 노드 수를 어떻게 구해야 할 지 모르겠다")

                else:
                    logger.warning(f"node num idx {idx}에서 오류\nnode_nums={node_nums}" + \
                    f"Error: params_의 affine에는 이전 노드 개수(int)가 들어가야 한다.")

                feature_map_size = node_nums[idx]
                out_sizes[idx] = params_[idx+1] ### affine은 앞뒤 노드 수가 모두 필요함
                idx += 1

            # print(feature_map_size)
        
        # node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, 50, 10])
        # print("매개변수를 사용하는 레이어들의")
        # print(node_nums, "이전 노드 개수")
        # print(out_sizes, "출력 크기")
        
        # 가중치 초깃값 설정
        if 'relu' in layers_info:
            weight_init_scales = np.sqrt(2.0 / node_nums) # He 초깃값
        elif 'sigmoid' in layers_info or 'tanh' in layers_info:
            weight_init_scales = np.sqrt(1.0 / node_nums) # Xavier 초깃값
        else:
            weight_init_scales = 0.01
            print("\nError: There is no activation function. (relu or sigmoid or tanh)\n")
        
        # 신경망 생성
        idx = 0 # 가중치를 사용하는 레이어의 위치
        pre_channel_num = params_[0][0] if type(params_[0]) == tuple else params_[0] ### params_[0][1]
        for layer_idx, layer in enumerate(layers_info):

            if 'conv' in layer or 'convolution' in layer:
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(  
                    params_[idx+1]['filter_num'], pre_channel_num, 
                    params_[idx+1]['filter_size'], params_[idx+1]['filter_size'])
                self.params['b' + str(idx+1)] = np.zeros(params_[idx+1]['filter_num'])
                pre_channel_num = params_[idx+1]['filter_num']

                self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                params_[idx+1]['stride'], params_[idx+1]['pad'], reshape_2dim= (False if 'sum' not in layer else True) ))
                
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'affine':
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn( node_nums[idx], out_sizes[idx] )
                self.params['b' + str(idx+1)] = np.zeros(out_sizes[idx])
                self.layers.append(Affine(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)]))
                
                self.layer_idxs_used_params.append(layer_idx)
                idx += 1

            elif layer == 'norm' or layer == 'batchnorm':
                self.params['gamma' + str(idx)] = np.ones(out_sizes[idx-1])
                self.params['beta' + str(idx)] = np.zeros(out_sizes[idx-1])
                self.layers.append(BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)]))
                
            elif layer == 'relu':
                self.layers.append(Relu())
            elif layer == 'sigmoid':
                self.layers.append(Sigmoid())
            elif layer == 'pool' or layer == 'pooling':
                self.layers.append(Pooling(
                pool_h=self.pooling_params["pool_h"], 
                pool_w=self.pooling_params["pool_w"], 
                stride=self.pooling_params["stride"]))
            elif layer == 'dropout' or layer == 'drop':
                self.layers.append(Dropout(self.dropout_ration))
            elif 'softmaxloss' in layers_info or 'softloss' in layers_info:
                self.last_layer = SoftmaxWithLoss()
            elif 'sigmoidloss' in layers_info or 'sigloss' in layers_info:
                self.last_layer = SigmoidWithLoss()
            elif 'squaredloss' in layers_info:
                self.last_layer = SquaredLoss()
            elif 'not0samplingloss' in layers_info or 'not0samploss' in layers_info:
                self.last_layer = Not0SamplingLoss(not0_num=self.not0_size) ## 5로도 시험, num0를 푸는 문제마다 달라지게 할 수 있을까?
            else:
                print(f"\nError: Undefined layer ({layer})\n")
                return False
        
        # 불러온 신경망의 매개변수들 덮어쓰기
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
        
        # print(self.last_layer, type(self.last_layer), self.last_layer.__class__.__name__)
        if type(self.last_layer) == Not0SamplingLoss:
            loss = self.last_layer.forward(y, t, x) ### Not0SamplingLoss는 문제 데이터도 필요함
        else:
            loss = self.last_layer.forward(y, t)
            
        return loss + weight_decay

    def accuracy(self, x, t, save_wrong_idxs=False, multiple_answers=False, verbose=False, percentage=True):
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
            return acc / x.shape[0] * (100 if percentage else 1), wrong_idxs
        else:
            return acc / x.shape[0] * (100 if percentage else 1)

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

    def save_params(self, trainer, str_data_info, save_inside_dir=True, pkl_dir="saved_pkls"):
        
        self.network_infos["layers_info"]         = self.layers_info
        self.network_infos["params_"]             = self.params_
        self.network_infos["dropout_ration"]      = self.dropout_ration
        self.network_infos["pooling_params"]      = self.pooling_params
        self.network_infos["weight_decay_lambda"] = self.weight_decay_lambda
        # self.network_infos["mini_batch_size"]     = self.mini_batch_size
        self.network_infos["saved_network_pkl"]   = self.saved_network_pkl
        self.network_infos["learning_num"]        = self.learning_num
        self.network_infos["params"]              = self.params
        # self.network_infos["layers"] = self.layers # pkl파일 용량이 너무 커짐 (이미 layers_info가 있으니 똑같이 다신 만들 수 있음)
        # self.network_infos["layer_idxs_used_params"] = self.layer_idxs_used_params  ### 새로운 레이어들이 또 append됨
        # print(self.saved_network_pkl)
        if len(trainer.test_accs) == 0:
            if self.saved_network_pkl == None:
                acc = "None"
            else: ### self. 빼먹음
                acc = self.saved_network_pkl.split(" ")[-3].lstrip("acc_") # " " -> "_" 같이 수정해주어야 함
        else:
            acc = math.floor(trainer.test_accs[-1]*100)/100
        
        file_name = f"{str_data_info} acc_{acc} ln_{self.learning_num} " + \
            f"{trainer.optimizer.__class__.__name__} lr_{trainer.optimizer.lr} {self.network_dir} params.pkl"
        file_path = file_name
        
        if save_inside_dir: ### if save_inside_dir: 안에 선언이 있으면 선언되지 않을 수 있음
            file_path = f"{pkl_dir}/{self.network_dir}/" + file_path
        
        if not os.path.exists(f"{pkl_dir}/{self.network_dir}"):
            os.makedirs(f"{pkl_dir}/{self.network_dir}")
            print(f"{pkl_dir}/{self.network_dir} 폴더 생성")
        
        # main process
        # params = {}
        # for key, val in self.params.items():
        #     params[key] = val
        # self.network_infos["params"] = params

        with open(file_path, 'wb') as f:
            pickle.dump(self.network_infos, f)

        print(f"\n네트워크 저장 성공!\n({file_name})")

    def load_params(self, file_name="params.pkl", pkl_dir="saved_pkls"):

        if (".pkl" not in file_name) or (file_name[-4:] != ".pkl"):
            file_name = f"{file_name}.pkl"
        file_path = file_name

        # network_dir = file_path.split(" [")[0].split(" ")[-1] + " [" + file_path.split(" [")[1].rstrip(" params.pkl") 
        network_dir = file_path.split(" ")[-2] # " " -> "_" 같이 수정해주어야 함
        file_path = f"{network_dir}/{file_path}"

        if (file_path.split("/")[:-1] != f"{pkl_dir}"):
            file_path = f"{pkl_dir}/{file_path}"

        # if not os.path.exists(f"{pkl_dir}/{network_dir}"):
        #     os.makedirs(f"{pkl_dir}/{network_dir}")
        #     print(f"Error: {pkl_dir}/{network_dir} 폴더가 없어서 생성")

        if not os.path.exists(file_path):
            # print(f"Error: {file_path} 파일이 없음")
            file_path = file_name
            if not os.path.exists(file_path):
                # print(f"Error: {file_path} 파일이 없음")
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

        params = network_infos["params"]
        for key, val in params.items():
            self.params[key] = val
        # self.loaded_params = network_infos["params"] ### 복사가 아닌 저장된 위치를 가리킴
        # self.layers = network_infos["layers"] # pkl파일 용량이 너무 커짐
        # self.layer_idxs_used_params = network_infos["layer_idxs_used_params"] ### 새로운 레이어들이 또 append됨
        
        # print(network_infos["layers_info"], network_infos["learning_num"], network_infos["saved_network_pkl"] )

        
        # if self.mini_batch_size != network_infos["mini_batch_size"]:
        #     print("\nError: '불러올 신경망의 미니배치 수'와 '처음 선언한 미니배치 수'가 일치하지 않는다", end=" ")
        #     print("(앞으로 풀 문제의 수를 나누어 떨어지게 하는 것이 좋음)")
        #     answer = input("\n어느 것을 선택하시겠습니까? 선언한 수(any)/불러온 수(1): ")
        #     if answer == "1":
        #         self.mini_batch_size = network_infos["mini_batch_size"] 
        #     print("선택한 미니배치 수: ", self.mini_batch_size)

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
        
        scores_nomalized = ( (scores - np.min(scores)) / scores.ptp() * 100).reshape(15, 15)
        winning_odds = (softmax(scores) * 100).reshape(15, 15)
        
        print("\n=== 점수(scores) ===")
        np.set_printoptions(linewidth=np.inf, formatter={'all':lambda x: ( # 세자리 출력(100) 방지
        bcolors.according_to_score(x) + str(int(np.minimum(x, 99))).rjust(2) + bcolors.ENDC)})
        print(scores_nomalized)
        print("\n=== 각 자리의 확률 분배(%) ===")
        np.set_printoptions(linewidth=np.inf, formatter={'all':lambda x: (
        bcolors.according_to_chance(x) + str(int(np.minimum(x, 99))).rjust(2) + bcolors.ENDC)})
        print(winning_odds)
        print(f"\n=== Question ===")
        print_board(board, mode="Q")
        print("\n=== AI's Answer ===")
        print_board(board, winning_odds, mode="QnAI")
        print(f"구한 좌표: x={str(x).rjust(2)}, y={str(y).rjust(2)} ({ math.floor(winning_odds[y, x]*100)/100 }%)")

        return x, y
