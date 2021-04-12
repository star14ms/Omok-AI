# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('modules')  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from modules.test import print_board
import math # save_params에서 acc에 floor사용

class DeepConvNet:
    
    def __init__(self, layers_info = [
            'conv', 'Relu',
            'conv', 'Relu',
            'conv', 'Relu',
            'conv2d', 'Relu',
            'softmax'
        ], params = [(1, 15, 15),
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
            {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},

        ], dropout_ration=0.5, pooling_params={"pool_h": 2, "pool_w": 2, "stride": 2}, 
           params_pkl_file=None, weight_decay_lambda=0, mini_batch_size=100):

        self.weight_decay_lambda = weight_decay_lambda
        self.mini_batch_size = mini_batch_size
        self.params_file_name = None
        self.learning_num = 0

        # 각 층의 뉴런 하나당 앞층의 뉴런과 연결된 노드 수 저장
        node_nums = np.array([0 for i in range(len(params))])
        conv_params = []
        self.conv_Sum_Cannels = False
        feature_map_size = params[0][1] if type(params[0]) == tuple else params[0]
        idx = 0
        for layer_idx, layer in enumerate(layers_info):
            
            if 'conv' in layer or 'convolution' in layer:
    
                if type(params[idx]) == tuple: ### 'tuple'
                    if type(params[idx+1]) == dict:
                        node_nums[idx] = params[idx][0] * (params[idx+1]['filter_size'] ** 2)
                    else:
                        print(f"error : node num idx {idx}에서 오류1")
                    conv_params.append(params[idx+1])
                elif type(params[idx]) == dict:
                    if type(params[idx+1]) == dict:
                        node_nums[idx] = params[idx]['filter_num'] * (params[idx+1]['filter_size'] ** 2)
                    else:
                        print(f"error : node num idx {idx}에서 오류2")
                    conv_params.append(params[idx+1])
                else:
                    print(f"error : node num idx {idx}에서 오류3")
                
                # 특징 맵 크기 구하기
                feature_map_size = (( feature_map_size + 2*params[idx+1]['pad'] - params[idx+1]['filter_size'] ) / params[idx+1]['stride']) + 1
                
                if ( feature_map_size + 2*params[idx+1]['pad'] - params[idx+1]['filter_size'] ) % params[idx+1]['stride'] != 0:
                    print("Error: 합성곱 계층 출력 크기가 정수가 아님")
                idx += 1
                
                # 채널들을 합치는 합성곱 계층인지 여부를 저장 (다음 노드 수가 달라짐: /채널 수)
                if 'end' in layer:
                    self.conv_Sum_Cannels = True
                else:
                    self.conv_Sum_Cannels = False
                    
            elif layer == 'pool' or layer == 'pooling':
                if (feature_map_size - pooling_params["pool_h"]) % pooling_params["stride"] != 0:
                    print("Error: 풀링 계층 출력 크기가 정수가 아님")
                feature_map_size = (feature_map_size - pooling_params["pool_h"]) / pooling_params["stride"] + 1

            elif layer == 'affine':

                if type(params[idx]) == dict:
                    if type(params[idx+1]) == int:
                        if self.conv_Sum_Cannels:
                            node_nums[idx] = params[idx]['filter_num'] * (feature_map_size**2)
                            self.conv_Sum_Cannels = False
                        else:
                            node_nums[idx] = feature_map_size**2

                        node_nums[idx+1] = params[idx+1] ### affine은 앞뒤 노드 수가 모두 필요함
                    else:
                        print(f"error : node num idx {idx}에서 오류4")
                    conv_params.append(params[idx])

                elif type(params[idx]) == int:
                    if type(params[idx+1]) == int:
                        node_nums[idx] = params[idx]
                        node_nums[idx+1] = params[idx+1]
                    else:
                        print(f"error : node num idx {idx}에서 오류5")

                elif type(params[idx]) == tuple:
                    if type(params[idx+1]) == int:
                        node_nums[idx] = params[idx][1] * params[idx][2]
                        node_nums[idx+1] = params[idx+1]
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
        layers_info = [layer.lower() for layer in layers_info]
        if 'relu' in layers_info:
            weight_init_scales = np.sqrt(2.0 / node_nums) # He 초깃값
        elif ('sigmoid' in layers_info) or ('tanh' in layers_info):
            weight_init_scales = np.sqrt(1.0 / node_nums) # Xavier 초깃값
        else:
            print("\nError: There is no activation function. (relu or sigmoid or tanh)\n")
            RuntimeWarning()
    
        pre_channel_num = params[0][0] if type(params[0]) == tuple else params[0] ### params[0][1]
        self.params = {} # 매개변수 저장
        self.layers = [] # 레이어 생성
        self.layer_idxs_used_params = [] # 매개변수를 사용하는 레이어의 위치
        idx = 0 
        for layer_idx, layer in enumerate(layers_info):

            if 'conv' in layer or 'convolution' in layer:
                self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(  conv_params[idx]['filter_num'],
                                                pre_channel_num, conv_params[idx]['filter_size'], conv_params[idx]['filter_size']  )
                self.params['b' + str(idx+1)] = np.zeros(conv_params[idx]['filter_num'])
                pre_channel_num = conv_params[idx]['filter_num']

                if '2d' not in layer:
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
            elif 'softmax' in layers_info or 'soft' in layers_info:
                self.last_layer = SoftmaxWithLoss()
            elif 'sigmoidloss' in layers_info or 'sigloss' in layers_info:
                self.last_layer = SigmoidWithLoss()
            elif 'squaredloss' in layers_info or 'loss' in layers_info:
                self.last_layer = SquaredLoss()
            else:
                print(f"\nError: Undefined function.({layer})\n")
                return False

        # print(self.layers)

        if params_pkl_file != None:
            self.load_params(params_pkl_file)

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

    def accuracy(self, x, t, save_wrong_idxs=False, multiple_answers=False):
        batch_size = self.mini_batch_size
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
        wrong_idxs = []

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
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

        if save_wrong_idxs:
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

    def save_params(self, trainer, save_inside_dir=True, pkl_dir="saved_network_pkls"):
        
        if len(trainer.test_accs) == 0:
            acc = self.params_file_name.split("_")[-2].lstrip("acc_") # " " -> "_" 같이 수정해주어야 함
        else:
            acc = math.floor(train_acc*10000)/100
        file_name = f"{trainer.optimizer.__class__.__name__}_lr={trainer.optimizer.lr}_ln={self.learning_num}_acc={acc}_params.pkl"
        
        optimizer = file_name.split("_")[0] ### if save_inside_dir: 안에 있으면 선언되지 않을 수 있음
        if save_inside_dir:
            file_name = f"{pkl_dir}/{optimizer}/" + file_name

        if not os.path.exists(f"{pkl_dir}/{optimizer}"):
            os.makedirs(f"{pkl_dir}/{optimizer}")
            print(f"Error: .{pkl_dir}/{optimizer} 폴더가 없어서 생성")

        # main process
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

        print(f"\n네트워크 저장 성공! ({file_name})")

    def load_params(self, file_name="params.pkl", pkl_dir="saved_network_pkls"):

        if (".pkl" not in file_name) or (file_name[-4:] != ".pkl"):
            file_name = f"{file_name}.pkl"
        file_path = file_name

        optimizer = file_path.split("_")[0] # " " -> "_" 같이 수정해주어야 함
        file_path = f"{optimizer}/{file_path}"

        if (file_path.split("/")[0] != f"{pkl_dir}"):
            file_path = f"{pkl_dir}/{file_path}"

        if not os.path.exists(f"{pkl_dir}/{optimizer}"):
            os.makedirs(f"{pkl_dir}/{optimizer}")
            print(f"Error: {file_path} 파일이 없음, {pkl_dir}/{optimizer} 폴더가 없어서 생성")

        if not os.path.exists(file_path):
            file_path = file_name
            if not os.path.exists(file_path):
                file_path = f"{pkl_dir}/{file_name}"
                if not os.path.exists(file_path):
                    print(f"Error: {file_path} 파일이 없음")
                    return

        # main process
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]

        print(f"\n학습된 네트워크 불러오기 성공! ({file_name})")
        self.params_file_name = file_name

        name_splited = file_path.split("_")
        load_ln_success = False
        for name in name_splited:
            if "ln=" in name:
                ln = name.lstrip("ln=")
                load_ln_success = True
        if not load_ln_success:
            print("Error: 누적 학습 횟수 가져오기 실패 (.pkl 파일 이름에 누적 학습 횟수(ln_)가 없음)")
        else:
            try:   
                self.learning_num += int(ln)
            except:
                print("Error: 누적 학습 횟수 가져오기 실패 (.pkl 파일 이름에서 누적 학습 정보(ln_)가 정수형이 아님")

    def think_win_xy(self, board):
        score_board = self.predict(board.reshape(1, 1, 15, 15))
        x, y = score_board.argmax()%15, score_board.argmax()//15
        
        winning_chance = (softmax(score_board)*100).reshape(15, 15)

        print("\n=== 점수(scores) ===")
        print(score_board.astype(np.int64).reshape(15, 15))
        print("\n=== 각 자리의 확률 분배(%) ===")
        print(winning_chance.astype(np.int64))
        print(f"\n=== Question ===")
        print_board(board, mode="Q")
        print("\n=== AI's Answer ===")
        print_board(board, winning_chance.round(0), mode="QnA_AI")
        print(f"구한 좌표: [{x}, {y}] ({winning_chance[y, x].round(1)}%)")

        return x, y