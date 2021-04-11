# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
sys.path.append('learning')  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    
    def __init__(self, layers_info = [
        'conv', 'relu', 'conv', 'relu', 'pool', # 'conv', 'relu',
        'conv', 'relu', 'conv', 'relu', 'pool',
        'conv', 'relu', 'conv', 'relu', 'pool',
        'affine', 'norm', 'relu', 'dropout', 'affine', 'dropout', 'softmax'
        ], params = [(1, 28, 28),
        {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},

        {'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},

        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        {'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
        50, 10], dropout_ration=0.5, pooling_params={"pool_h": 2, "pool_w": 2, "stride": 2}, 
        params_pkl_name=None, mini_batch_size=100):
        # (100, 1, 28, 28)  (16, 1, 3, 3)  (100, 16, 28, 28)
        # (100, 16, 28, 28) (16, 16, 3, 3) (100, 16, 28, 28) /2
        # (100, 16, 14, 14) (32, 16, 3, 3) (100, 32, 14, 14)
        # (100, 32, 14, 14) (32, 32, 3, 3) (100, 32, 16, 16) +2, /2 
        # (100, 32, 8, 8)   (64, 32, 3, 3) (100, 64, 8, 8)
        # (100, 64, 8, 8)   (64, 64, 3, 3) (100, 64, 8, 8) /2
        # (100, 64, 4, 4)   (1024, 50)     (100, 50)
        # (100, 50)         (50, 10)       (100, 10)

        # 重みの初期化===========

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

                if 'end' not in layer:
                    self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                    conv_params[idx]['stride'], conv_params[idx]['pad']))
                else:
                    self.layers.append(Convolution(self.params['W' + str(idx+1)], self.params['b' + str(idx+1)], 
                    conv_params[idx]['stride'], conv_params[idx]['pad'], convEnd=True))
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

        self.mini_batch_size = mini_batch_size
        self.learning_num = 0
        if params_pkl_name != None:
            name_splited = params_pkl_name.split(" ")
            for name in name_splited:
                if "ln_" in name:
                    ln = name.lstrip("ln_")
            self.load_params(params_pkl_name)
            self.learning_num += int(ln)

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
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, save_wrong_idxs=False, multiple_answers=False):
        batch_size = self.mini_batch_size
        if not multiple_answers:
            if t.ndim != 1 : t = np.argmax(t, axis=1)
        # else:
        #     t_yxs = []
        #     max_t_num = 0
        #     t_copy[...] = t

        #     for t_one in t:
        #         t_yx = np.argwhere(t_one == np.amax(t_one)).flatten()
        #         t_yxs.append(t_yx)
        #         max_t_num = max(max_t_num, len(t_yx))

        #     for i in range(max_t_num):
        #         t_yxs1 = t_yxs[...][0]

        acc = 0.0
        wrong_idxs = []

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)

            if not multiple_answers:
                tt = t[i*batch_size:(i+1)*batch_size]
                acc += np.sum(y == tt)
            # else:
            #     tt_yxs = t_yxs[i*batch_size:(i+1)*batch_size]
            #     # print(y[0], tt_yxs[0])
            #     acc += np.sum(y in tt_yxs)

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
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db
            if layer_idx+1 < len(self.layers) and type(self.layers[layer_idx+1]) == BatchNormalization:
                # print(i+1, layer_idx+1)
                grads['gamma' + str(i+1)] = self.layers[layer_idx+1].dgamma
                grads['beta' + str(i+1)] = self.layers[layer_idx+1].dbeta

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate(self.layer_idxs_used_params):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]
