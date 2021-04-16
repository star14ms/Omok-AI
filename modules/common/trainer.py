# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *
import pickle
import math # save_graph_datas에서 acc에 floor사용

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100, give_up={'epoch':-1},
                 optimizer='SGD', optimizer_param={'lr':0.01},
                 evaluate_sample_num_per_epoch=None, verbose=True, verbose_epoch=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        self.give_up = give_up
        self.loss = 0
        self.isgiveup = False
        self.verbose_epoch = verbose_epoch

        # optimizer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(int(self.train_size / mini_batch_size), 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_losses = []
        # self.test_losses = []
        self.train_accs = []
        self.test_accs = []
        self.graph_datas = {}

    def train(self):
        self.isgiveup = False
        train_loss_append = self.train_losses.append
        # test_loss_append = self.test_losses.append
        train_acc_append = self.train_accs.append
        test_acc_append = self.test_accs.append

        for i in range(self.max_iter):

            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            self.network.learning_num += self.batch_size
            
            loss = self.network.loss(x_batch, t_batch)
            self.loss = loss
            train_loss_append(loss)
            # test_loss = self.network.loss(self.x_test, self.t_test)
            # test_loss_append(test_loss)
            if self.verbose: print(f"{i+1}/{self.max_iter} - loss: " + str( round(loss, 4) ))
            
            if self.current_iter % self.iter_per_epoch == 0:
                self.current_epoch += 1
                
                x_train_sample, t_train_sample = self.x_train, self.t_train
                x_test_sample, t_test_sample = self.x_test, self.t_test
                if not self.evaluate_sample_num_per_epoch is None:
                    length = self.evaluate_sample_num_per_epoch
                    x_train_sample, t_train_sample = self.x_train[:length], self.t_train[:length]
                    x_test_sample, t_test_sample = self.x_test[:length], self.t_test[:length]
                    
                train_acc = self.network.accuracy(x_train_sample, t_train_sample) # , self.t_train.shape[0]
                train_acc_append(train_acc)
                test_acc = self.network.accuracy(x_test_sample, t_test_sample) # , self.t_test.shape[0]
                test_acc_append(test_acc)

                if self.verbose_epoch: 
                    print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(
                        math.floor(train_acc*100)/100) + "%, test acc:" + str(
                        math.floor(test_acc*100)/100) + "% ===")
                    # print("=== " + str(self.current_epoch) + " / train:" + format(loss, ".10f") + ", test:" + format(test_loss, ".10f") + " (loss) ===")
                    # print("=== epoch: " + str(self.current_epoch) + "train_loss: " + str(round(loss, 3)) + " ===")
                
                if self.give_up['epoch'] == self.current_epoch:
                    if 'test_loss' in self.give_up and (np.isnan(loss) or loss > self.give_up['test_loss']):
                        if self.verbose: print(f"I give up! (loss:{self.loss})")
                        self.isgiveup = True
                        break
                    if 'test_acc' in self.give_up and test_acc < self.give_up['test_acc']:
                        if self.verbose: print("I give up!")
                        self.isgiveup = True
                        break
                    
            self.current_iter += 1

        train_acc = self.network.accuracy(self.x_train, self.t_train)
        train_acc_append(train_acc)
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        test_acc_append(test_acc)

        self.graph_datas = {
            "train_losses": self.train_losses, 
            "train_accs": self.train_accs, 
            "test_accs": self.test_accs
            }
        
        if self.verbose_epoch:
            print("=============== Final Test Accuracy ===============")
            print("train acc: " + str(round(train_acc, 2)) + "%", end=", ")
            print("test acc: " + str(round(test_acc, 2)) + "%")


    def save_graph_datas(self, network, save_inside_dir=True, pkl_dir="saved_pkls"):
        
        if len(self.test_accs) == 0:
            acc = network.saved_network_pkl.split(" ")[-3].lstrip("acc_")
        else:
            acc = math.floor(self.test_accs[-1]*100)/100
        file_name = f"{self.optimizer.__class__.__name__} lr_{self.optimizer.lr} ln_{network.learning_num} acc_{acc} {network.network_dir} graphdata.pkl"
        file_path = file_name

        if save_inside_dir:
            file_path = f"{pkl_dir}/{network.network_dir}/" + file_path
    
        if not os.path.exists(f"{pkl_dir}/{network.network_dir}"):
            os.makedirs(f"{pkl_dir}/{network.network_dir}")
            print(f"{pkl_dir}/{network.network_dir} 폴더 생성")
    
        with open(file_path, 'wb') as f:
            pickle.dump(self.graph_datas, f)
    
        print(f"\n그래프 데이터 저장 성공!\n({file_name})")


def load_graph_datas(file_name="graphdata.pkl", pkl_dir="saved_pkls"): ### class 밖에선 self 지워야 함

    if (".pkl" not in file_name) or (file_name[-4:] != ".pkl"):
        file_name = f"{file_name}.pkl"
    file_path = file_name

    # network_dir = file_path.split(" [")[0].split(" ")[-1] + " [" + file_path.split(" [")[1].rstrip(" params.pkl")
    network_dir = file_path.split(" ")[-2] # " " -> "_" 같이 수정해주어야 함
    file_path = f"{network_dir}/{file_path}"

    if (file_path.split("/")[:-1] != f"{pkl_dir}"):
        file_path = f"{pkl_dir}/{file_path}"

    if not os.path.exists(f"{pkl_dir}/{network_dir}"):
        os.makedirs(f"{pkl_dir}/{network_dir}")
        print(f"Error: {pkl_dir}/{network_dir} 폴더가 없어서 생성")

    if not os.path.exists(file_path):
        print(f"Error: {file_path} 파일이 없음")
        file_path = file_name
        if not os.path.exists(file_path):
            print(f"Error: {file_path} 파일이 없음")
            file_path = f"{pkl_dir}/{file_name}"
            if not os.path.exists(file_path):
                print(f"Error: {file_name} 파일이 없음")
                return

    # main process
    with open(file_path, 'rb') as f:
        graph_datas = pickle.load(f)
        
    print(f"\n그래프 데이터 불러오기 성공!\n({file_name})")
    return graph_datas
    