# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from common.optimizer import *

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100, give_up={'epoch':None, 'test_loss':None},
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
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.test_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train(self):
        self.isgiveup = False
        train_loss_append = self.train_loss_list.append
        test_loss_append = self.test_loss_list.append
        train_acc_append = self.train_acc_list.append
        test_acc_append = self.test_acc_list.append

        for i in range(self.max_iter):

            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            grads = self.network.gradient(x_batch, t_batch)
            self.optimizer.update(self.network.params, grads)
            self.network.learning_num += self.batch_size
            
            loss = self.network.loss(x_batch, t_batch)
            self.loss = round(loss, 4)
            train_loss_append(loss)
            # test_loss = self.network.loss(self.x_test, self.t_test)
            # test_loss_append(test_loss)
            if self.verbose: print(f"{i+1}/{self.max_iter} - train loss:" + str(round(loss, 4)))
            
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
                    print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(round(train_acc, 4)) + ", test acc:" + str(round(train_acc, 4)) + " ===")
                    # print("=== " + str(self.current_epoch) + " / train:" + format(loss, ".10f") + ", test:" + format(test_loss, ".10f") + " (loss) ===")
                    # print("=== epoch: " + str(self.current_epoch) + "train_loss: " + str(round(loss, 3)) + " ===")
                
                if self.current_epoch == self.give_up['epoch'] and self.give_up != {}:
                    # if np.isnan(loss) or loss > self.give_up['test_loss']:
                    #     if self.verbose: print(f"I give up! (loss:{self.loss})")
                    #     self.isgiveup = True
                    #     break
                    if test_acc < self.give_up['test_acc']:
                        if self.verbose: print("I give up!")
                        self.isgiveup = True
                        break
                    
            self.current_iter += 1

        # test_acc = self.network.accuracy(self.x_test, self.t_test)

        # print("=============== Final Test Accuracy ===============")
        # print("test acc:" + str(test_acc))

