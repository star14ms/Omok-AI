from modules.make_datas import make_datas, split_datas
from network import DeepConvNet
from modules.common.trainer import Trainer
import datetime as dt
import time
import numpy as np
from modules.common.util import print_time, Alerm
from modules.plot import plot
import pickle

start_time = time.time()

# 학습할 데이터 만들기
x_datas, t_datas = make_datas._4to5(score=1, blank_score=0)
x_train, t_train, x_test, t_test = split_datas.even_odd(x_datas, t_datas)

# 학습 정보 출력 함수 정의
def print_learning_info(optimizer, lr, last_val_accs, isgiveup):
    
    average_accuracy = sum(last_val_accs)/len(last_val_accs)
    if not isgiveup:
        accuracy_info = str("acc: {}%".format( round(average_accuracy, 2) ))
    else:
        accuracy_info = str("acc: None")

    standard_deviation = sum((np.array(last_val_accs)-average_accuracy) ** 2) / len(last_val_accs) ** 1/2
    
    if lr < 0.001:
        lr = np.format_float_scientific(lr)
    
    learning_info = str(f"{optimizer} | lr: {lr} | {accuracy_info} (표준편차: {standard_deviation}")
    print(learning_info, end=" ")
    
    if isgiveup:
        print(f"-- I gave up")
    elif average_accuracy > 80:
        print("-- good!! 💚")
    else:
        print()

# 매개변수 생성
exponents = {
    'SGD':      range(-2, -3, -1),
    # 'Momentum': range(0, -14, -1),
    # 'Adagrad':  range(0, -14, -1),
    # 'Adam':     range(0, -14, -1),
    # 'Nesterov': range(0, -14, -1),
    # 'Rmsprop':  range(0, -14, -1),
}
attempts_number = 3
epochs = 1

give_up = {'epoch': 3} # 'test_acc':0.1
mini_batch_size = 110
results_train = {}
results_val = {}
print(f"\nattempts:{attempts_number}, epochs:{epochs}")

# 매개변수 최적화 방법과 학습률에 따른 딥러닝 효율 비교
for optimizer in exponents:
    print("\noptimizer: " + optimizer)
    
    for exponent in exponents[optimizer]:
        lr = 10 ** exponent
        attempts = 0
        last_val_accs = []

        for _ in range(attempts_number):
            attempts += 1
            network = DeepConvNet(mini_batch_size=mini_batch_size)
            
            trainer = Trainer(epochs=epochs, optimizer=optimizer, optimizer_param={'lr':lr}, verbose=False, 
                mini_batch_size=mini_batch_size, give_up=give_up, verbose_epoch=False,
                network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
            trainer.train()

            if trainer.isgiveup: break

            key = f"{optimizer}_{attempts} | lr: {lr} "
            results_train[key] = trainer.train_accs ### *100 하면 100번 복사됨
            results_val[key] = trainer.test_accs
            last_val_accs.append(trainer.test_accs[-1])
            
        print_learning_info(optimizer, lr, last_val_accs, isgiveup=trainer.isgiveup)

# 학습 끝나면 소요 시간 출력하고 알람 울리기
print_time(start_time)
# Alerm()

pkl_file = "Hyper_Parameter_Optimization.pkl"
results = {"results_train": results_train, "results_val": results_val}

with open(pkl_file, 'wb') as f:
    pickle.dump(results, f)

print("=========== Hyper-Parameter Optimization Result ===========")
plot.many_accuracy_graphs(results_train, results_val, graph_draw_num=20, col_num=5, sort=False)

with open(pkl_file, 'rb') as f:
    r = pickle.load(f)
plot.many_accuracy_graphs(r["results_train"], r["results_val"], graph_draw_num=20, col_num=5, sort=False)