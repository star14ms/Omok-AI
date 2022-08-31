from modules.make_datas import board_datas as bd
from network import DeepConvNet
from modules.common.trainer import Trainer
import numpy as np
from modules.common.util import time, Alerm
from modules.plot import plot
import pickle, os

pkl_file = None
pkl_file = "hyperparameter_optimization_info" + ".pkl"
if pkl_file != None:
    with open(pkl_file, 'rb') as f:
        r = pickle.load(f)
    plot.many_accuracy_graphs(r["results_train"], r["results_val"], graph_draw_num=20, col_num=5)
    plot.many_loss_graphs(r["results_losses"], graph_draw_num=20, col_num=5, verbose=False, smooth=False)
    plot.many_loss_graphs(r["results_losses"], graph_draw_num=20, col_num=5, verbose=False, smooth=True)
    exit()

# 학습할 데이터 만들기
x_datas, t_datas = bd.make_4to5()
(x_train, t_train), _ = bd.split_train_test(x_datas, t_datas)
(x_train, t_train), (x_val, t_val) = bd.split_train_val(x_train, t_train)
print(f"\n훈련 데이터, 검증 데이터: {len(x_train)}개, {len(x_val)}개")

# 시험할 최적화 방법 종류와 학습률, 가중치 감소 계수 범위 지정 (log 스케일 10^n)
optimizers = {
    # 'SGD':      {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Momentum': {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Adagrad':  {"lr_min": -1, "lr_max": -5, "wd_min": -8, "wd_max": -4}
    'Adam':     {"lr_min": -1, "lr_max": -5, "wd_min": -8, "wd_max": -4}
    # 'Nesterov': {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Rmsprop':  {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
}
optimization_trial = 50
attempts_number = 1
epochs = 10

give_up = {'epoch': 3} # 'test_acc':0.1
mini_batch_size = 5
results_losses = {}
results_train = {}
results_val = {}
digit = 6

# 학습 정보 출력 함수 정의
def print_learning_info(str_lr, str_wd, val_accs, attempts_number, isgiveup):
    
    average_acc = sum(val_accs) / len(val_accs)          
    
    accuracy_info = "acc: None"
    if not isgiveup:
        accuracy_info = ("acc: %.2f%%" % average_acc).rjust(5)
    
    str_margin_of_error95 = ""
    if attempts_number > 1:
        standard_deviation = (sum( (np.array(val_accs)-average_acc)**2 ) / len(val_accs)) ** (1/2) ### ** 우선도가 /보다 높음
        standard_error = standard_deviation / (len(val_accs) ** (1/2)) 
        margin_of_error95 = round(1.96 * standard_error, 2)

        str_margin_of_error95 = ("(±%.2f%%)" % margin_of_error95).rjust(5)
    
    print(f"lr: {str_lr.rjust(digit+2)} | wd⁁: {str_wd} | {accuracy_info} {str_margin_of_error95}", end=" ")
    
    if isgiveup:
        print(f"-- I gave up")
    elif average_acc > 80:
        print("-- good!! 💚")
    else:
        print()

print(f"trial:{optimization_trial}, attempts:{attempts_number}, epochs:{epochs}")
print("\n탐색 시작!!")
start_time = time.time()
first_trial = True

# 매개변수 최적화 방법과 학습률에 따른 딥러닝 효율 비교
for optimizer in optimizers:

    for i in range(optimization_trial):
        lr = 10 ** np.random.uniform(optimizers[optimizer]["lr_min"], optimizers[optimizer]["lr_max"])
        wd = 10 ** np.random.uniform(optimizers[optimizer]["wd_min"], optimizers[optimizer]["wd_max"])

        val_accs = []
        for attempts in range(attempts_number):
            network = DeepConvNet(mini_batch_size=mini_batch_size, weight_decay_lambda=wd)
            
            trainer = Trainer(epochs=epochs, optimizer=optimizer, optimizer_param={'lr':lr}, verbose=False, 
                mini_batch_size=mini_batch_size, give_up=give_up, verbose_epoch=False,
                network=network, x_train=x_train, t_train=t_train, x_test=x_val, t_test=t_val)
            trainer.train()
            
            if trainer.isgiveup: break

            str_lr = f"%.{digit}f"% lr ### %{digit}.f
            str_wd = str(np.format_float_scientific(wd, precision=2)) # 
            key = (f"{optimizer} | " if len(optimizers) > 1 else "") + f"lr: {str_lr} | wd: {str_wd}" + (f" | {attempts}" if attempts > 1 else "")
            
            results_losses[key] = trainer.train_losses
            results_train[key] = trainer.train_accs ### *100 하면 100번 복사됨
            results_val[key] = trainer.test_accs
            val_accs.append(trainer.test_accs[-1])
        
        if first_trial:
            first_trial = False
            first_trial_time = int(time.time() - start_time)
            estimated_time = first_trial_time * optimization_trial * attempts_number * len(optimizers)
            print(f"예상 소요 시간: {time.str_hms(estimated_time)}")
        if i == 0: 
            print("\noptimizer: " + optimizer)
        print(f"{i+1}".rjust(2)+"/"+f"{optimization_trial}".rjust(2), end=" | ")
        print_learning_info(str_lr, str_wd, val_accs, attempts_number, isgiveup=trainer.isgiveup)

# 학습 끝나면 소요 시간 출력하고, 정보 저장 후, 알람 울리기
print(time.str_hms_delta(start_time))
Alerm()

def save_compare_optim_lr_info(results_train, results_val, results_losses, network_dir=network.network_dir, pkl_dir="saved_pkls"):
    results = {"results_train":results_train, "results_val":results_val, "results_losses":results_losses}
    
    pkl_file = f"hyperparameter_optimization_info".replace(":","").replace("'","").replace('"','')
    
    answer = input(f"\n저장할 위치: {pkl_dir}/(any) | 현재 위치(1) | 저장 안함(2): ")
    if answer == "1":
        pkl_path = f"{pkl_file}.pkl"
    elif answer == "2":
        print("저장 안했다^^")
        return
    else:
        pkl_path = f"{pkl_dir}/{network_dir}/{pkl_file}.pkl"
        
    if not os.path.exists(f"{pkl_dir}/{network_dir}"):
        os.makedirs(f"{pkl_dir}/{network_dir}")
        print(f"{pkl_dir}/{network_dir} 폴더 생성")
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"{pkl_file} 저장 성공!")

save_compare_optim_lr_info(results_train, results_val, results_losses, network_dir=network.network_dir)

print("=========== Hyper-Parameter Optimization Result ===========")
plot.many_accuracy_graphs(results_train, results_val, graph_draw_num=20, col_num=5, sort=True)
plot.many_loss_graphs(results_losses, graph_draw_num=20, col_num=5, sort=True, verbose=False, smooth=False)
plot.many_loss_graphs(results_losses, graph_draw_num=20, col_num=5, sort=True, verbose=False, smooth=False)
