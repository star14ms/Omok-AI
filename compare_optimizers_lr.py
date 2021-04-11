from learning.make_datas import make_4to5_datas
from learning.deep_convnet import DeepConvNet
from learning.common.trainer import Trainer
import datetime, time

start = time.time()

# 학습할 데이터 만들기
x_datas, t_datas, t_datas_real = make_4to5_datas(score=1, blank_score=0)
len_datas = x_datas.shape[0] # 가로 ~825, 세로 825~1650, \대각선 1650~2255, /대각선 2255~2860
x_train, t_train = x_datas[range(0, len_datas, 2)], t_datas_real[range(0, len_datas, 2)] ### 0, 1 X
x_test, t_test = x_datas[range(1, len_datas, 2)], t_datas_real[range(1, len_datas, 2)] ### 0, 1 X

# 학습 정보 출력 함수 정의
def print_learning_info(acc_list, attempts_number):
    
    if not trainer.isgiveup:
        accuracy = round(sum(acc_list)/attempts_number, 2)
        accuracy_info = str(" \ acc: " + str(accuracy) + "%")
    else:
        accuracy_info = str(" \ acc: None")

    if exponent <= 0:
        lr = format(10**exponent, f".{-exponent}f")
    else:
        lr = 10**exponent

    learning_info = str(optimizer+" \ "+"lr: "+str(lr)+accuracy_info)
    print(learning_info, end=" ")
    
    if trainer.isgiveup:
        print(f"-- I gave up")
    elif accuracy > 80:
        print("-- good!! 💚")
    else:
        print()

# 매개변수 생성
# 'SGD', 'Momentum', 'Adagrad', 'Adam', 'Nesterov', 'Rmsprop'
optimizers = ['SGD', 'Momentum', 'Nesterov', 'Rmsprop'] 
exponents = range(1, -4, -1) # 1, -7, -1

attempts_number = 10
give_up={'epoch':3, 'test_acc':0.1}

epochs = 10
mini_batch_size = 110

# 매개변수 최적화 방법과 학습률에 따른 딥러닝 효율 비교
for optimizer in optimizers:
    print("\noptimizer: " + str(optimizer))
    for exponent in exponents:
        acc_list = []
        for _ in range(attempts_number):
            network = DeepConvNet(
                layers_info = [
                    'convEnd', 'Relu',
                    'softmax'
                ], params = [(1, 15, 15),
                    # {'filter_num':10, 'filter_size':1, 'pad':0, 'stride':1},
                    # {'filter_num':10, 'filter_size':3, 'pad':1, 'stride':1},
                    # {'filter_num':10, 'filter_size':5, 'pad':2, 'stride':1},
                    # {'filter_num':10, 'filter_size':7, 'pad':3, 'stride':1},
                    {'filter_num':10, 'filter_size':9, 'pad':4, 'stride':1},
                ], mini_batch_size=mini_batch_size)
            
            trainer = Trainer(epochs=epochs, optimizer=optimizer, optimizer_param={'lr':10 ** exponent}, verbose=False, 
                mini_batch_size=mini_batch_size, give_up=give_up, verbose_epoch=False,
                network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
            trainer.train()

            if trainer.isgiveup:
                break
            acc = trainer.test_acc_list[-1]*100
            acc_list.append(acc)

        print_learning_info(acc_list, attempts_number)


time_delta = time.time() - start
print(f"\n{ time_delta // 3600 }h { time_delta//60 - time_delta//3600*60 }m { time_delta % 60 }s")

# 학습 끝나면 알람 울리기
now = dt.datetime.today()
if int(now.strftime('%S')) < 52:
    alarm_time = now + dt.timedelta(minutes=1)
else:
    alarm_time = now + dt.timedelta(minutes=2)
alarm_time = alarm_time.strftime('%X')
driver = webdriver.Chrome(r'C:\Users\danal\Documents\programing\chromedriver.exe')
driver.get(f'https://vclock.kr/#time={alarm_time}&title=%EC%95%8C%EB%9E%8C&sound=musicbox&loop=1')
driver.find_element_by_xpath('//*[@id="pnl-main"]').click()