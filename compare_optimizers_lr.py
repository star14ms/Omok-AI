from modules.make_datas import make_datas, split_datas
from network import DeepConvNet
from modules.common.trainer import Trainer
import datetime as dt
import time
# 알람 울리기
import datetime as dt
from selenium import webdriver

start = time.time()

# 학습할 데이터 만들기
x_datas, t_datas, t_datas_real = make_datas._4to5(score=1, blank_score=0)
x_train, t_train, x_test, t_test = split_datas.even_odd(x_datas, t_datas)

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
exponents = {       # 0, -7, -1
    'SGD':      range(-1, -2, -1),
    'Momentum': range(-2, -3, -1), 
    'Adagrad':  range(-2, -3, -1), 
    'Adam':     range(-2, -3, -1), 
    'Nesterov': range(-2, -3, -1), 
    'Rmsprop':  range(-3, -4, -1),
}
attempts_number = 5
epochs = 10

give_up = {'epoch': 3} # 'test_acc':0.1
mini_batch_size = 110

print(f"\nattempts:{attempts_number}, epochs:{epochs}")

# 매개변수 최적화 방법과 학습률에 따른 딥러닝 효율 비교
for optimizer in exponents:
    print("\noptimizer: " + str(optimizer))
    for exponent in exponents[optimizer]:
        acc_list = []
        for _ in range(attempts_number):
            network = DeepConvNet(mini_batch_size=mini_batch_size)
            
            trainer = Trainer(epochs=epochs, optimizer=optimizer, optimizer_param={'lr':10 ** exponent}, verbose=False, 
                mini_batch_size=mini_batch_size, give_up=give_up, verbose_epoch=False,
                network=network, x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test)
            trainer.train()

            if trainer.isgiveup:
                break
            acc = trainer.test_acc_list[-1]*100
            acc_list.append(acc)

        print_learning_info(acc_list, attempts_number)


time_delta = int(time.time() - start)
h, m, s = (time_delta // 3600), (time_delta//60 - time_delta//3600*60), (time_delta % 60)
print(f"\n{h}h {m}m {s}s")

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