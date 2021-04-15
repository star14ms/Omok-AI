import numpy as np
import matplotlib.pyplot as plt
from common.util import smooth_curve
import math
# %matplotlib inline 이게 뭐고, 어떻게 사용하는거지???
    
class plot:
    
    def loss_graph(train_losses, smooth=True):
        x = np.arange(len(train_losses))
    
        if smooth:
            plt.plot(x, smooth_curve(train_losses), f"-", label="loss")
        else:
            plt.plot(x, train_losses, f"-", label="loss")
    
        plt.xlabel("iterations")
        plt.ylabel("loss")
        ylim = math.ceil(train_losses[0]*10)/10
        plt.ylim(0, ylim)
        plt.legend(loc='upper right')
        plt.show()
    

    def loss_graphs(train_losses, str_optims, smooth=True, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'), ylim=2.5):
        x = np.arange(len(train_losses[str_optims[0]]))
        max_y = 0
        for str_optim, color in zip(str_optims, colors):
            y = smooth_curve(train_losses[str_optim]) if smooth else train_losses[str_optim]
            plt.plot(x, y, label=str_optim) # f"-{color}"
            max_y = train_losses[str_optim][0] if max_y > train_losses[str_optim][-1] else max_y
        plt.xlabel("iterations")
        plt.ylabel("loss")
        ylim = math.ceil(max_y*10)/10
        plt.ylim(0, ylim)
        plt.legend(loc='upper right')
   

    def accuracy_graph(train_accs, test_accs):
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_accs))
        plt.plot(x, train_accs, f"-", label='train acc')
        plt.plot(x, test_accs, f"--", label='test acc')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right') # 그래프 이름 표시
        plt.show()


    def accuracy_graphs(train_accs, test_accs, str_optims, colors=('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')):
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_accs['SGD']))
        for str_optim, color in zip(str_optims, colors):
            plt.plot(x, train_accs[str_optim], f"-{color}", label=str_optim+'train acc')
            plt.plot(x, test_accs[str_optim], f"--{color}", label=str_optim+'test acc')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right') # 그래프 이름 표시
    

    def many_accuracy_graphs(results_train, results_val, graph_draw_num=9, col_num=3, sort=True):
        row_num = int(np.ceil(graph_draw_num / col_num))
        i = 0
    
        if sort:
            results_val_items = sorted(results_val.items(), key=lambda x:x[1][-1], reverse=True)
        else:
            results_val_items = results_val.items()
        
        for key, val_accs in results_val_items:
            val_acc = str(round(val_accs[-1], 2)) ### str().rjust(5)
            print(f"{'B' if sort else 'T'}est-{i+1} (val acc:{val_acc.rjust(5)}%) | " + key)
            
            plt.subplot(row_num, col_num, i+1)
            plt.title(key)
            plt.ylim(0, 100)
            if i % 5: plt.yticks([])
            plt.xticks([])
            x = np.arange(len(val_accs))
            plt.plot(x, val_accs)
            plt.plot(x, results_train[key], "--")
            i += 1
            
            if i >= graph_draw_num:
                break
        
        plt.show()
    