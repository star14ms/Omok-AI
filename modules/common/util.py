# coding: utf-8
import numpy as np
import sys
import time as base_time

def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


class time():
    
    def time() -> float:
        return base_time.time()

    def sleep(secs: float) -> None:
        base_time.sleep(secs)

    def sec_to_hms(second):
        second = int(second)
        h, m, s = (second // 3600), (second//60 - second//3600*60), (second % 60)
        return h, m, s
    
    def str_hms_delta(start_time, hms=False, rjust=False, join=':'):
        time_delta = base_time.time() - start_time
        h, m, s = time.sec_to_hms(time_delta)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")

    def str_hms(second, hms=False, rjust=False, join=':'):
        h, m, s = time.sec_to_hms(second)
        if not hms:
            return "{1}{0}{2:02d}{0}{3:02d}".format(join, h, m, s)
        elif rjust: 
            m, s = str(m).rjust(2), str(s).rjust(2)
        
        return str(f"{h}h{join}{m}m{join}{s}s")


def Alerm(webdriver_path=r'C:\Users\danal\Documents\programing\chromedriver.exe', loading_sec=7):
    import datetime as dt
    from selenium import webdriver
    
    now = dt.datetime.today()

    if int(now.strftime('%S')) < 60 - loading_sec:
        alarm_time = now + dt.timedelta(minutes=1)
    else:
        alarm_time = now + dt.timedelta(minutes=2)

    alarm_time = alarm_time.strftime('%X')
    driver = webdriver.Chrome(webdriver_path)
    driver.get(f'https://vclock.kr/#time={alarm_time}&title=%EC%95%8C%EB%9E%8C&sound=musicbox&loop=1')
    driver.find_element_by_xpath('//*[@id="pnl-main"]').click()
    input()


class bcolors:
    
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ORANGE = '\u001b[38;5;208m'

    c = [HEADER, OKBLUE, OKCYAN, OKGREEN, WARNING, FAIL, ENDC, BOLD, UNDERLINE, ORANGE]

    def test():
        for color in bcolors.c:
            print(color + "Warning: No active frommets remain. Continue?" + bcolors.ENDC)

    def according_to_score(x):
        if x < 1:
            return bcolors.ENDC
        elif 1 <= x < 20:
            return bcolors.FAIL
        elif 20 <= x < 40:
            return bcolors.ORANGE
        elif 40 <= x < 60:
            return bcolors.WARNING
        elif 60 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD
    
    def according_to_chance(x):
        if x < 0.5:
            return bcolors.ENDC
        elif 0.5 <= x < 5:
            return bcolors.FAIL
        elif 5 <= x < 20:
            return bcolors.ORANGE
        elif 20 <= x < 50:
            return bcolors.WARNING
        elif 50 <= x < 80:
            return bcolors.OKGREEN
        elif 80 <= x < 95:
            return bcolors.OKBLUE
        elif 95 <= x < 99:
            return bcolors.HEADER
        else:
            return bcolors.BOLD

    def ANSI_codes():
        for i in range(0, 16):
            for j in range(0, 16):
                code = str(i * 16 + j)
                sys.stdout.write(u"\u001b[38;5;" + code + "m" + code.ljust(4))
            print(u"\u001b[0m")


import logging
def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # 로그 포멧 정의
    formatter = logging.Formatter(
        '\n"%(pathname)s", line %(lineno)d, in %(module)s\n%(levelname)-8s: %(message)s')
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger


def rotate_2dim_array(arr, d, add_0dim=True): # 2차원 배열을 90도 단위로 회전해 반환한다. 
    # 이때 원 배열은 유지되며, 새로운 배열이 탄생한다. 
    # 이는 회전이 360도 단위일 때도 해당한다. 
    # 2차원 배열은 행과 열의 수가 같은 정방형 배열이어야 한다.
    # arr: 회전하고자 하는 2차원 배열. 입력이 정방형 행렬이라고 가정한다. 
    # d: 90도씩의 회전 단위. -1: -90도, 1: 90도, 2: 180도, ...

    size = len(arr)
    ret = np.array([ [0]*size for _ in range(size) ])
    N = size - 1
    if d % 8 not in (1, 2, 3, 4, 5, 6, 7): 
        for r in range(size): 
            for c in range(size): 
                ret[r][c] = arr[r][c] 
    elif d % 8 == 1:
        for r in range(size): 
            for c in range(size): 
                ret[c][N-r] = arr[r][c] 
    elif d % 8 == 2: 
        for r in range(size): 
            for c in range(size): 
                ret[N-r][N-c] = arr[r][c] 
    elif d % 8 == 3: 
        for r in range(size): 
            for c in range(size): 
                ret[N-c][r] = arr[r][c] 

    elif d % 8 == 4:
        for r in range(size): 
            for c in range(size): 
                ret[r][N-c] = arr[r][c]
    elif d % 8 == 5: # arr.T
        for r in range(size): 
            for c in range(size): 
                ret[N-c][N-r] = arr[r][c]
    elif d % 8 == 6:
        for r in range(size): 
            for c in range(size): 
                ret[N-r][c] = arr[r][c]
    elif d % 8 == 7:
        for r in range(size): 
            for c in range(size): 
                ret[c][r] = arr[r][c]

    if not add_0dim:
        return ret
    else:
        return ret.reshape(1, size, size)

def rotate_2dim_array_idx(dim2_arr_idx, d, size=15, add_0dim=True):
    x, y = dim2_arr_idx % size, dim2_arr_idx // size
    N = size -1

    if d % 8 not in (1, 2, 3, 4, 5, 6, 7): 
        pass
    elif d % 8 == 1:
        x, y = N-y, x
    elif d % 8 == 2:
        x, y = N-x, N-y
    elif d % 8 == 3:
        x, y = y, N-x

    elif d % 8 == 4:
        x, y = N-x, y
    elif d % 8 == 5: # arr.T
        x, y = N-y, N-x
    elif d % 8 == 6:
        x, y = x, N-y
    elif d % 8 == 7:
        x, y = y, x

    if not add_0dim:
        return y*15 + x
    else:
        return np.array([[y*15 + x]])
