import time, win32con, win32api, win32gui, ctypes
from pywinauto import clipboard # 채팅창내용 가져오기 위해
import pyperclip
import ctypes

CF_TEXT = 1

kernel32 = ctypes.windll.kernel32
kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
kernel32.GlobalLock.restype = ctypes.c_void_p
kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
user32 = ctypes.windll.user32
user32.GetClipboardData.restype = ctypes.c_void_p

def get_clipboard_text():
    user32.OpenClipboard(0)
    try:
        if user32.IsClipboardFormatAvailable(CF_TEXT):
            data = user32.GetClipboardData(CF_TEXT)
            data_locked = kernel32.GlobalLock(data)
            text = ctypes.c_char_p(data_locked)
            value = text.value
            kernel32.GlobalUnlock(data_locked)
            if value==None:
                return None
            return value.decode('utf-8')
    finally:
        user32.CloseClipboard()

# Yixin 프로그램 창 (열려있는 상태, 최소화 X, 창뒤에 숨어있는 비활성화 상태 가능)
hwndMain = win32gui.FindWindow(None, "Yixin")

PBYTE256 = ctypes.c_ubyte * 256
_user32 = ctypes.WinDLL("user32")
GetKeyboardState = _user32.GetKeyboardState
SetKeyboardState = _user32.SetKeyboardState
PostMessage = win32api.PostMessage
SendMessage = win32gui.SendMessage
FindWindow = win32gui.FindWindow
IsWindow = win32gui.IsWindow
GetCurrentThreadId = win32api.GetCurrentThreadId
GetWindowThreadProcessId = _user32.GetWindowThreadProcessId
AttachThreadInput = _user32.AttachThreadInput

MapVirtualKeyA = _user32.MapVirtualKeyA
MapVirtualKeyW = _user32.MapVirtualKeyW

MakeLong = win32api.MAKELONG
w = win32con

X_line = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]
Y_line = [str(i) for i in range(15, 0, -1)]
zero_zero_pos = None
interval = 20

class Yixin:

    def init(sleep_sec=2):
        print("\nYixin 기본 설정 요구사항: 'Computer plays...' 양쪽 다 끄기")
        print("바둑판 클릭 설정 (0,0 좌표에 마우스 놓을 준비)")
        time.sleep(sleep_sec)
        Yixin.set_zero_zero_pos()
        
    def reset():
        Yixin.delete_log()
        Yixin.Click_setting("undo_all")
        # xy_pre = None ### 초기화를 안하면 xy_pre == xy가 영원히 True ([None, None]으로) 

    def Click_setting(object_):
        if "undo" in object_:
            # print("undo")
            Click(-40, (30 if "all" not in object_ else 0))
        elif "redo" in object_:
            # print("redo")
            Click(-40, 50)
        else:
            Click(0,-25)
            time.sleep(0.1)
            Click(0, (20 if object_ == "plays_b" else 30), start_zero=False)
        time.sleep(0.1)

    def last_stone_save(): # Yixin의 수 저장
        texts = Yixin.copy_log()  # 채팅내용 가져오기
        if len(texts.split("\n")) < 3: return None
        decisions = texts.split("\n")[-3]
        # print(decisions.split(" ")[0])
    
        if decisions.split(" ")[0] == "BESTLINE:":
            if "[" not in decisions: return [None, None]
            str_xy = decisions.replace('[', '').replace(']', '').split(" ")[1]
            xy = [X_line.index(str_xy[0]), Y_line.index(str_xy[1:])] ### str_xy[1] = G10 -> 1
            # print(str_xy)
            return xy
        elif "PASS" in decisions: return [None, "PASS"]

        return None
    
    def copy_log(interval=interval): # Yixin창의 모든 텍스트를 클립보드에 복사 ( Ctrl + A , C )
        global hwndMain
        ctext = None
        while ctext == None: # 복사 실패했을 때 오류 방지
            PostKeyEx(hwndMain, ord('A'), [w.VK_CONTROL], False)
            time.sleep(0.2)
            PostKeyEx(hwndMain, ord('X'), [w.VK_CONTROL], False)
            time.sleep(0.2)
            ctext = get_clipboard_text() # clipboard.GetData()
    
        return ctext
    
    def delete_log():
        global hwndMain
        Click(16*interval, 0)
        PostKeyEx(hwndMain, ord('A'), [w.VK_CONTROL], False)
        time.sleep(0.1)
        PostKeyEx(hwndMain, ord('\b'), [], False)
        time.sleep(0.1)

    def think_win_xy(whose_turn=-1, interval=interval, undo=False):
        Click(16*interval, 0)
        xy = None ### 여기다 xy = None을 넣으면 이전 수가 저장이 안 됨
        while xy == None: ### pre를 사용해도 이전 턴과 답이 같을 수 있음 (두길 원하지만 못 둔 곳을 뒀다고 판단하게 됨)
            xy = Yixin.last_stone_save()
            time.sleep(0.2)
        Yixin.delete_log()
        pyperclip.copy(' ')
        # print(f"{X_line[xy[0]]}{Y_line[xy[1]]}" if xy[0]!=None else xy)

        if undo or xy[0]==None:
            # print(f"(undo={undo}, xy={xy}, turn: {whose_turn})")
            Yixin.Click_setting("undo")
        if whose_turn == 1: ### 컴퓨터가 두지 않는 상태 Click(1, 0) X
            Yixin.Click_setting("plays_b")
            Yixin.Click_setting("plays_w")
        if xy[1]=="PASS": 
            Yixin.Click_setting("redo")

        return xy[0], xy[1]

    def set_zero_zero_pos(ready=3):
        print("마우스를 0,0 자리에 가져다 놓아라")
        for i in range(ready):
            print(ready-i)
            time.sleep(1)
        
        global zero_zero_pos
        zero_zero_pos = win32api.GetCursorPos()

    def test(interval=interval, sleep_sec=0.1):
        Yixin.Click_setting("plays_w")
        for y in range(15):
            for x in range(15):
                Click(x, y, board_xy=True)
                time.sleep(sleep_sec)
                Yixin.Click_setting("undo")


def Click(x, y, start_zero=True, board_xy=False):
    if start_zero:
        global zero_zero_pos
        win32api.SetCursorPos(zero_zero_pos)
    if board_xy: x, y = x*interval, y*interval
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, x,y,0,0)
    # time.sleep(1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 1,0,0,0) # 오류 보완

def PostKeyEx(hwnd, key, shift, specialkey): # 조합키 쓰기 위해
    if IsWindow(hwnd):

        ThreadId = GetWindowThreadProcessId(hwnd, None)

        lparam = MakeLong(0, MapVirtualKeyA(key, 0))
        msg_down = w.WM_KEYDOWN
        msg_up = w.WM_KEYUP

        if specialkey:
            lparam = lparam | 0x1000000

        if len(shift) > 0:  # Если есть модификаторы - используем PostMessage и AttachThreadInput
            pKeyBuffers = PBYTE256()
            pKeyBuffers_old = PBYTE256()

            SendMessage(hwnd, w.WM_ACTIVATE, w.WA_ACTIVE, 0)
            AttachThreadInput(GetCurrentThreadId(), ThreadId, True)
            GetKeyboardState(ctypes.byref(pKeyBuffers_old))

            for modkey in shift:
                if modkey == w.VK_MENU:
                    lparam = lparam | 0x20000000
                    msg_down = w.WM_SYSKEYDOWN
                    msg_up = w.WM_SYSKEYUP
                pKeyBuffers[modkey] |= 128

            SetKeyboardState(ctypes.byref(pKeyBuffers))
            time.sleep(0.01)
            PostMessage(hwnd, msg_down, key, lparam)
            time.sleep(0.01)
            PostMessage(hwnd, msg_up, key, lparam | 0xC0000000)
            time.sleep(0.01)
            SetKeyboardState(ctypes.byref(pKeyBuffers_old))
            time.sleep(0.01)
            AttachThreadInput(GetCurrentThreadId(), ThreadId, False)

        else:  # Если нету модификаторов - используем SendMessage
            SendMessage(hwnd, msg_down, key, lparam)
            SendMessage(hwnd, msg_up, key, lparam | 0xC0000000)
    
def main():
    Yixin.init()
    Yixin.test()

    # x, y = Yixin.think_win_xy(whose_turn=1, undo=True)
    # print(f"Yixin_bxy: x={x}, y={y}")
    
    # Yixin.Click_network_xy(14, 14)
    # x, y = Yixin.think_win_xy()
    # print(f"Yixin_wxy: x={x}, y={y}")

if __name__ == '__main__':
    main()
