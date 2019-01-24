import pyautogui as pa
import random

pa.PAUSE = 3

key_dict = {
    'ctrl': (135,1010), 'alt': (300,1010), 
}

def press_key(key_name, delay):
    rand_num = random.randint(-5,5)
    key_pos = key_dict[key_name]
    pa.doubleClick(key_pos[0]+rand_num, key_pos[1]+rand_num)
    pa.PAUSE = delay

for i in range(100):
    press_key('alt', 0.1)
    press_key('alt', 0.1)
    press_key('alt', 0.1)
    press_key('alt', 0.1)
    press_key('alt', 0.1)
    press_key('ctrl', 0.1)
    press_key('ctrl', 1)


