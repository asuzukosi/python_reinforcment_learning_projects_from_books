import gymnasium as gym
import queue, threading, time
from pynput.keyboard import Listener, Key
from utils import cv2_resize_image


def Keyboard(queue):
    # create on press callback function for when a key is pressed, you have access to the key that was pressed
    def on_press(key):
        if key == Key.esc:
            queue.put(-1)
        elif key == Key.space:
            queue.put(ord(' '))
        else:
            key = str(key).replace("'", '')
            if key in ['w', 'a', 's', 'd']:
                queue.put(ord(key))

    # create an on release callback function for when a key is released
    def on_release(key):
        if key == Key.esc:
            return False
    
    # start listener thead that listens for key presses on the keyboard
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def start_game(queue):
    atari = gym.make("Breakout-v4", render_mode="human")
    key_to_act = atari.env.get_keys_to_action()
    key_to_act = {k[0]: a for k, a in key_to_act.items() if len(k) > 0 }
    (observation, _) = atari.reset()
    
    import numpy
    from PIL import Image

    img = numpy.dot(observation, [0.2126, 0.7152, 0.0722])
    img = cv2_resize_image(img)
    img = Image.fromarray(img)
    img.save(f"images/{0}.jpg")
    
    
    while True:
        atari.render()
        action = 0 if queue.empty() else queue.get(block=False)
        if action == -1:
            break
        action = key_to_act.get(action, 0)
        observation, reward, terminated, truncated, info = atari.step(action)
        if action != 0:
            pass
            # print(f"action: {action}, reward: {reward}")
        if terminated or truncated:
            print("Game finished")
            break
        
        time.sleep(0.05)
        

if __name__ == "__main__":
    my_queue = queue.Queue(maxsize=10)
    keyboard = threading.Thread(target=Keyboard, args=(my_queue,))
    keyboard.start()
    start_game(my_queue)
   