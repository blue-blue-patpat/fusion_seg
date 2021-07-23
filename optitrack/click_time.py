import pynput, time

def click_time():
    with pynput.mouse.Events() as event:
        for i in event:
            if isinstance(i, pynput.mouse.Events.Click):
                return(i.x, i.y, i.button, i.pressed, time.time())    

print(click_time()))