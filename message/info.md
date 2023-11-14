# DingTalk Message

## Usage
### Infomation
```python
# Send message every 30 seconds
bot = TimerBot(interval=30)

bot.add_task("test", MSG_INFO)

bot.enable()
```

### Error
```python
# Will @ all group members
bot.add_task("test", MSG_ERROR)
```

### Markdown
```python
import cv2

img = cv2.imread("./test.png")

# img will be resized to 256*256 and compressed if possible
bot.add_md(title="test", text="# 【test】 \n ![img]({})\n".format(bot.img2b64(img)))
```

## Examples
### With Matplotlib
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

bot =TimerBot(3)

fig = plt.figure()

fig.add_subplot(1, 1, 1).plot([1,2,3], [1,2,3])

fig.canvas.draw()

img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

bot.add_md("test", "【TEST】 \n ![img]({})".format(bot.img2b64(img)))

bot.enable()

while True:
    continue
```
