import ctypes
from multiprocessing import Value, Queue, Process
from time import sleep
from dingtalkchatbot.chatbot import DingtalkChatbot


WEBHOOK_KEY = "https://oapi.dingtalk.com/robot/send?access_token=bdd84c8b655db220bce46094b231b1bfecb5527acc812ee2f5dd4bbf6509f219"
MSG_INFO = 1
MSG_ERROR = 2


class TimerBot(Process):
    """
    Kinect Subscriber
    """
    def __init__(self, interval: float = 10) -> None:
        super().__init__(daemon=True)
        self.info_queue = Queue(100)
        self.error_queue = Queue(100)
        
        self.interval = Value(ctypes.c_float, interval)
        self.enabled = Value(ctypes.c_bool, False)

        self.start()
    
    def run(self):
        bot = DingtalkChatbot(WEBHOOK_KEY)

        while True:
            if not self.enabled.value:
                sleep(0.5)
                continue
            if not self.error_queue.empty():
                text = self.error_queue.get()
                bot.send_text("【ERROR】" + text, True)
            if not self.info_queue.empty():
                text = self.info_queue.get()
                bot.send_text("【INFO】" + text, False)
            # Sleep only if no error waits in queue
            if self.error_queue.empty():
                sleep(self.interval.value)

    def enable(self):
        self.enabled.value = True
    
    def disable(self):
        self.enabled.value = False

    def add_task(self, text: str, level: int = 1):
        if level == 1:
            if self.info_queue.full():
                self.info_queue.get()
                self.info_queue.get()
                self.info_queue.get()

            self.info_queue.put(text)
        else:
            if self.error_queue.full():
                self.error_queue.get()
                self.error_queue.get()
                self.error_queue.get()
            self.error_queue.put(text)

    def print(self, text):
        print(text)
        if self.enabled.value:
            self.add_task(text)
