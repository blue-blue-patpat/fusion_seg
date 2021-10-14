import os
import time
import numpy as np
import pandas as pd
from dataloader.result_loader import ResultLoader
from dataloader.utils import file_paths_from_dir, ymdhms_time
from message.dingtalk import TimerBot


class FileMonitor():
    def __init__(self, root_path, dir_key, ext: str = "*", interval: int = 600, at_mobiles: str = "") -> None:
        self.root_path = root_path
        self.dir_key = dir_key
        self.interval = interval
        self.at_mobiles = at_mobiles

        self.files = ResultLoader(root_path)

        self.bot = TimerBot(interval/5)
        self.bot.enable()

    def run(self):
        while True:
            msg = self.check_files()
            self.bot.add_md("Monitor Report", msg, at_mobiles=self.at_mobiles)
            time.sleep(self.interval)

    def update_dir(self):
        dir_list = file_paths_from_dir(self.root_path, '!', enable_print=True, collect_dir=True)
        params = [
            dict(tag=d, ext="*") for d in dir_list if d[-len(self.dir_key):] == self.dir_key
        ]
        self.files.params = params
        self.files.run()

    def check_files(self):
        self.update_dir()

        msg = "# 【FileMonitor】 : {}  \n| task | mdf | cnt | avr |  \n| ---- | ---- | ---- | ---- |  \n".format(ymdhms_time())
        
        for key, value in self.files.file_dict.items():
            assert isinstance(value, pd.DataFrame)
            if value.empty or "id" not in value.columns:
                continue
            
            task = key.split('/')[-3]

            value.fillna(0, inplace=True)
            value["id"] = value["id"].astype(int)
            value.sort_values("id", ascending=True, inplace=True)
            
            if len(value) == 0:
                msg += "| <font color=#FF0000>{}</font>; | 0; | 0; | 0; |  \n".format(task)
                continue

            count = value.iloc[-1]["id"]
            last_mtime = os.path.getmtime(value.iloc[-1]["filepath"])

            if len(value) <= 10:
                speed = (last_mtime - os.path.getmtime(value.iloc[0]["filepath"])) / len(value)
            else:
                speed = (last_mtime - os.path.getmtime(value.iloc[-10]["filepath"])) / 10
            
            if speed > 180:
                color = "#FF0000"
            elif speed > 120:
                color = "#FF7F00"
            else:
                color = "#00FF00"
            
            msg += "| <font color={}>{}</font>; | {}; | {}; | {:.2f}; |  \n".format(color, task, ymdhms_time(last_mtime)[-8:], count, speed)

        return msg

    def init_info(self, key):
        self.info[key] = dict(
            counts = [],
            mtimes = [],
            utimes = [],
            itime = time.time()
        )

    def update_info(self, key, count, mtime = 0):
        self.info[key]["counts"].append(count)
        self.info[key]["mtimes"].append(mtime)
        self.info[key]["utimes"].append(time.time())
    
    def clean_info(self, key):
        if len(self.info[key]["counts"]) > 1000:
            self.info[key]["counts"] = self.info[key]["counts"][:500]
            self.info[key]["mtimes"] = self.info[key]["mtimes"][:500]
            self.info[key]["utimes"] = self.info[key]["utimes"][:500]
