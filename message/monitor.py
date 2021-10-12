import os
import time
from dataloader.result_loader import ResultLoader
from dataloader.utils import file_paths_from_dir, ymdhms_time
from message.dingtalk import TimerBot


class FileMonitor():
    def __init__(self, root_path, dir_key, interval: int = 600) -> None:
        self.root_path = root_path
        self.dir_key = dir_key
        self.interval = interval

        self.files = ResultLoader(root_path)
        self.files.params = self.update_dir()
        self.files.run()

        self.bot = TimerBot(60)
        self.bot.enable()

        self.info = {}

    def run(self):
        while True:
            msg = self.check_files()
            self.bot.print(msg)
            time.sleep(self.interval)

    def update_dir(self):
        dir_list = file_paths_from_dir(self.root_path, '!', enable_print=True, collect_dir=True)
        params = [
            dict(tag=d, ext="*") for d in dir_list if self.dir_key in d
        ]
        return params

    def check_files(self):
        msg = "{} : [FileMonitor]\n".format(ymdhms_time())
        
        for key, value in self.files.file_dict.items():
            if key not in self.info:
                self.init_info(key)
            if len(value) == 0:
                self.update_info(key, 0)
                continue
            res = self.files.select_item_in_tag(len(value)-1, "id", key)
            self.update_info(key, len(value), os.path.getmtime(res["filepath"]))

        for key, info in self.info.items():
            self.clean_info(key)

            task = key.split('/')[-3]

            if len(info["counts"]) == 0:
                msg += "{} is empty for {:.2f} minutes.\n".format(task, (info["utimes"][-1] - info["itime"]) / 60)
                continue

            if len(info["counts"]) < 5:
                speed = -1
            else:
                count = (info["counts"][-1] - info["counts"][-5])
                if count != 0:
                    speed = (info["utimes"][-1] - info["utimes"][-5])
                else:
                    speed = -1
            msg += "{} last modify at {}, total {}, {:.2f} s/frame.\n".format(task, ymdhms_time(info["mtimes"][-1]), info["counts"][-1], speed)
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
