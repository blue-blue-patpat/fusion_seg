import time
import numpy as np
from multiprocessing.dummy import Pool

from dataloader.result_loader import ResultFileLoader
from dataloader.utils import ymdhms_time
from minimal.bridge import JointsBridge


OPTI_DATA = "optitrack"
KINECT_DATA = "kinect"

OPTI_SOURCE = OPTI_DATA
KINECT_SOURCES = ["sub1_skeleton", "sub2_skeleton", "master_skeleton"]
KINECT_SUB1_SOURCE = 0
KINECT_SUB2_SOURCE = 1
KINECT_MAS_SOURCE = 2
KINECT_SUB_MEAN_SOURCE = "sub1_sub2_mean_skeleton"


class MinimalInput:
    def __init__(self, loader: ResultFileLoader, scale: float, data_type: str) -> None:
        self.input_dict = {}
        self.loader = loader
        self.scale = scale
        self.data_type = data_type
        if data_type == OPTI_DATA:
            self.jnts_source = OPTI_SOURCE
        elif data_type == KINECT_DATA:
            self.jnts_source = KINECT_SUB_MEAN_SOURCE
        self.pool = Pool(5)

    def update(self, idx: int, pre_update: int=0, force_update: bool=False):
        update(self, idx, force_update)

        for i in range(1, pre_update+1):
            self.pool.apply_async(update, (self, idx+i))

    def remove(self, idx: int):
        if idx in self.input_dict.keys():
            del self.input_dict[idx]

    def save_revert_transform(self, idx: int, file_path: str):
        np.savez(file_path, **self.input_dict[idx]["transform"])

    def __getitem__(self, idx):
        self.update(idx, 2)
        wait_count = 0
        while idx not in self.input_dict.keys() or self.input_dict[idx].get("lock", True):
            time.sleep(1)
            print("{} : [MinimalInput] Waiting for sub-process to collect data...".format(ymdhms_time()))
            wait_count += 1
            # Retry every 30 seconds
            if wait_count % 30 == 0:
                print("{} : [MinimalInput] Retrying...".format(ymdhms_time()))
                self.update(idx, force_update=True)
            # Give up after 2 retries
            if wait_count > 100:
                raise RuntimeError("[MinimalInput] Waiting for too long and exit.")
        return self.input_dict[idx]


def update(self: MinimalInput, idx: int, force_update: bool = False):
    if idx in self.input_dict.keys() and not force_update:
        return
    if idx >= len(self.loader):
        return
    self.input_dict[idx] = dict(
        lock=True
    )
    result, info = self.loader[idx]
    brg = JointsBridge()
    # brg.set_scale(self.scale)
    if self.jnts_source == OPTI_SOURCE:
        raw_jnts = result[OPTI_SOURCE]
    elif self.jnts_source == KINECT_SUB_MEAN_SOURCE:
        raw_jnts = np.mean([result[KINECT_SOURCES[KINECT_SUB1_SOURCE]], result[KINECT_SOURCES[KINECT_SUB2_SOURCE]]], axis=0)
    else:
        raw_jnts = result[KINECT_SOURCES[self.jnts_source]]
    brg.init_input(raw_jnts, np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
    _jnts, _pcl = brg.map(self.data_type)
    # R, t, scale = brg.revert_transform()
    self.input_dict[idx] = dict(
        jnts=_jnts,
        pcl=_pcl,
        info=info,
        lock=False
        # transform=dict(
        #     R=R, t=t, scale=scale
        # )
    )
    if np.isnan(raw_jnts).sum() > 0:
        self.input_dict[idx]["info"]["nan"] = True
    print("{} : [MinimalInput] Frame {} successfully loaded as {} type.".format(ymdhms_time(), idx, self.jnts_source))
