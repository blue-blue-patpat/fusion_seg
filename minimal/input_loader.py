import numpy as np
from multiprocessing.dummy import Pool

from dataloader.result_loader import ResultFileLoader
from dataloader.utils import ymdhms_time
from minimal.bridge import JointsBridge

 
class MinimalInput:
    def __init__(self, loader: ResultFileLoader, scale: float, data_type: str) -> None:
        self.input_dict = {}
        self.loader = loader
        self.scale = scale
        self.data_type = data_type
        self.pool = Pool(5)

    def update(self, idx: int, pre_update: int=0):
        update(self, idx)

        for i in range(1, pre_update+1):
            self.pool.apply_async(update, (self, idx+i))

    def remove(self, idx: int):
        if idx in self.input_dict.keys():
            del self.input_dict[idx]

    def save_revert_transform(self, idx: int, file_path: str):
        np.savez(file_path, **self.input_dict[idx]["transform"])

    def __getitem__(self, idx):
        self.update(idx, 2)
        return self.input_dict[idx]


def update(self: MinimalInput, idx: int):
    if idx in self.input_dict.keys() or idx >= len(self.loader):
        return
    self.input_dict[idx] = {}
    result, info = self.loader[idx]
    brg = JointsBridge()
    brg.set_scale(self.scale)
    brg.init_input(result["optitrack"], np.vstack([result["master_pcl"], result["sub1_pcl"], result["sub2_pcl"]]))
    _jnts, _pcl = brg.map(self.data_type)
    R, t, scale = brg.revert_transform()
    self.input_dict[idx] = dict(
        jnts=_jnts,
        pcl=_pcl,
        info=info,
        transform=dict(
            R=R, t=t, scale=scale
        )
    )
    if np.isnan(result["optitrack"]).sum() > 0:
        self.input_dict[idx]["info"]["nan"] = True
    print("{} : [MinimalInput] Frame {} successfully loaded.".format(ymdhms_time(), idx))
