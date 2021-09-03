import numpy as np

from dataloader.result_loader import KinectResultLoader


def kinect_signal(loader: KinectResultLoader, start: int=0, end: int=None, R=None, t=None):
    if start < 0:
        start = 0
    if end is None or end > len(loader):
        end = len(loader)
    
    _x = []
    _t = []
    for i in range(start, end):
        item = loader.select_by_skid(i)
        skeleton = np.load(item["kinect/{}/skeleton".format(loader.device)]["filepath"])[0][:,:3]/1000 @ R.T + t
        _x.append(skeleton[0,1])
        _t.append(float(item["kinect/{}/skeleton".format(loader.device)]["dt"]))
    return _x, _t
