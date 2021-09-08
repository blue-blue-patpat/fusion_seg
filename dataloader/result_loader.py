import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from dataloader.utils import file_paths_from_dir, filename_decoder
from sync.offsets import Offsets


class ResultLoader():
    def __init__(self, result_path) -> None:
        self.path = result_path
        self.params = {}
        self.file_dict = {}
        self.clfs = {}

    def run(self):
        for param in self.params:
            self.parse_files(os.path.join(self.path, param["tag"]), **param)
        return self.file_dict

    def generator(self, item_key="id"):
        for i in range(len(self)):
            yield self.select_item(i, item_key)

    def select_item(self, value, item_key, exact=True):
        res = {}
        for tag in self.file_dict.keys():            
            res[tag] = self.select_item_in_tag(value, item_key, tag, exact)
        return res

    def select_item_in_tag(self, value, item_key, tag, exact=True):
        df = self.file_dict[tag]
        res = None
        if exact:
            # auto convert type when comparing strs
            if not np.issubdtype(df[item_key].dtypes, np.number):
                value = str(value)
            # match equal
            res = dict(df[df[item_key]==value].iloc[0])
        else:
            # match closest
            clfs_key = str(tag) + str(item_key)
            # persistent clf
            if clfs_key not in self.clfs.keys():
                X = np.array([df[item_key]]).T.astype(np.float64)
                y = df.index
                # if np.isnan(X).any():
                #     np.nan_to_num(X, False)
                self.clfs[clfs_key] = KNeighborsClassifier(n_neighbors=1).fit(X, y)
            res = dict(df.iloc[self.clfs[clfs_key].predict(np.array([[value]], dtype=np.float64))[0]])
        return res

    def select_by_id(self, idx):
        return self.select_item(idx, item_key="id")

    def parse_files(self, path=None, ext='.png', tag="default"):
        if path is None:
            path = self.path
        file_paths = file_paths_from_dir(path, extension=ext)
        """
        self.file_dict
        |
        |-tag1=master/color: DataFrame([ id tm  filepath ]
        |                              [ 0  123 /home/xxx]
        |                              [ 1  125 ...      ])
        |-...
        |-tag2=sub1/depth:   DataFrame([ id tm  filepath ])
        |-...
        """
        self.file_dict[tag] = pd.DataFrame([filename_decoder(f) for f in file_paths])

    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError("Index key {} is out of range {}".format(key, len(self)))
        return self.select_by_id(key)

    def __len__(self):
        return min([len(v) for v in self.file_dict.values()])

    def __repr__(self) -> str:
        res = super().__repr__()
        for tag, df in self.file_dict.items():
            res += "\ntag={}\tlen={}".format(tag, len(df))
        res += "\n"
        return res

    def __str__(self) -> str:
        return self.__repr__()


class ArbeResultLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="arbe", ext=".npy"),
            ]
        else:
            self.params = params
        self.run()


class OptitrackCalibResultLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="calib/input", ext=".csv"),
            ]
        else:
            self.params = params
        self.run()


class OptitrackResultLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="optitrack", ext=".npz"),
            ]
        else:
            self.params = params
        self.run()


class RealSenseResultLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="realsense/color", ext=".png"),
                dict(tag="realsense/depth", ext=".png"),
                # dict(tag="realsense/vertex", ext=".npy"),
            ]
        else:
            self.params = params
        self.run()


class KinectResultLoader(ResultLoader):
    def __init__(self, result_path, params=None, device="master") -> None:
        super().__init__(result_path)
        self.device = device
        if params is None:
            self.params = [
                dict(tag="kinect/{}/color".format(device), ext=".png"),
                dict(tag="kinect/{}/depth".format(device), ext=".png"),
                dict(tag="kinect/{}/pcls".format(device), ext=".npy"),
                dict(tag="kinect/{}/skeleton".format(device), ext=".npy"),
            ]
        else:
            self.params = params
        self.run()

    def select_by_skid(self, skid):
        idx = -1
        try:
            idx = self.select_item_in_tag(skid, "skid", "kinect/{}/skeleton".format(self.device))["id"]
            return self.select_item(idx, "id")
        except Exception as e:
            print(idx, skid)
            raise e

    def generator(self, item_key="id"):
        print("[KinectResultLoader] WARNING: May meet empty skeleton. Try generator_by_skeleton instead.")
        return super().generator(item_key=item_key)

    def generator_by_skeleton(self):
        for i in range(len(self)):
            yield self.select_by_skid(i)


class KinectMKVtLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="kinect/master", ext=".mkv"),
                dict(tag="kinect/sub1", ext=".mkv"),
                dict(tag="kinect/sub2", ext=".mkv"),
            ]
        else:
            self.params = params
        self.run()


class KinectJsonLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="kinect/master", ext=".json"),
                dict(tag="kinect/sub1", ext=".json"),
                dict(tag="kinect/sub2", ext=".json"),
            ]
        else:
            self.params = params
        self.run()


class OptitrackCSVLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="optitrack", ext=".csv"),
            ]
        else:
            self.params = params
        self.run()


class ResultFileLoader():
    def __init__(self, root_path: str, skip_head: int=0, skip_tail: int=0,
                 enabled_sources: list=None, disabled_sources: list=[],
                 offsets: dict=None) -> None:
        """
        ResultFileLoader init

        :param root_path: root path
        :param skip_head: skip first $skip_head$ radar frames, default 0
        :param skip_tail: skip first $skip_tail$ radar frames, default 0
        :param enabled_sources: enabled result sources, default enable all sources
        :param disabled_sources: disabled result sources, default []
        :param offsets: if $root_path$/offsets.txt doesn't exist, init offsets with $offsets$
        :returns: None
        """
        self.root_path = root_path

        self.skip_head = skip_head
        self.skip_tail = skip_tail

        # Init file sources
        if enabled_sources is None:
            self.sources = ["arbe", "master", "sub1", "sub2", "kinect_skeleton", "optitrack", "mesh"]
        else:
            self.sources = enabled_sources

        # Remove disabled sources
        _sources = list(set(self.sources) - set(disabled_sources))
        _sources.sort(key=self.sources.index)
        self.sources = _sources

        # Init data
        self.init_arbe_source()
        self.init_kinect_source()
        self.init_optitrack_source()
        self.init_mesh_source()

        # Init calib
        self.init_calib_source()

        # Verify if offsets can be restored from file
        if Offsets.verify_file(root_path):
            # Init from file
            self.offsets = Offsets.from_file(root_path)
        elif isinstance(offsets, dict):
            # Init from arg
            self.offsets = Offsets(offsets)
        else:
            # Init empty offsets
            self.offsets = Offsets(dict(zip(self.sources, [0]*len(self.sources))), base=self.sources[0])

        self.fps = 30

        self.results = self.info = {}

    def init_arbe_source(self):
        """
        Init Arbe radar results
        """
        if "arbe" not in self.sources:
            print("[ResultFileLoader] Source 'arbe' is not enabled, may cause unexpected errors.")
            return

        self.a_loader = ArbeResultLoader(self.root_path)

        # Verify if params are illegal
        if len(self.a_loader) < (self.skip_head + self.skip_tail):
            raise IndexError("[ResultFileLoader] Skip count exceeds radar frame limit.")

    def init_kinect_source(self):
        """
        Init Azure Kinect results
        """
        self.k_mas_loader = KinectResultLoader(self.root_path, device="master") if "master" in self.sources else None
        self.k_sub1_loader = KinectResultLoader(self.root_path, device="sub1") if "sub1" in self.sources else None
        self.k_sub2_loader = KinectResultLoader(self.root_path, device="sub2") if "sub2" in self.sources else None

    def init_optitrack_source(self):
        """
        Init OptiTrack body tracker results
        """
        self.o_loader = OptitrackResultLoader(self.root_path) if "optitrack" in self.sources else None
            
    def init_mesh_source(self):
        if "mesh" in self.sources:
            # TODO: complete mesh init
            self.mesh_loader = ResultLoader()

    def init_calib_source(self):
        """
        Init transform matrix
        """
        from calib.utils import to_radar_transform_mat

        self.trans = to_radar_transform_mat(self.root_path)

    def select_kinect_trans_item_by_t(self, k_loader: KinectResultLoader, t: float) -> None:
        """
        Select Kinect transformed results by timestamp
        """
        if k_loader.device in self.offsets.keys():
            _t = t + self.offsets[k_loader.device] / self.fps
            res_skeleton = k_loader.select_item_in_tag(_t, "st", "kinect/{}/skeleton".format(k_loader.device), False)
            res = k_loader.select_by_skid(res_skeleton["skid"])

        elif "master" in self.offsets.keys():
            _t = t + self.offsets["master"] / self.fps
            if self.k_mas_loader is None:
                self.k_mas_loader = KinectResultLoader(self.root_path, device="master")
            mas_res_skeleton = self.k_mas_loader.select_item_in_tag(_t, "st", "kinect/master/skeleton", False)
            mas_res = self.k_mas_loader.select_by_skid(mas_res_skeleton["skid"])

            res = k_loader.select_item(mas_res["kinect/master/pcls"]["id"], "id", False)

        trans_mat = self.trans["kinect_{}".format(k_loader.device)]
        if "kinect_skeleton" in self.sources:
            self.results.update({
                "{}_skeleton".format(k_loader.device): np.load(res["kinect/{}/skeleton".format(k_loader.device)]["filepath"])[:,:,:3].reshape(-1,3) / 1000 @ trans_mat["R"].T + trans_mat["t"]
            })
            self.info.update({
                k_loader.device: res["kinect/{}/skeleton".format(k_loader.device)]
            })
        if "kinect_pcl" in self.sources:
            pcl = np.load(res["kinect/{}/pcls".format(k_loader.device)]["filepath"]).reshape(-1, 3)
            pcl = pcl[pcl.any(axis=1)]
            self.results.update({
                "{}_pcl".format(k_loader.device):  pcl / 1000 @ trans_mat["R"].T + trans_mat["t"],
            })
            self.info.update({
                "{}_pcl".format(k_loader.device): res["kinect/{}/pcls".format(k_loader.device)]
            })

    def select_trans_optitrack_item_by_t(self, o_loader: OptitrackResultLoader, t: float) -> None:
        """
        Select OptiTrack transformed results by timestamp
        """
        _t = t + self.offsets["optitrack"] / self.fps
        res = o_loader.select_item(_t, "st", False)
        arr = np.load(res["optitrack"]["filepath"])
        self.results.update(dict(
            optitrack=arr["markers"][:,:,:3].reshape(-1,3) @ self.trans["optitrack"]["R"].T + self.trans["optitrack"]["t"],
            optitrack_person_count=arr["markers"].shape[0]
        ))
        self.info.update(dict(
            optitrack = res["optitrack"]
        ))

    def __getitem__(self, index: int) -> tuple:
        """
        Returns the $index$^th radar frame with its synchronized frames of other devices
        """
        i = index + self.skip_head
        if index not in range(0, self.__len__()):
            raise IndexError("[ResultFileLoader] Index out of range {}.".format(range(0, self.__len__())))
        self.results = self.info = {}
        arbe_res = self.a_loader[i]

        if "arbe" in self.sources:
            self.results = dict(
                arbe=np.load(arbe_res["arbe"]["filepath"])[:,:3]
            )
        self.info = dict(
            arbe=arbe_res["arbe"]
        )

        t = float(arbe_res["arbe"]["st"])
        if self.k_mas_loader:
            self.select_kinect_trans_item_by_t(self.k_mas_loader, t)
        if self.k_sub1_loader:
            self.select_kinect_trans_item_by_t(self.k_sub1_loader, t)
        if self.k_sub2_loader:
            self.select_kinect_trans_item_by_t(self.k_sub2_loader, t)
        if self.o_loader:
            self.select_trans_optitrack_item_by_t(self.o_loader, t)
        return self.results, self.info

    def __len__(self):
        return len(self.a_loader) - self.skip_head - self.skip_tail

    def __repr__(self):
        res = super().__repr__()
        res += "\n----\noffsets="
        res += str(self.offsets)
        res += "\n----\n"

        if "arbe" in self.sources:
            res += str(self.a_loader)
            res += "----\n"

        if "master" in self.sources:
            res += str(self.k_mas_loader)
            res += "----\n"

        if "sub1" in self.sources:
            res += str(self.k_sub1_loader)
            res += "----\n"

        if "sub2" in self.sources:
            res += str(self.k_sub2_loader)
            res += "----\n"

        if "optitrack" in self.sources:
            res += str(self.o_loader)
            res += "----\n"

        if "mesh" in self.sources:
            res += str(self.mesh_loader)
            res += "----\n"

        return res

    def __str__(self) -> str:
        return self.__repr__()
