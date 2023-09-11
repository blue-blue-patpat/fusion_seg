import os
import cv2
import numpy as np
import pandas as pd
import pickle
import open3d as o3d
from sklearn.neighbors import KNeighborsClassifier
from dataloader.utils import file_paths_from_dir, filename_decoder
from sync.offsets import Offsets, CalibOffsets


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
        if df.empty:
            return {}
        if exact:
            # auto convert type when comparing strs
            if not np.issubdtype(df[item_key].dtypes, np.number):
                value = str(value)
            # match equal
            res_df = df[df[item_key]==value]
            if res_df.empty:
                res = {}
            else:
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

    def load_calibration(self):
        import json
        calib_path = os.path.join(self.path, "kinect/{}/calibration_raw.json".format(self.device))
        if not os.path.exists(calib_path):
            print("[KinectResultLoader] WARNING: Camera Clibration file not found.")
            self.R = np.eye(3)
            self.t = np.zeros(3)
            return
        with open(calib_path, "r") as f:
            d = json.load(f)
        cameras = d["CalibrationInformation"]["Cameras"]
        for camera in cameras:
            if camera["Location"] == "CALIBRATION_CameraLocationPV0":
                self.R = np.array(camera["Rt"]["Rotation"]).reshape((3,3))
                self.t = np.array(camera["Rt"]["Translation"]) * 1000
                break

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
        self.mkvs = dict([(k, v.loc[0,"filepath"]) for k, v in self.file_dict.items()])


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


class MinimalLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(result_path)
        if params is None:
            self.params = [
                dict(tag="minimal/obj", ext=".obj"),
                dict(tag="minimal/param", ext=".npz"),
                # dict(tag="minimal/trans", ext=".npz"),
            ]
        else:
            self.params = params
        self.run()


class PKLLoader(ResultLoader):
    def __init__(self, result_path, skip_head=0, skip_tail=0, enable_sources=None) -> None:
        super().__init__(result_path)
        self.params = [dict(tag="pkl", ext=".pkl")]
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        self.run()

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index key {} is out of range {}".format(idx, len(self)))
        res = self.select_by_id(idx + self.skip_head)
        with open(res['pkl']['filepath'], 'rb') as f:
            frame = pickle.load(f)
        return frame

    def __len__(self):
        return min([len(v) for v in self.file_dict.values()]) - self.skip_head - self.skip_tail


class SequenceLoader(object):
    def __init__(self, seq_path:str, skip_head=0, skip_tail=0, enabled_sources=None) -> None:
        self.seq_path = seq_path
        self.skip_head = skip_head
        self.skip_tail = skip_tail
        if enabled_sources is None:
            self.sources = ["radar", "image", "mesh"]
        else:
            self.sources = enabled_sources
        # load transformation matrix
        with open(os.path.join(seq_path, 'calib.txt')) as f:
            self.calib = eval(f.readline())

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, 'radar'))) - self.skip_head - self.skip_tail

    def __getitem__(self, idx:int):
        result = {}
        if 'radar' in self.sources:
            result['radar'] = np.load(os.path.join(self.seq_path, 'radar', 'frame_{}.npy'.format(idx + self.skip_head)))
        if 'image' in self.sources:
            result['image'] = cv2.imread(os.path.join(self.seq_path, 'image', 'frame_{}.png'.format(idx + self.skip_head)))
        if 'depth' in self.sources:
            result['depth'] = cv2.imread(os.path.join(self.seq_path, 'depth', 'frame_{}.png'.format(idx + self.skip_head)))
        if 'mesh' in self.sources:
            result['mesh'] = np.load(os.path.join(self.seq_path, 'mesh', 'frame_{}.npz'.format(idx + self.skip_head)))
        return result


class ResultFileLoader():
    """
    Load files from sources

    Sources:
        arbe:
            directory: auto load
            arbe_pcl file <N*3 ndarray>: "arbe" or "arbe_pcl"
            arbe_feature <N*5? ndarray>: "arbe_feature"
        kinect:
            directory: "master", "sub1", "sub2". if "kinect_skeleton" exists, load files by skid.
            $device$_skeleton <person_count*K*3 ndarray>: "kinect_skeleton"
            $device$_pcl <N*3 ndarray>: "kinect_pcl"; if "kinect_pcl_remove_zeros" enabled, return valid points.
        optitrack:
            directory: "optitrack"
            optitrack <person_count*K*3 ndarray>: "optitrack"
        mesh:
            directory: "mesh"
            mesh_param {pose: <10, ndarray>, shape: <72, ndarray>, vertices: <2,100~*3 ndarray>, keypoints: <23*3 ndarray>}: "mesh_param"
            mesh_obj (verts, faces): "mesh_obj"
    """
    def __init__(self, root_path: str, skip_head: int=0, skip_tail: int=0,
                 enabled_sources: list=None, disabled_sources: list=[],
                 offsets: dict=None, select_key: str="arbe") -> None:
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

        self.select_key = select_key

        # Init file sources
        if enabled_sources is None:
            self.sources = ["arbe","arbe_feature",'master','sub2',"kinect_color",
                            "kinect_depth","kinect_pcl","mesh","mosh","mesh_param"]
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
        self.init_info()

        # Init calib
        self.init_calib_source()

        # Verify if offsets can be restored from file
        if Offsets.verify_file(root_path):
            # Init from file
            self.sync_offsets = Offsets.from_file(root_path)
        elif isinstance(offsets, dict):
            # Init from arg
            self.sync_offsets = Offsets(offsets)
        else:
            # Init empty offsets
            self.sync_offsets = Offsets(dict(zip(self.sources, [0]*len(self.sources))), base=self.sources[0])

        # rectify calib
        if CalibOffsets.verify_file(root_path):
            # Init from file
            self.calib_offsets = CalibOffsets.from_file(root_path)
        else:
            self.calib_offsets = CalibOffsets()
            for k, v in self.trans.items():
                self.calib_offsets.update({k:{'R':v['R'].tolist(), 't':v['t'].tolist()}})
        
        self.rectify_calibration()
            
        # triggered by source reindex
        self.init_reindex()

        self.after_init_hook()

        self.fps = 30

        self.results = self.info = {}

    def init_arbe_source(self):
        """
        Init Arbe radar results
        """
        if "arbe" not in self.sources:
            self.a_loader = None
            print("[ResultFileLoader] Source 'arbe' is not enabled, may cause unexpected errors.")
        else:
            self.a_loader = ArbeResultLoader(self.root_path)

            # Verify if params are illegal
            if len(self.a_loader) < (self.skip_head + self.skip_tail):
                raise IndexError("[ResultFileLoader] Skip count exceeds radar frame limit.")

    def init_kinect_source(self):
        """
        Init Azure Kinect results
        """
        self.kinect_params_template = []
        self.kinect_devices = ["master", "sub1", "sub2"]
        self.enabled_kinect_devices = [device for device in self.kinect_devices if device in self.sources]

        if "kinect_pcl" in self.sources:
            self.kinect_params_template.append(dict(tag="kinect/{}/pcls", ext=".npy"))

        if "kinect_skeleton" in self.sources:
            self.kinect_params_template.append(dict(tag="kinect/{}/skeleton", ext=".npy"))

        if "kinect_color" in self.sources:
            self.kinect_params_template.append(dict(tag="kinect/{}/color", ext=".png"))

        if "kinect_depth" in self.sources:
            self.kinect_params_template.append(dict(tag="kinect/{}/depth", ext=".png"))

        for device in self.kinect_devices:
            self.__dict__["k_{}_loader".format(device)] = None

        for device in self.enabled_kinect_devices:
            self.__dict__["k_{}_loader".format(device)] = KinectResultLoader(self.root_path, device=device, params=[
                dict(tag=item["tag"].format(device), ext=item["ext"]) for item in self.kinect_params_template
            ])

            if "kinect_skeleton" in self.sources:
                self.__dict__["k_{}_loader".format(device)].load_calibration()

    def init_optitrack_source(self):
        """
        Init OptiTrack body tracker results
        """
        self.o_loader = OptitrackResultLoader(self.root_path) if "optitrack" in self.sources else None
            
    def init_mesh_source(self):
        """
        Init Mesh
        """
        if "mesh" in self.sources:
            mesh_type = "mosh" if "mosh" in self.sources else "minimal"
            params = []
            if "mesh_param" in self.sources:
                params.append(dict(tag="{}/param".format(mesh_type), ext=".npz"))
            if "mesh_obj" in self.sources:
                params.append(dict(tag="{}/obj".format(mesh_type), ext=".obj"))
            self.mesh_loader = MinimalLoader(self.root_path, params=params) 
        else:
            self.mesh_loader = None

    def init_info(self):
        import json
        self.info_dict = {}
        if os.path.exists(os.path.join(self.root_path, "infomation.json")):
            with open(os.path.join(self.root_path, "infomation.json"), "r") as f:
                record_info = json.load(f)
            for k,v in record_info.items():
                self.info_dict[k.strip()] = v
        else:
            print("[ResultFileLoader] Information not found.")

    def init_calib_source(self):
        """
        Init transform matrix
        """
        from calib.utils import to_radar_transform_mat

        self.trans = to_radar_transform_mat(self.root_path)

    def init_skip(self):
        if "mesh" in self.sources and "mosh" not in self.sources:
            rid_arr = np.array(self.mesh_loader.file_dict["minimal/param"]["rid"], dtype=int)
            self.skip_head = max(self.skip_head, rid_arr.min())
            self.skip_tail = max(self.skip_tail, len(self.a_loader) - 1 - rid_arr.max())

    def init_reindex(self):
        if "reindex" not in self.sources:
            return

        assert isinstance(self.a_loader.file_dict["arbe"], pd.DataFrame)

        # copy and sort by dt
        sorted_a_df = self.a_loader.file_dict["arbe"].sort_values("dt", ascending=True, inplace=False)

        # remove skipped frames
        if self.skip_head > 0:
            sorted_a_df.drop(range(self.skip_head), axis=0, inplace=True)
        
        if self.skip_tail > 0:
            sorted_a_df.drop(range(len(sorted_a_df) - self.skip_tail, len(sorted_a_df)), axis=0, inplace=True)

        # remove duplicate frames
        sorted_a_df.drop_duplicates(subset=["dt"],keep="first", inplace=True)

        # re-index id
        # sorted_a_df.drop("id", axis=1, inplace=True)
        sorted_a_df.insert(0, "reindexed_id", range(len(sorted_a_df)))

        self.a_loader.file_dict["arbe"] = sorted_a_df
        self.a_loader.clfs = {}

        self.skip_head = self.skip_tail = 0

    def after_init_hook(self):
        if self.select_key == "arbe":
            if self.a_loader is None:
                self.a_loader = ArbeResultLoader(self.root_path)
            self.init_skip()

        if self.select_key == "mesh":
            if "masid" in list(self.mesh_loader.file_dict.values())[0].columns:
                self.mesh_kinect_key_device = "master"
            elif "sub1id" in list(self.mesh_loader.file_dict.values())[0].columns:
                self.mesh_kinect_key_device = "sub1"
            elif "sub2id" in list(self.mesh_loader.file_dict.values())[0].columns:
                self.mesh_kinect_key_device = "sub2"
            if self.__dict__["k_{}_loader".format(self.mesh_kinect_key_device)] is None:
                self.__dict__["k_{}_loader".format(self.mesh_kinect_key_device)] =\
                KinectResultLoader(self.root_path, device=self.mesh_kinect_key_device, params=[
                    dict(tag=item["tag"].format(self.mesh_kinect_key_device), ext=item["ext"])
                        for item in self.kinect_params_template
                ])
        
        if len(self.enabled_kinect_devices) > 0:
            for device in self.kinect_devices:
                if self.__dict__["k_{}_loader".format(device)] is None and device in self.sync_offsets.keys():
                    self.__dict__["k_{}_loader".format(device)] = KinectResultLoader(self.root_path, device=device)

    def rectify_calibration(self):
        for k, v in self.calib_offsets.items():
            if k in self.trans.keys():
                self.trans[k] = {'R':np.array(v['R']), 't':np.array(v['t'])}
    
    def select_radar_item_by_id(self, r_loader: ArbeResultLoader, idx: int) -> dict:
        if "reindex" in self.sources:
            arbe_res = r_loader.select_item(idx, "reindexed_id")
        else:
            arbe_res = r_loader[idx]

        arbe_arr = None
        if "arbe_pcl" in self.sources or "arbe" in self.sources:
            arbe_arr = np.load(arbe_res["arbe"]["filepath"])
            self.results.update(dict(
                arbe=arbe_arr[:,:3]
            ))
        if "arbe_feature" in self.sources:
            if arbe_arr is None:
                arbe_arr = np.load(arbe_res["arbe"]["filepath"])
            self.results.update(dict(
                arbe_feature=arbe_arr[:,3:]
            ))
        self.info.update(dict(
            arbe=arbe_res["arbe"]
        ))
        return arbe_res

    def select_kinect_trans_item_by_t(self, k_loader: KinectResultLoader, t: float) -> None:
        """
        Select Kinect transformed results by timestamp
        """
        if k_loader.device in self.sync_offsets.keys():
            _t = t + self.sync_offsets[k_loader.device] / self.fps
            # Use index: skid
            if "kinect_skeleton" in self.sources:
                res_skeleton = k_loader.select_item_in_tag(_t, "st", "kinect/{}/skeleton".format(k_loader.device), False)
                res = k_loader.select_by_skid(res_skeleton["skid"])
            else:
                res = k_loader.select_item(_t, "st", False)

        else:
            for device in self.kinect_devices:
                if device in self.sync_offsets.keys():
                    _loader = self.__dict__["k_{}_loader".format(device)]
                    _t = t + self.sync_offsets[device] / self.fps
                    # Init loader if None
                    if _loader is None:
                        _loader = self.__dict__["k_{}_loader".format(device)] = KinectResultLoader(self.root_path, device=device)
                    # Use index: skid
                    if "kinect_skeleton" in self.sources:
                        device_res_skeleton = _loader.select_item_in_tag(_t, "st", "kinect/{}/skeleton".format(device), False)
                        device_res = _loader.select_by_skid(device_res_skeleton["skid"])
                        res = k_loader.select_item(device_res["kinect/{}/pcls".format(device)]["id"], "id", False)
                    else:
                        res = k_loader.select_item(_t, "st", False)

        trans_mat = self.trans["kinect_{}".format(k_loader.device)]
        if "kinect_skeleton" in self.sources:
            self.results.update({
                "{}_skeleton".format(k_loader.device): (np.load(
                    res["kinect/{}/skeleton".format(k_loader.device)]["filepath"]
                    )[:,:,:3].reshape(-1,3) @ k_loader.R.T + k_loader.t)
                    / 1000 @ trans_mat["R"].T + trans_mat["t"]
            })
            self.info.update({
                "{}_skeleton".format(k_loader.device): res["kinect/{}/skeleton".format(k_loader.device)]
            })
        if "kinect_pcl" in self.sources:
            pcl = np.load(res["kinect/{}/pcls".format(k_loader.device)]["filepath"]).reshape(-1, 3)
            pcl_empty_filter = pcl.any(axis=1)
            if "kinect_pcl_remove_zeros" in self.sources:
                pcl = pcl[pcl_empty_filter] / 1000 @ trans_mat["R"].T + trans_mat["t"]
            else:
                pcl = pcl / 1000 @ trans_mat["R"].T + trans_mat["t"]
                pcl[np.logical_not(pcl_empty_filter)] = 0
            self.results.update({
                "{}_pcl".format(k_loader.device):  pcl,
            })
            self.info.update({
                "{}_pcl".format(k_loader.device): res["kinect/{}/pcls".format(k_loader.device)]
            })
        if "kinect_color" in self.sources:
            self.results.update({
                "{}_color".format(k_loader.device): cv2.imread(res["kinect/{}/color".format(k_loader.device)]["filepath"]),
            })
            
        if "kinect_depth" in self.sources:
            self.results.update({
                "{}_depth".format(k_loader.device): cv2.imread(res["kinect/{}/depth".format(k_loader.device)]["filepath"]),
            })

    def select_trans_optitrack_item_by_t(self, o_loader: OptitrackResultLoader, t: float) -> None:
        """
        Select OptiTrack transformed results by timestamp
        """
        _t = t + self.sync_offsets["optitrack"] / self.fps
        res = o_loader.select_item(_t, "st", False)
        arr = np.load(res["optitrack"]["filepath"])
        self.results.update(dict(
            optitrack=arr["markers"][:,:,:3].reshape(-1,3) @ self.trans["optitrack"]["R"].T + self.trans["optitrack"]["t"],
            optitrack_person_count=arr["markers"].shape[0]
        ))
        self.info.update(dict(
            optitrack = res["optitrack"]
        ))

    def select_mesh_item(self, mesh_loader: MinimalLoader, idx: int, key: str = "rid") -> dict:
        """
        Select Mesh transformed results by radar id
        """
        mesh_res = mesh_loader.select_item(idx, key, False)
        if len(mesh_res["minimal/param"]) == 0:
            self.results.update(dict(mesh_param=None, mesh_obj=None))
            self.info.update(dict(mesh=None))
       
        if "mesh_param" in self.sources:
            try:
                self.results.update(dict(
                    mesh_param=np.load(mesh_res["minimal/param"]["filepath"]),
                ))
                self.info.update(dict(
                    mesh_param=mesh_res["minimal/param"]
                ))
            except Exception as e:
                self.results.update(dict(
                    mesh_param=None,
                ))
        if "mesh_obj" in self.sources:
            mesh = o3d.io.read_triangle_mesh(mesh_res["minimal/obj"]["filepath"])
            verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            self.results.update(dict(
                mesh_obj=(verts, faces),
            ))
            self.info.update(dict(
                mesh_obj=mesh_res["minimal/obj"]
            ))
        return mesh_res

    def select_mosh_item(self, mesh_loader: MinimalLoader, idx: int, key: str = "id") -> dict:
        from mosh.utils import mosh_pose_transform
        """
        Select Mesh transformed results by radar id
        """
        mesh_res = mesh_loader.select_item(idx, key, False)
        if len(list(mesh_res.values())) == 0:
            self.results.update(dict(mesh_param=None, mesh_obj=None))
            self.info.update(dict(mesh=None))
       
        if "mesh_param" in self.sources:
            try:
                # add offset to translation
                mosh_params = dict(np.load(mesh_res["mosh/param"]["filepath"]))
                # optitrack to radar coordinate
                trans, root_orient = mosh_pose_transform(mosh_params['pose'][:3], mosh_params['pose'][3:6], mosh_params['joints'][0], self.trans["optitrack"])
                pose = np.hstack((trans, root_orient.reshape(-1), mosh_params['pose'][6:]))
                mosh_params['pose'] = pose
                mosh_params['vertices'] = mosh_params['vertices'] @ self.trans["optitrack"]["R"].T + self.trans["optitrack"]["t"]
                mosh_params['joints'] = mosh_params['joints'] @ self.trans["optitrack"]["R"].T + self.trans["optitrack"]["t"]

                self.results.update(dict(
                    mesh_param=mosh_params,
                ))
                self.info.update(dict(
                    mesh_param=mesh_res["mosh/param"]
                ))
            except Exception as e:
                print(e)
                self.results.update(dict(
                    mesh_param=None,
                ))
        if "mesh_obj" in self.sources:
            mesh = o3d.io.read_triangle_mesh(mesh_res["mosh/obj"]["filepath"])
            verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            verts = verts @ self.trans["optitrack"]["R"].T + self.trans["optitrack"]["t"]
            self.results.update(dict(
                mesh_obj=(verts, faces),
            ))
            self.info.update(dict(
                mesh_obj=mesh_res["mosh/obj"]
            ))
        return mesh_res

    def select_mosh_item_by_t(self, mesh_loader: MinimalLoader, t: float) -> None:
        """
        Select OptiTrack transformed results by timestamp
        """
        _t = t + self.sync_offsets.get('mosh', self.sync_offsets['optitrack']) / self.fps
        res = mesh_loader.select_item(_t, "st", False)
        return self.select_mosh_item(mesh_loader, list(res.values())[0]["id"])

    def select_by_radar(self, index: int) -> tuple:
        """
        Returns the $index$^th radar frame with its synchronized frames of other devices
        """
        arbe_res =  self.select_radar_item_by_id(self.a_loader, index)

        t = float(arbe_res["arbe"]["st"])
        rid = int(arbe_res["arbe"]["id"])
        
        for device in self.kinect_devices:
            if self.__dict__["k_{}_loader".format(device)] is not None and device in self.sources:
                self.select_kinect_trans_item_by_t(self.__dict__["k_{}_loader".format(device)], t)

        if self.o_loader is not None:
            self.select_trans_optitrack_item_by_t(self.o_loader, t)
        if self.mesh_loader is not None:
            if "mosh" not in self.sources:
                self.select_mesh_item(self.mesh_loader, rid, "rid")
            else:
                self.select_mosh_item_by_t(self.mesh_loader, t)

        self.results["information"] = self.info_dict

    def select_by_mesh(self, index: int) -> tuple:
        mesh_res = self.select_mesh_item(self.mesh_loader, index, "id")

        if "masid" in mesh_res["minimal/param"] and not np.isnan(float(mesh_res["minimal/param"]["masid"])):
            kinect_key_device_id = int(mesh_res["minimal/param"]["masid"])
            self.mesh_kinect_key_device = "master"
        elif "sub1id" in mesh_res["minimal/param"] and not np.isnan(float(mesh_res["minimal/param"]["sub1id"])):
            kinect_key_device_id = int(mesh_res["minimal/param"]["sub1id"])
            self.mesh_kinect_key_device = "sub1"
        elif "sub2id" in mesh_res["minimal/param"] and not np.isnan(float(mesh_res["minimal/param"]["sub2id"])):
            kinect_key_device_id = int(mesh_res["minimal/param"]["sub2id"])
            self.mesh_kinect_key_device = "sub2"
        # Init device loader if necessary
        if self.__dict__["k_{}_loader".format(self.mesh_kinect_key_device)] is None:
            self.after_init_hook()

        # Use key device
        key_res = self.__dict__["k_{}_loader".format(self.mesh_kinect_key_device)].select_item(kinect_key_device_id, "id", False)
        key_t = float(list(key_res.values())[0]["st"])

        for device in self.enabled_kinect_devices:
            self.select_kinect_trans_item_by_t(self.__dict__["k_{}_loader".format(device)], key_t)
        self.results["information"] = self.info_dict

    def __getitem__(self, index: int) -> tuple:
        self.results = {}
        self.info = {}

        i = index + self.skip_head
        if index not in range(0, self.__len__()):
            raise IndexError("[ResultFileLoader] {} Index {} out of range {}.".format(self.root_path, index, self.__len__()))

        if self.select_key == "arbe":
            self.select_by_radar(i)
        elif self.select_key == "mesh":
            self.select_by_mesh(i)

        return self.results, self.info

    def __len__(self):
        if self.select_key == "arbe":
            return len(self.a_loader) - self.skip_head - self.skip_tail
        elif self.select_key == "mesh":
            return len(self.mesh_loader) - self.skip_head - self.skip_tail
        else:
            raise NotImplementedError("[ResultFileLoader] Unknow select_key type")

    def __repr__(self):
        res = super().__repr__()
        res += "\n----\noffsets="
        res += str(self.sync_offsets)
        res += "\n----\n"

        if "arbe" in self.sources:
            res += str(self.a_loader)
            res += "----\n"

        if "master" in self.sources:
            res += str(self.k_master_loader)
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
