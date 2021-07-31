import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from dataloader.utils import file_paths_from_dir, filename_decoder


class ResultLoader():
    def __init__(self, result_path) -> None:
        self.path = result_path
        self.file_dict = {}
        self.clfs = {}

    def run(self):
        raise Warning("[ResultLoader] This method should be overriden.")

    def gen_item(self):
        for i in range(len(self)):
            yield self.select_by_id(i)

    def select_item(self, value, tag, exact=True):
        res = {}
        for k, df in self.file_dict.items():            
            if exact:
                # auto convert type when comparing strs
                if not np.issubdtype(df[tag].dtypes, np.number):
                    value = str(value)
                # match equal
                res[k] = df[df[tag]==value]
            else:
                # match closest
                # persistent clf
                if tag not in self.clfs.keys():
                    X = np.array([df[tag]]).T.astype(np.float64)
                    y = df.index
                    self.clfs[tag] = KNeighborsClassifier(n_neighbors=1).fit(X, y)
                res[k] = df[self.clfs[tag].predict([value])[0]]
        return res

    def select_by_id(self, idx):
        return self.select_item(idx, tag="id")

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
        |-tag2=sub1/depth:   DataFrame [ id tm filepath]
        |-...
        """
        self.file_dict[tag] = pd.DataFrame([filename_decoder(f) for f in file_paths])

    def __getitem__(self, key):
        if key >= len(self):
            raise IndexError("Index key {} is out of range {}".format(key, len(self)))
        return self.select_by_id(key)

    def __len__(self):
        return min([len(v) for v in self.file_dict.values()])


class KinectResultLoader(ResultLoader):
    def __init__(self, result_path, params=None) -> None:
        super().__init__(os.path.join(result_path, "kinect"))
        if params is None:
            self.params = [
                dict(tag="master/color", ext=".png"),
                dict(tag="master/depth", ext=".png"),
                dict(tag="master/skeleton", ext=".npy"),
            ]
        else:
            self.params = params
        self.run()

    def run(self):
        for param in self.params:
            self.parse_files(os.path.join(self.path, param["tag"]), **param)
        return self.file_dict
