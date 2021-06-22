#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File       :   data_align.py
@Contact    :   wyzlshx@foxmail.com

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
2021/6/6 11:15    wxy        1.0         Module dataloader
"""

# import lib
import os
import re
from itertools import islice
from sortedcontainers import SortedDict
import pandas as pd


def load_file_list_from_dir(path: str, filetype='', allow_recursion=True, log_file=None, formatter=None) -> SortedDict:
    """
    return files of current directory and its sub-dirs

    :param path: file path
    :param filetype: file extension
    :param allow_recursion:
    :param log_file:
    :param formatter: index format method
    :return: {formatter()['idx']: filepath}
    """
    print('Loading file list from %s' % path, file=log_file)
    dir_list = os.listdir(path)
    files = SortedDict()
    for f in dir_list:
        if allow_recursion and os.path.isdir(os.path.join(path, f)):
            files.update(load_file_list_from_dir(os.path.join(path, f), filetype, allow_recursion, log_file, formatter))

        filename = os.path.splitext(f)
        if filetype in filename[1]:
            if formatter is not None:
                idxes = formatter(filename[0])['idx']
                for idx in idxes:
                    files[idx] = os.path.join(path, f)
            else:
                files[filename[0]] = os.path.join(path, f)
    return files


def normal_filename_formatter(filename: str) -> dict:
    """
    index formatter, convert filename to dict index

    :param filename:
    :return: {'idx': [file_indexes]}
    """
    values = filename.split('_')
    res = {
        'frame_id': int(values[0]),
        'timestamp': float(re.sub(r"[^0-9.]", "", values[1])),
        'idx': [float(re.sub(r"[^0-9.]", "", values[1]))]
    }
    return res


def closest_item(sorted_dict: SortedDict, key) -> tuple:
    """
    return nearest item to key in source.index()

    :param sorted_dict:
    :param key: comparable object
    :return: tuple (index, value)
    """
    assert len(sorted_dict) > 0
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    res_key = min(keys, key=lambda k: abs(key - k))
    return res_key, sorted_dict[res_key]


def align(gt_path: str, camera_path: str, radar_path: str, output_path: str, **kwargs) -> pd.DataFrame:
    """
    align camera data and gt data according to radar timestamp

    :param gt_path:
    :param camera_path:
    :param radar_path:
    :param output_path: if not None, return DataFrame save as CSV file. Example: './test.csv'
    :param kwargs: filename format method,
    gt_formatter, default return tuple index (start_ts, end_ts);
    camera_formatter, default return current ts;
    radar_formatter, default return current ts
    gt_filetype, default '.ts'
    camera_filetype, default '.png'
    radar_filetype, default '.png'
    abspath, if True save as absolute path, default True
    :return: pandas.DataFrame
    """
    gt_formatter = kwargs.get('gt_formatter',
                              lambda s:
                              {
                                  'idx': (int(item) for item in s.split('_'))
                              }
                              )
    camera_formatter = kwargs.get('camera_formatter', normal_filename_formatter)
    radar_formatter = kwargs.get('radar_formatter', normal_filename_formatter)

    gt_filetype = kwargs.get('gt_filetype', '.ts')
    camera_filetype = kwargs.get('camera_filetype', '.png')
    radar_filetype = kwargs.get('radar_filetype', '.png')

    abspath = kwargs.get('abspath', True)

    gt_files = load_file_list_from_dir(gt_path, filetype=gt_filetype, formatter=gt_formatter)
    camera_files = load_file_list_from_dir(camera_path, filetype=camera_filetype, formatter=camera_formatter)
    radar_files = load_file_list_from_dir(radar_path, filetype=radar_filetype, formatter=radar_formatter)

    if not os.path.exists(os.path.split(output_path)[0]):
        os.mkdir(output_path)

    clm = ['TIMESTAMP', 'RADAR_PATH', 'CAMERA_PATH', 'GT_PATH', 'GT_TIME', 'X', 'Y', 'Z']
    df = pd.DataFrame(columns=clm)

    def get_time_offset(s):
        line_items = s.split('\t')
        # ignore file head with length 4, normal is 278
        if len(line_items) < 10:
            return -1.
        tm_offset = -1.
        # ignore file head with unavailable input
        try:
            # time offset
            tm_offset = float(re.sub(r"[^0-9.]", "", line_items[1]))
        except:
            pass
        return tm_offset

    for idx, frame in radar_files.items():
        gt_file = closest_item(gt_files, idx)[1]
        gt_df = pd.read_csv(gt_file)
        gt_tm_base = float(os.path.splitext(os.path.split(gt_file)[1])[0].split('_')[0])

        gt_tm = gt_df[gt_df.columns.tolist()[0]].map(get_time_offset)
        gt_closest_item = gt_df.iloc[(gt_tm + gt_tm_base - idx).abs().argsort()[:1]].values[0][0].split('\t')

        if abspath:
            radar_match_path = os.path.abspath(frame)
            camera_match_path = os.path.abspath(closest_item(camera_files, idx)[1])
            gt_match_path = os.path.abspath(gt_file)
        else:
            radar_match_path = frame
            camera_match_path = closest_item(camera_files, idx)[1]
            gt_match_path = gt_file

        row = pd.DataFrame([[idx, radar_match_path, camera_match_path, gt_match_path,
                             float(gt_closest_item[1])+gt_tm_base,
                             float(gt_closest_item[2]), float(gt_closest_item[3]), float(gt_closest_item[4])]],
                           columns=clm)
        df = df.append(row)

    if output_path is not None:
        df.to_csv(output_path)
    return df


if __name__ == '__main__':
    df = align('./__test__/gt', './__test__/camera', './__test__/radar', './__test__/output/test.csv', abspath=False)
    print(df)
