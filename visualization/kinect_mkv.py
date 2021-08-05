from argparse import ArgumentParser

from typing import Optional, Tuple

import os
import cv2
import numpy as np

from pyk4a import ImageFormat, PyK4APlayback

from dataloader.utils import clean_dir, ymdhms_time


def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")


def save_and_play(playback: PyK4APlayback, save_path="./", start_tm=0):
    i = 0
    while True:
        try:
            capture = playback.get_next_capture()
            if i==0:
                color_tm_offset = capture.color_timestamp_usec
                depth_tm_offset = capture.depth_timestamp_usec
            if capture.color is not None:
                color = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
                cv2.imshow("Color", color)
                cv2.imwrite(os.path.join(save_path, "color/id={}_tm={}.png".format(i, start_tm + (capture.color_timestamp_usec - color_tm_offset)/1000000)), color)
            if capture.depth is not None:
                cv2.imshow("Depth", colorize(capture.depth, (None, 5000)))
                cv2.imwrite(os.path.join(save_path, "depth/id={}_tm={}.png".format(i, start_tm + (capture.depth_timestamp_usec - depth_tm_offset)/1000000)), capture.depth)
            key = cv2.waitKey(10)
            print(i, end="\r")
            i += 1
            if key != -1:
                break
        except EOFError:
            break
    cv2.destroyAllWindows()


def save(playback: PyK4APlayback, save_path="./", start_tm=0):
    from multiprocessing.dummy import Pool
    info = [0, 0]
    pool = Pool()
    def process(capture, save_path, start_tm, idx, info, color_tm_offset, depth_tm_offset):
        if capture.color is not None:
            cv2.imwrite(os.path.join(save_path, "color/id={}_tm={}.png".format(idx, start_tm + (capture.color_timestamp_usec - color_tm_offset)/1000000)), capture.color)
        if capture.depth is not None:
            cv2.imwrite(os.path.join(save_path, "depth/id={}_tm={}.png".format(idx, start_tm + (capture.depth_timestamp_usec - depth_tm_offset)/1000000)), capture.depth)
        if capture.depth_point_cloud is not None:
            np.save(os.path.join(save_path, "pcls/id={}_tm={}".format(idx, start_tm + (capture.depth_timestamp_usec - depth_tm_offset)/1000000)), capture.depth_point_cloud)
        info[1] += 1

    while True:
        try:
            capture = playback.get_next_capture()
            if info[0]==0:
                color_tm_offset = capture.color_timestamp_usec
                depth_tm_offset = capture.depth_timestamp_usec
            pool.apply_async(process, (capture, save_path, start_tm, info[0], info, color_tm_offset, depth_tm_offset))
            
            print("{} : [Kinect MKV] Extracting {}/{} frame.".format(ymdhms_time(), info[0], info[1]), end="\r")
            info[0] += 1
        except EOFError:
            break
    pool.close()
    pool.join()
    print()


def extract_mkv(filename, enable_view=False) -> None:
    print("{} : [Kinect MKV] Extracting mkv to color and depth; enable_view={}.".format(ymdhms_time(), enable_view))
    playback = PyK4APlayback(filename)
    playback.open()

    save_path, _ = os.path.split(filename)
    filename, extension = os.path.splitext(_)

    with open(os.path.join(save_path, "info.txt"), 'r') as f:
        tm = float(f.readline())

    clean_dir(os.path.join(save_path, "color"))
    clean_dir(os.path.join(save_path, "depth"))
    clean_dir(os.path.join(save_path, "pcls"))

    info(playback)
    if enable_view:
        save_and_play(playback, save_path, tm)
    else:
        save(playback, save_path, tm)

    playback.close()
