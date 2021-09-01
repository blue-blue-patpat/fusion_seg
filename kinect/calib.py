import numpy as np
import cv2
import cv2.aruco as aruco


def calibrate_kinect(image, pcls, intrinsic_mtx=None, distortion=None, aruco_size=0.8, show_result=False):
    if intrinsic_mtx is None:
        intrinsic_mtx = np.array([[982.31170654296875, 0, 1017.8777465820312],
                        [0, 982.267578125, 785.10211181640625],
                        [0,       0, 1]])
    if distortion is None:
        distortion = np.zeros((1, 4))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners,ids,rejectedImgPoints = aruco.detectMarkers(gray,aruco_dict,parameters=parameters)
    rvec,tvec,_ = aruco.estimatePoseSingleMarkers(corners, aruco_size, intrinsic_mtx, distortion)

    t = np.average([get_point_from_pcl(pcls, *np.asarray(corner, dtype=np.int16)) for corner in corners[0][0]], axis=0)

    t, R = [], []

    if rvec.shape[0]==0:
        print('please try other aruco!!!')
        return None
    for i in range(len(rvec)):
        # center = np.asarray(rmsd.centroid(corners[i][0]), dtype=np.int16)
        # print(center)
        # t.append(get_point(pcls, *center))
        t.append(np.average([get_point_from_pcl(pcls, *np.asarray(corner, dtype=np.int16)) for corner in corners[i][0]], axis=0))
        R.append(cv2.Rodrigues(rvec[i])[0])
    

    if show_result:
        imaxis = aruco.drawDetectedMarkers(image.copy(), corners, ids)
        for i in range(len(tvec)):
            imaxis = aruco.drawAxis(imaxis, intrinsic_mtx, distortion, rvec[i], tvec[i], 0.1)

        #center_p(corners, depth)
        cv2.imshow('frame', imaxis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return np.average(R, axis=0), np.average(t, axis=0)


def get_point_from_pcl(pcl: np.ndarray, h_idx, w_idx) -> np.ndarray:
    # init flag array
    checked_hw = np.zeros(pcl.shape[:2])

    # make sure index is int
    w_idx = int(w_idx)
    h_idx = int(h_idx)

    # if index exceeds range, return [0,0,0]
    if h_idx >= pcl.shape[0] or w_idx >= pcl.shape[1]:
        return [0,0,0]
    
    # init candidate indexes list
    candidate_idxs = [
        (h_idx, w_idx),
    ]

    # BFS the whole pcl array
    step = 0
    while len(candidate_idxs) > 0:
        _h, _w = candidate_idxs.pop(0)
        if checked_hw[_w, _h] == 1:
            continue

        point = pcl[_w, _h]
        checked_hw[w_idx, h_idx] = 1

        if point.any():
            print("[Point From PCL] Return after {} steps, input idxs {}, result idxs {}".format(step, (h_idx, w_idx), (_h, _w)))
            return point
        _candidate_idxs = [
            (_h, _w-1),
            (_h, _w+1),
            (_h-1, _w),
            (_h+1, _w)
        ]
        candidate_idxs += _candidate_idxs
        step += 1
