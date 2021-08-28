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
    w_idx = int(w_idx)
    h_idx = int(h_idx)
    if h_idx >= pcl.shape[0] or w_idx >= pcl.shape[1]:
        return [0,0,0]
    if pcl[w_idx, h_idx].all():
        return pcl[w_idx, h_idx]
    else:
        candidate_idxs = [
            (h_idx, w_idx-1),
            (h_idx, w_idx+1),
            (h_idx-1, w_idx),
            (h_idx+1, w_idx)
        ]
        for idxs in candidate_idxs:
            point = get_point_from_pcl(pcl, *idxs)
            if point.all():
                return point
