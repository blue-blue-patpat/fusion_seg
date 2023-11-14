import cv2
import os
import torch
import numpy as np
from PIL import Image
from ..p4t.tools import copy2cpu

INTRINSIC = np.array(
        [
            [2.4818480434445055e+03, 0., 8.6582666108041667e+02],
            [0, 2.0067972011730976e+03, 6.7647122567546967e+02],
            [0, 0, 1],
        ], dtype=np.float64
    )

def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()

class SequenceLoader2(object):
    def __init__(self, seq_path: str) -> None:
        self.seq_path = seq_path
        # load transformation matrix
        # with open(os.path.join(seq_path, "calib.txt")) as f:
        #     self.calib = eval(f.readline())

    def __len__(self):
        return len(os.listdir(os.path.join(self.seq_path, "img")))

    def __getitem__(self, idx: int):
        result = {}
        result["lidar"] = np.load(os.path.join(self.seq_path, 'pcd', 'a{}.npy'.format(idx)))
        result["image"] = cv2.imread(os.path.join(self.seq_path, "img", "a{}.jpg".format(idx)))
        result['label'] = Image.open(os.path.join(self.seq_path, 'label', 'a{}.png'.format(idx)))
        
        return result


def trans_mat_2_tensor(trans_mat):
    trans_mat_array = np.hstack((trans_mat["R"], np.array([trans_mat["t"]]).T))
    trans_mat_array = np.vstack((trans_mat_array, [0, 0, 0, 1]))
    return trans_mat_array


def trans_mat_2_dict(trans_mat):
    trans_mat = copy2cpu(trans_mat)
    trans_mat_dict = {"R": trans_mat[:3, :3], "t": trans_mat[:3, 3]}
    return trans_mat_dict


def project_pcd(pcd, trans_mat=None, intrinsic=None, image_size=np.array([1920, 1080])):
    """
    Project pcd to the image plane
    """
    if trans_mat is not None:
        pcd = (pcd - trans_mat["t"]) @ trans_mat["R"]
    intrinsic = np.array(intrinsic) if intrinsic is not None else INTRINSIC["master"]
    pcd_2d = ((pcd / pcd[:, 2:3]) @ intrinsic.T)[:, :2]
    pcd_2d = np.floor(pcd_2d).astype(int)
    pcd_2d[:, [0, 1]] = pcd_2d[:, [1, 0]]
    # filter out the points exceeding the image size
    pcd_2d = np.where(pcd_2d < image_size - 1, pcd_2d, image_size - 1)
    pcd_2d = np.where(pcd_2d > 0, pcd_2d, 0)
    return pcd_2d


def project_pcl(pcd, trans_mat=None, intrinsic=None, image_size=np.array([1536, 2048])):
    """
    Project pcd to the image plane
    """
    if trans_mat is not None:
        pcd = (pcd - trans_mat['t']) @ trans_mat['R']
    intrinsic = np.array(intrinsic) if intrinsic is not None else INTRINSIC
    pcd_2d = (pcd @ intrinsic.T)
    pcd_2d = np.floor(pcd_2d).astype(int)
  
    pcd_2d[:, [0, 1]] = pcd_2d[:, [1, 0]]
    
    # filter out the points exceeding the image size
    #pcd_2d = np.where(pcd_2d<image_size-1, pcd_2d, image_size-1)

    pcd_2d = np.where(pcd_2d>0, pcd_2d, 0)
    return pcd_2d

def normalize_pcd(pcd):

    ### 点云中心归一化
    centroid = np.mean(pcd, axis=0)
    pcd = pcd - centroid

    ### 点云缩放归一化
    m = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
    pcd = pcd / m
    return pcd


def project_pcd_torch(
    pcd, trans_mat=None, intrinsic=None, image_size=torch.tensor([1536, 2048])
):
    """
    Project pcd to the image plane
    """
    if trans_mat is not None:
        pcd = (pcd - trans_mat[:, None, :3, 3]) @ trans_mat[:, :3, :3]
    intrinsic = intrinsic if intrinsic is not None else INTRINSIC["master"]
    pcd_2d = ((pcd / pcd[:, :, 2:3]) @ torch.tensor(intrinsic).T.float().cuda())[
        :, :, :2
    ]
    pcd_2d = torch.floor(pcd_2d).long()
    pcd_2d[:, :, [0, 1]] = pcd_2d[:, :, [1, 0]]
    image_size = image_size.cuda()
    pcd_2d = torch.where(pcd_2d < image_size - 1, pcd_2d, image_size - 1)
    pcd_2d = torch.where(pcd_2d > 0, pcd_2d, 0)
    return pcd_2d


def filter_pcd(
    bounding_pcd: np.ndarray,
    target_pcd: np.ndarray,
    bound: float = 0.2,
    offset: float = 0,
):
    """
    Filter out the pcds of pcd_b that is not in the bounding_box of pcd_a
    """
    upper_bound = bounding_pcd[:, :3].max(axis=0) + bound
    lower_bound = bounding_pcd[:, :3].min(axis=0) - bound
    lower_bound[2] += offset

    mask_x = (target_pcd[:, 0] >= lower_bound[0]) & (target_pcd[:, 0] <= upper_bound[0])
    mask_y = (target_pcd[:, 1] >= lower_bound[1]) & (target_pcd[:, 1] <= upper_bound[1])
    mask_z = (target_pcd[:, 2] >= lower_bound[2]) & (target_pcd[:, 2] <= upper_bound[2])
    index = mask_x & mask_y & mask_z
    return target_pcd[index]


def pad_pcd(pcd, num_points, return_choices=True):
    """
    Pad pcd to the same number of points
    """
    if pcd.shape[0] > num_points:
        r = np.random.choice(pcd.shape[0], size=num_points, replace=False)
    else:
        repeat, residue = num_points // pcd.shape[0], num_points % pcd.shape[0]
        r = np.random.choice(pcd.shape[0], size=residue, replace=False)
        r = np.concatenate([np.arange(pcd.shape[0]) for _ in range(repeat)] + [r], axis=0)
    if return_choices:
        return pcd[r, :], r
    return pcd[r, :]


def crop_image(
    joints: np.ndarray,
    image: np.ndarray,
    trans_mat: dict = None,
    visual: bool = False,
    margin: float = 0.2,
    square: bool = False,
    intrinsic=None,
    return_box: bool = False,
):
    """
    Crop the person area of image
    """
    # transform the joints to camera coordinate
    if trans_mat is not None:
        joints = (joints - trans_mat["t"]) @ trans_mat["R"]
    intrinsic = intrinsic if intrinsic is not None else INTRINSIC["master"]
    joint_max = joints.max(axis=0) + margin
    joint_min = joints.min(axis=0) - margin
    # get 3d bounding box from joints
    box_3d = np.array(
        [
            [joint_min[0], joint_min[1], joint_min[2]],
            [joint_min[0], joint_min[1], joint_max[2]],
            [joint_min[0], joint_max[1], joint_max[2]],
            [joint_min[0], joint_max[1], joint_min[2]],
            [joint_max[0], joint_max[1], joint_max[2]],
            [joint_max[0], joint_min[1], joint_max[2]],
            [joint_max[0], joint_max[1], joint_min[2]],
            [joint_max[0], joint_min[1], joint_min[2]],
        ]
    )
    box_2d = []
    # project 3d bounding box to 2d image plane
    box_2d = project_pcd(box_3d, intrinsic=intrinsic)
    box_min = box_2d.min(0)
    box_max = box_2d.max(0)
    if square:
        size = box_max - box_min
        diff = abs(size[0] - size[1]) // 2
        if size[0] > size[1]:
            box_max[1] += diff
            box_min[1] -= diff
        elif size[0] < size[1]:
            box_max[0] += diff
            box_min[0] -= diff
    box_max = np.where(box_max < image.shape[:2], box_max, image.shape[:2])
    box_min = np.where(box_min > 0, box_min, 0)
    # crop image
    crop_img = image[box_min[0] : box_max[0], box_min[1] : box_max[1]]

    if visual:
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.rectangle(image, box_min[::-1], box_max[::-1], (0, 0, 255), 2)
        cv2.imshow("img", image)
        cv2.waitKey(0)

    if return_box:
        return crop_img, box_min, box_max
    return crop_img


def get_rgb_value(pcd, image, visual=False, ret_image=False):
    pcd_2d = project_pcd(pcd)

    pcd_color = image[pcd_2d[:, 0], pcd_2d[:, 1]]
    pcd_with_feature = np.hstack((pcd, pcd_color / 255))

    if visual:
        image[pcd_2d[:, 0], pcd_2d[:, 1]] = [0, 255, 0]
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow("img", image)
        cv2.waitKey(0)

    return pcd_with_feature


def get_rgb_feature(pcd, image, visual=False):
    pcd_2d = project_pcd(pcd)

    pcd_color = image[pcd_2d[:, 0], pcd_2d[:, 1]]
    pcd_with_feature = np.hstack((pcd, pcd_color / 255))

    if visual:
        image[pcd_2d[:, 0], pcd_2d[:, 1]] = [0, 255, 0]
        cv2.namedWindow("img", 0)
        cv2.resizeWindow("img", 640, 480)
        cv2.imshow("img", image)
        cv2.waitKey(0)

    return pcd_with_feature


def convert_square_image(image):
    """
    convert to square with slice
    """
    img_h, img_w, img_c = image.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = int(abs(img_w - img_h) / 2)
        image = image.transpose((1, 0, 2)).copy() if img_w < img_h else image
        background = np.full((long_side, long_side, img_c), 255, np.uint8)
        background[loc : loc + short_side] = image[...]
        image = background.transpose((1, 0, 2)).copy() if img_w < img_h else background
    return image


def rodrigues_2_rot_mat(rvecs):
    batch_size = rvecs.shape[0]
    r_vecs = rvecs.reshape(-1, 3)
    total_size = r_vecs.shape[0]
    thetas = torch.norm(r_vecs, dim=1, keepdim=True)
    is_zero = torch.eq(torch.squeeze(thetas), torch.tensor(0.0))
    u = r_vecs / thetas

    # Each K is the cross product matrix of unit axis vectors
    # pyformat: disable
    zero = torch.autograd.Variable(
        torch.zeros([total_size], device="cuda")
    )  # for broadcasting
    Ks_1 = torch.stack([zero, -u[:, 2], u[:, 1]], axis=1)  # row 1
    Ks_2 = torch.stack([u[:, 2], zero, -u[:, 0]], axis=1)  # row 2
    Ks_3 = torch.stack([-u[:, 1], u[:, 0], zero], axis=1)  # row 3
    # pyformat: enable
    Ks = torch.stack([Ks_1, Ks_2, Ks_3], axis=1)  # stack rows

    identity_mat = torch.autograd.Variable(
        torch.eye(3, device="cuda").repeat(total_size, 1, 1)
    )
    Rs = (
        identity_mat
        + torch.sin(thetas).unsqueeze(-1) * Ks
        + (1 - torch.cos(thetas).unsqueeze(-1)) * torch.matmul(Ks, Ks)
    )
    # Avoid returning NaNs where division by zero happened
    R = torch.where(is_zero[:, None, None], identity_mat, Rs)

    return R.reshape(batch_size, -1)


def gen_random_indices(
    max_random_num, min_random_num=0, random_ratio=1.0, random_num=True, size=0
):
    # generate x% of max_random_num random indices
    pb = np.random.rand() if random_num else 1.0
    num_indices = (
        np.floor(pb * random_ratio * (max_random_num - min_random_num))
        if not size
        else size
    )
    indices = np.random.choice(
        np.arange(min_random_num, max_random_num), replace=False, size=int(num_indices)
    )
    return np.sort(indices)


import cv2
import numpy as np
import matplotlib.pyplot as plt


cmap = plt.cm.jet

def read_bin(bin_path, intensity=False):
    """
    读取kitti bin格式文件点云
    :param bin_path:   点云路径
    :param intensity:  是否要强度
    :return:           numpy.ndarray `N x 3` or `N x 4`
    """
    lidar_points = np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))
    if not intensity:
        lidar_points = lidar_points[:, :3]
    return lidar_points


def read_calib(calib_path):
    """
    读取kitti数据集标定文件
    下载的彩色图像是左边相机的图像, 所以要用P2
    extrinsic = np.matmul(R0, lidar2camera)
    intrinsic = P2
    P中包含第i个相机到0号摄像头的距离偏移(x方向)
    extrinsic变换后的点云是投影到编号为0的相机(参考相机)坐标系中并修正后的点
    intrinsic(P2)变换后可以投影到左边相机图像上
    P0, P1, P2, P3分别代表左边灰度相机，右边灰度相机，左边彩色相机，右边彩色相机
    :return: P0-P3 numpy.ndarray           `3 x 4`
             R0 numpy.ndarray              `4 x 4`
             lidar2camera numpy.ndarray    `4 x 4`
             imu2lidar numpy.ndarray       `4 x 4`

    >>> P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m = read_calib(calib_path)
    >>> extrinsic_m = np.matmul(R0, lidar2camera_m)
    >>> intrinsic_m = P2
    """
    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))
    imu2lidar_m = np.array(list(map(float, raw[6].split()[1:]))).reshape((3, 4))
    imu2lidar_m = np.vstack((imu2lidar_m, np.array([0, 0, 0, 1])))
    return P0, P1, P2, P3, R0, lidar2camera_m, imu2lidar_m


def image2camera(point_in_image, intrinsic):
    """
    图像系到相机系反投影
    :param point_in_image: numpy.ndarray `N x 3` (u, v, z)
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: numpy.ndarray `N x 3` (x, y, z)
    u = fx * X/Z + cx
    v = fy * Y/Z + cy
    X = (u - cx) * Z / fx
    Y = (v - cy) * z / fy
       [[fx, 0,  cx, -fxbi],
    K=  [0,  fy, cy],
        [0,  0,  1 ]]
    """
    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    u = point_in_image[:, 0]
    v = point_in_image[:, 1]
    z = point_in_image[:, 2]
    x = ((u - intrinsic[0, 2]) * z - intrinsic[0, 3]) / intrinsic[0, 0]
    y = ((v - intrinsic[1, 2]) * z - intrinsic[1, 3]) / intrinsic[1, 1]
    point_in_camera = np.vstack((x, y, z))
    return point_in_camera


def lidar2camera(point_in_lidar, extrinsic):
    """
    雷达系到相机系投影
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :return: point_in_camera numpy.ndarray `N x 3`
    """
    point_in_lidar = np.hstack((point_in_lidar, np.ones(shape=(point_in_lidar.shape[0], 1)))).T
    point_in_camera = np.matmul(extrinsic, point_in_lidar)[:-1, :]  # (X, Y, Z)
    point_in_camera = point_in_camera.T
    return point_in_camera


def camera2image(point_in_camera, intrinsic):
    """
    相机系到图像系投影
    :param point_in_camera: point_in_camera numpy.ndarray `N x 3`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = point_in_camera.T
    point_z = point_in_camera[-1]

    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    point_in_camera = np.vstack((point_in_camera, np.ones((1, point_in_camera.shape[1]))))
    point_in_image = (np.matmul(intrinsic, point_in_camera) / point_z)  # 向图像上投影
    point_in_image[-1] = point_z
    point_in_image = point_in_image.T
    return point_in_image


def lidar2image(point_in_lidar, extrinsic, intrinsic):
    """
    雷达系到图像系投影  获得(u, v, z)
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = lidar2camera(point_in_lidar, extrinsic)
    point_in_image = camera2image(point_in_camera, intrinsic)
    return point_in_image


def get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w):
    """
    获取fov内的点云mask, 即能够投影在图像上的点云mask
    :param point_in_lidar:   雷达点云 numpy.ndarray `N x 3`
    :param extrinsic:        外参 numpy.ndarray `4 x 4`
    :param intrinsic:        内参 numpy.ndarray `3 x 3` or `3 x 4`
    :param h:                图像高 int
    :param w:                图像宽 int
    :return: point_in_image: (u, v, z)  numpy.ndarray `N x 3`
    :return:                 numpy.ndarray  `1 x N`
    """
    point_in_image = lidar2image(point_in_lidar, extrinsic, intrinsic)
    front_bound = point_in_image[:, -1] > 0
    point_in_image[:, 0] = np.round(point_in_image[:, 0])
    point_in_image[:, 1] = np.round(point_in_image[:, 1])
    u_bound = np.logical_and(point_in_image[:, 0] >= 0, point_in_image[:, 0] < w)
    v_bound = np.logical_and(point_in_image[:, 1] >= 0, point_in_image[:, 1] < h)
    uv_bound = np.logical_and(u_bound, v_bound)
    mask = np.logical_and(front_bound, uv_bound)
    return point_in_image[mask], mask




