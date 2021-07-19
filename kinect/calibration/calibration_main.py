import argparse
import os
import cv2
from typing import Tuple, Dict, Iterable, List, Union
from open3d import open3d as o3d
from realsense_device_manager import Device
import pyrealsense2 as rs
import numpy as np
import rmsd
from cv2 import aruco

from camera import initialize_connected_cameras, extract_color_image, close_connected_cameras
from utilities import create_if_not_exists, dump_dict_as_json, DEFAULT_DATA_DIR,load_json_to_dict

ReferencePoints = Dict[str, Dict[str, List[Union[int, Tuple[float, float, float]]]]]

def parse_args_aruco_dection() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Detects all available acuro markers')
    parser.add_argument('--data_dir', type=lambda item: create_if_not_exists(item), required=False,
                        help='Data location to load and dump config files', default=DEFAULT_DATA_DIR)
    parser.add_argument('--remove_previous_data',default=True, action='store_true',
                        help='If set, each reference-point file in data_dir will be removed before performing new '
                             'detection.')
    return parser.parse_args()


def remove_previous_data(data_dir: str):
    for file in os.listdir(data_dir):
        if file.endswith('_reference_points.json'):
            os.remove(os.path.join(data_dir, file))


def detect_aruco_targets(rgb_image: np.array) -> Tuple[np.array, List[float]]:
    aruco_corners, aruco_ids, _ = aruco.detectMarkers(rgb_image, aruco.Dictionary_get(aruco.DICT_6X6_250))
    aruco_ids = aruco_ids.astype(float)
    return np.array([item[0] for item in aruco_corners]), [item[0] for item in aruco_ids]


def determine_aruco_center(corners: np.array) -> np.array:
    assert corners.shape == (4, 2,)
    return rmsd.centroid(corners)


def dump_reference_points(device_id: str, aruco_ids: List[float], aruco_centers: List[np.array], data_dir: str):
    reference_points = {
        'camera_id': device_id,
        'aruco': aruco_ids,
        'centers': aruco_centers
    }
    dump_dict_as_json(reference_points, os.path.join(data_dir, f'{device_id}_reference_points.json'))

def main_aruco_detection():
    args = parse_args_aruco_dection()
    #去除之前的数据
    if args.remove_previous_data:
        remove_previous_data(args.data_dir)
    #启动相机
    cameras = initialize_connected_cameras()
    for camera in cameras:
        frames = camera.poll_frames()
        
        color_frame = extract_color_image(frames)
        aruco_ = {}
        depth_intrinsics_total = {}
        #检测二维码角点
        aruco_corners_image_points, aruco_ids = detect_aruco_targets(color_frame)
        #定位二维码中心
        aruco_centers_image_points = [determine_aruco_center(corners) for corners in aruco_corners_image_points]
        count = 0
        for ids in aruco_ids:
            aruco_[ids] = aruco_centers_image_points[count]
            count = count + 1 
        #图像坐标系下的二维点转为相机坐标系下的三维点，这两个文件用的算法不太一样
        object_points = camera.image_points_to_object_points(aruco_, frames)
        #print(depth_intrinsics)
        #depth_intrinsics_total[camera.device_id] = depth_intrinsics
        aruco_ids = []
        aruco_centers_object_points = []
        for ids in object_points:
            aruco_ids.append(ids)
            aruco_centers_object_points.append(object_points[ids])
        #保存二维码标号和中心坐标
        dump_reference_points(camera.device_id, aruco_ids, aruco_centers_object_points, args.data_dir)
        dump_dict_as_json(depth_intrinsics_total,os.path.join(args.data_dir, 'depth_intrinsics_total.json'))
    close_connected_cameras(cameras)

# calibration

def parse_args_calibration() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Performes an extrinsic calibration for each available camera')
    parser.add_argument('--data_dir', type=lambda item: create_if_not_exists(item), default=DEFAULT_DATA_DIR,
                        help='Data location to load and dump config files')
    return parser.parse_args()

#读入二维码数据
def read_aruco_data(data_dir: str) -> dict:
    calibration_data = {}
    for file in os.listdir(data_dir):
        if file.endswith("_reference_points.json"):
            reference_points = load_json_to_dict(os.path.join(data_dir, file))
            calibration_data[reference_points['camera_id']] = {
                'aruco': reference_points['aruco'],
                'centers': reference_points['centers']
            }
    return calibration_data

#利用kabsch算法计算空间旋转平移矩阵
def calculate_transformation_kabsch(src_points: np.ndarray, dst_points: np.ndarray) -> Tuple[np.array, float]:
    """
    Calculates the optimal rigid transformation from src_points to
    dst_points
    (regarding the least squares error)
    Parameters:
    -----------
    src_points: array
        (3,N) matrix
    dst_points: array
        (3,N) matrix

    Returns:
    -----------
    rotation_matrix: array
        (3,3) matrix

    translation_vector: array
        (3,1) matrix
    rmsd_value: float
    """
    assert src_points.shape == dst_points.shape
    if src_points.shape[0] != 3:
        raise Exception("The input data matrix had to be transposed in order to compute transformation.")

    src_points = src_points.transpose()
    dst_points = dst_points.transpose()

    src_points_centered = src_points - rmsd.centroid(src_points)
    dst_points_centered = dst_points - rmsd.centroid(dst_points)

    rotation_matrix = rmsd.kabsch(src_points_centered, dst_points_centered)
    rmsd_value = rmsd.kabsch_rmsd(src_points_centered, dst_points_centered)

    translation_vector = rmsd.centroid(dst_points) - np.matmul(rmsd.centroid(src_points), rotation_matrix)

    return create_homogenous(rotation_matrix.transpose(), translation_vector.transpose()), rmsd_value

#将旋转平移矩阵整合为单应性矩阵
def create_homogenous(rotation_matrix: np.array, translation_vector: np.array) -> np.array:
    homogenous = np.append(rotation_matrix, [[vec] for vec in translation_vector], axis=1)
    homogenous = np.append(homogenous, np.array([[0, 0, 0, 1]]), axis=0)
    return homogenous

#定义base camera，同时识别到6、7、10且检测到的二维码总数最多的相机
def define_base_camera(aruco_data: ReferencePoints) -> str:
    """Finds the camera that detected each bottom target and also detected the highest amount of other targets"""
    base_camera = None
    detected_targets = 0
    bottom_arucos = {6.0,7.0,10.0}
    for k, v in aruco_data.items():
        print(set(v['aruco']))
        if bottom_arucos <= set(v['aruco']) and len(v['aruco']) > detected_targets:
            base_camera = k
            detected_targets = len(v['aruco'])

    assert base_camera is not None
    return base_camera


def calculate_relative_transformations(aruco_data: dict, base_camera: str) -> Dict[str, np.array]:
    transformations = {
        base_camera: np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=float)
    }
    dst_arucos = aruco_data[base_camera]['aruco']
    dst_points = aruco_data[base_camera]['centers']
    for k, v in aruco_data.items():
        if not k == base_camera:
            # 1. intersect arucos
            src_arucos = v['aruco']
            src_points = v['centers']
            intersection = set(dst_arucos).intersection(set(src_arucos))
            # 2. create two sorted lists of points
            assert len(intersection) > 2
            dst_sorted = []
            src_sorted = []
            for aruco_id in intersection:
                dst_sorted.append(dst_points[dst_arucos.index(aruco_id)])
                src_sorted.append(src_points[src_arucos.index(aruco_id)])

            transformation, rmsd_value = calculate_transformation_kabsch(np.array(src_sorted).transpose(),
                                                                         np.array(dst_sorted).transpose())
            print("RMS error for calibration with device number", k, "is :", rmsd_value, "m")
            transformations[k] = transformation

    return transformations


def calculate_absolute_transformations(reference_points: Iterable[Tuple[int, Tuple[float, float, float]]]) -> np.array:
    # find bottom markers
    src_x, src_center, src_z = None, None, None
    for marker_id, point in reference_points:
        if marker_id == 7.0:
            src_x = np.array(point)
        elif marker_id == 6.0:
            src_center = np.array(point)
        elif marker_id == 10.0:
            src_z = np.array(point)

    assert not (src_x is None or src_center is None or src_z is None)

    dst_center = np.array([0., 0., 0.])
    dst_x = np.array([np.linalg.norm(src_center - src_x), 0., 0.])
    dst_z = np.array([0., 0., np.linalg.norm(src_center - src_z)])

    src_points = np.array([src_x, src_z, src_center])
    dst_points = np.array([dst_x, dst_z, dst_center])
    transformation, rmsd_value = calculate_transformation_kabsch(src_points.transpose(), dst_points.transpose())
    print("RMS error for calibration to real world system is :", rmsd_value, "m")

    return transformation


def generate_extrinsics(aruco_data: dict) -> dict:
    base_camera = define_base_camera(aruco_data)
    base_camera_reference_points = zip(aruco_data[base_camera]['aruco'], aruco_data[base_camera]['centers'])
    #计算其他相机到base camera的单应性矩阵
    relative_transformations = calculate_relative_transformations(aruco_data, base_camera)
    #计算base camera 到世界坐标系的单应性矩阵
    absolute_transformation = calculate_absolute_transformations(base_camera_reference_points)
    final_transformations = {}
    for k, v in relative_transformations.items():
        final_transformations[k] = np.dot(absolute_transformation, v)
    #返回每个相机到世界坐标下的单应性矩阵
    return final_transformations


def main_calibration():
    """Creates a camera setup file containing camera ids and extrinsic information as 4 x 4 matrix"""
    args = parse_args_calibration()
    aruco_data = read_aruco_data(args.data_dir)
    final_transformations = generate_extrinsics(aruco_data)
    for k, v in final_transformations.items():
        final_transformations[k] = v.tolist()
    #保存相机到世界坐标系的单应性矩阵
    dump_dict_as_json(final_transformations, os.path.join(args.data_dir, 'camera_array.json'))

# measurement
def enumerate_connected_devices(context):
	connect_device = []
	for d in context.devices:
		if d.get_info(rs.camera_info.name).lower() != 'platform camera':
			connect_device.append(d.get_info(rs.camera_info.serial_number))
	return connect_device

class DeviceManager:
    def __init__(self, context, pipeline_configuration):
        assert isinstance(context, type(rs.context()))
        assert isinstance(pipeline_configuration, type(rs.config()))
        self._context = context
        self._available_devices = enumerate_connected_devices(context)
        self._enabled_devices = {}
        self._config = pipeline_configuration
        self._frame_counter = 0

    #启动单个相机，并对相机参数进行设置
    def enable_device(self, device_serial, enable_ir_emitter):
        pipeline = rs.pipeline()
        self._config.enable_device(device_serial)
        pipeline_profile = pipeline.start(self._config)

        # Set the acquisition parameters
        sensor_depth = pipeline_profile.get_device().first_depth_sensor()
        sensor_depth.set_option(rs.option.emitter_enabled, 1 if enable_ir_emitter else 0)
        sensor_color = pipeline.get_active_profile().get_device().query_sensors()[1]
        if device_serial == "043422251184":
			#sensor_color.set_option(rs.option.saturation, 56)
			#sensor_color.set_option(rs.option.white_balance,4600)
            sensor_color.set_option(rs.option.hue, 5)
            sensor_color.set_option(rs.option.brightness,15)
		#set_option(rs.option.enable_auto_exposure,1)
		#sensor_color.set_option(rs.option.enable_auto_white_balance,0)
        if device_serial == "043422250969":
            sensor_color.set_option(rs.option.hue, -3)
            sensor_color.set_option(rs.option.brightness, 15)
        if device_serial == "043422251838":
            sensor_color.set_option(rs.option.brightness, 15)
        self._enabled_devices[device_serial] = (Device(pipeline, pipeline_profile))
        return pipeline_profile

    def enable_all_devices(self, enable_ir_emitter=True):
        print(str(len(self._available_devices)) + " devices have been found")
        pipeline_profile_total = {}
        serial_total = []
        for serial in self._available_devices:
            pipeline_profile_total[serial]=self.enable_device(serial, enable_ir_emitter)
            serial_total.append(serial)
            print(serial,type(serial))
        print(serial_total)
        all_pipeline = self.get_all_pipeline()
        return all_pipeline,serial_total

    def get_all_pipeline(self):
        all_pipeline = {}
        for (serial, device) in self._enabled_devices.items():
            all_pipeline[serial] = device.pipeline
        return all_pipeline

    def disable_streams(self):
        self._config.disable_all_streams()



def parse_args_measure() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Measures the scene')
    parser.add_argument('--bottom', type=float, default=-0.5, help='Bottom of the measurement sphere in m')
    parser.add_argument('--height', type=float, default=1.0, help='Height of the measurement sphere in m')
    parser.add_argument('--radius', type=float, default=0.001, help='Radius of the measurement sphere in m')
    parser.add_argument('--data_dir', type=lambda item: create_if_not_exists(item), default=DEFAULT_DATA_DIR,
                        help='Data location to load and dump config files', )
    return parser.parse_args()

#判断点云是否超出范围
def is_in_measurement_cylinder(point: np.array, bottom: float, height: float, radius: float) -> bool:
    is_beside_cylinder = np.linalg.norm([point[0], point[2]]) < radius
    is_above_cylinder = point[2] > (bottom + height)
    is_below_cylinder = point[2] < bottom
    return not (is_above_cylinder or is_below_cylinder or is_beside_cylinder)


def remove_unnecessary_content(object_points, bottom: float, height: float, radius: float) -> np.array:
    """Removes points outside the defined measurement cylinder"""
    filtered_points = []
    for item in object_points:
        if is_in_measurement_cylinder(item, bottom, height, radius):
            filtered_points.append(item.tolist())
    return filtered_points


def apply_transformation(points: np.array, extrinsic: np.array) -> np.array:
    assert(points.shape[0] == 3)
    n = points.shape[1]
    print(extrinsic)
    points_ = np.vstack((points, np.ones((1, n))))
    points_trans_ = np.matmul(extrinsic, points_)
    points_transformed = np.true_divide(points_trans_[:3, :], points_trans_[[-1], :])
    return points_transformed

def create_output(vertices, colors, filename):
    vertices = np.hstack([vertices, colors])
    # 必须先写入，然后利用write()在头部插入ply header
    np.savetxt(filename, vertices, fmt='%f')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property float red
    		property float green
    		property float blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)

def convert_depth_frame_to_pointcloud(depth_image, depth_intrinsics):
    # print(depth_image.shape)
    [height, width] = depth_image.shape
    nx = np.linspace(0, width-1, width)
    ny = np.linspace(0, height-1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - depth_intrinsics.ppx)/depth_intrinsics.fx
    y = (v.flatten() - depth_intrinsics.ppy)/depth_intrinsics.fy
    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)

    #x = x[np.nonzero(z)]
    #y = y[np.nonzero(z)]
    #z = z[np.nonzero(z)]
    return x, y, z


def main_measure():
    
    args = parse_args_measure()
    dictionary = load_json_to_dict(os.path.join(args.data_dir, 'camera_array.json'))
    #all_connected_cams = camera.initialize_connected_cameras()
    
    frame_rate = 30
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.depth, 640,480, rs.format.z16, frame_rate)
    rs_config.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, frame_rate)
    device_manager = DeviceManager(rs.context(), rs_config)
    all_pipeline,serial_total = device_manager.enable_all_devices()
    
    point_cloud_all = np.array([-1, -1, -1]).reshape(1, -1)
    color_all = np.array([-1, -1, -1]).reshape(1, -1)

    #将深度通道向颜色配准
    align_to = rs.stream.color
    align = rs.align(align_to)

    intrinsics = {}
    for cam in serial_total:
        #读入单应性矩阵
        trans_matrix = np.array(dictionary[cam])
        #frame = cam.poll_frames()
        pipeline = all_pipeline[cam] 
        frame = pipeline.wait_for_frames()
        frame = align.process(frame)

        depth_frame: rs.depth_frame = frame.get_depth_frame()

        color_image = np.array(frame.get_color_frame().get_data())
        depth_image = np.array(frame.get_depth_frame().get_data()) 
        
        path_color = "image_test/color/" +"cam_" + str(cam) +".png"		
        cv2.imwrite(path_color,color_image)

        depth_profile = depth_frame.get_profile().as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()  

        #读入内参，并保存
        intrinsics[cam] = [depth_intrinsics.ppx,depth_intrinsics.ppy,depth_intrinsics.fx,depth_intrinsics.fy]
        B = color_image[:, :, 0].flatten()
        G = color_image[:, :, 1].flatten()
        R = color_image[:, :, 2].flatten()
        color = np.vstack((R, G))
        color = np.vstack((color, B))
        #Filter the depth_frame using the Temporal filter and get the corresponding pointcloud for each frame
        #filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80)
        #将深度图转为相机坐标系下的点云
        point_cloud = convert_depth_frame_to_pointcloud(depth_image,depth_intrinsics)
        point_cloud = np.array(point_cloud)
        #将相机坐标系下的点云转为世界坐标系
        point_cloud = apply_transformation(point_cloud,trans_matrix)
        #叠加RGB信息
        point_cloud = np.vstack((point_cloud, color))
        point_cloud = point_cloud.transpose()       
        #point_cloud = remove_unnecessary_content(point_cloud, args.bottom, args.height, args.radius)
        point_cloud = np.array(point_cloud)

        point_cloud_all = np.row_stack((point_cloud_all, point_cloud[:,0:3]))
        color_all = np.row_stack((color_all, point_cloud[:,3:6]))

    f1 = open(os.path.join(args.data_dir, 'intrinsics'),'w')
    f1.write(str(intrinsics))
    f1.close()

    np.save(os.path.join(args.data_dir, 'serial_total.npy'),serial_total)

    point_cloud_all = np.delete(point_cloud_all, 0, 0)
    color_all = np.delete(color_all, 0, 0)
    #保存点云文件
    path = "output/result.ply"
    create_output(point_cloud_all,color_all,path)
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main_aruco_detection()
    main_calibration()
    main_measure()
