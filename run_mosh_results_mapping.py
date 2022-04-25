import shutil
import time
import os
import numpy as np
import pickle
from mosh.utils import mesh_from_mosh_params, get_spec_files

"""
Match the timestamps of mosh results with optitrack
"""

def get_time_dict(driver_list=None):
    if driver_list is None:
        driver_list = ['/home/nesc525/drivers/1', '/home/nesc525/drivers/2', '/home/nesc525/drivers/3']
    time_dict = {}
    for driver_path in driver_list:
        for root_path in os.listdir(driver_path):
            npz_list = get_spec_files(os.path.join(driver_path, root_path, 'optitrack'), '.npz')
            if npz_list:
                npz_list.sort(key=lambda x: int(x.split('_')[0][3:]))
                key_t = float(npz_list[0].split('_')[1][3:-4])
                time_dict.update({key_t:dict(path=os.path.join(driver_path, root_path), opti_npz_list=npz_list)})
    
    return time_dict

def map_timestamp(mosh_path):
    time_dict = get_time_dict()
    if 'mosh_results' in mosh_path:
        res_path = os.path.join(mosh_path, 'soma_subject1')
        # res_path = '/home/nesc525/drivers/4/chen/mosh/mosh_results/SOMA_manulabeled_mpc_perple17/soma_subject1'
        for pkl_name in os.listdir(res_path):
            if 'stageii' in pkl_name:
                value_time = time.mktime(time.strptime(pkl_name[5:-12], "%Y-%m-%d-%I.%M.%S-%p"))
                key_time = min(time_dict.keys(), key=lambda x: abs(x - value_time))

                if abs(value_time - key_time) < 10:
                    print(os.path.join(res_path, pkl_name), '-->', time_dict[key_time]['path'])
                    param_save_path = os.path.join(time_dict[key_time]['path'], 'mosh', 'param')
                    save_path = os.path.join(time_dict[key_time]['path'], 'mosh', 'pkl')
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path)
                    os.makedirs(save_path)
                    if os.path.exists(param_save_path):
                        shutil.rmtree(param_save_path)
                    os.makedirs(param_save_path)
                    pkl_res = mesh_from_mosh_params(mosh_params=pickle.load(os.path.join(res_path, pkl_name)))

                    print(len(pkl_res['vertices']), len(time_dict[key_time]['opti_npz_list']))
                    mosh_offset = len(pkl_res['vertices']) - len(time_dict[key_time]['opti_npz_list'])
                    with open(os.path.join(time_dict[key_time]['path'], "offsets.txt"), "r") as f:
                        offset_dict = eval(f.readline())
                        base_key = f.readline()
                    offset_dict.update(dict(mosh=mosh_offset + offset_dict['optitrack']))
                    with open(os.path.join(time_dict[key_time]['path'], "offsets.txt"), "w") as f:
                        f.write(str(offset_dict) + '\n')
                        f.write(base_key)

                    shutil.copyfile(os.path.join(res_path, pkl_name), os.path.join(save_path, pkl_name))

                    for i in range(len(pkl_res['vertices'])):
                        np.savez(file=os.path.join(param_save_path, time_dict[key_time]['opti_npz_list'][i]), 
                                pose=pkl_res['pose'][i],
                                pose_hand=pkl_res['pose_hand'][i],
                                shape=pkl_res['shape'],
                                vertices=pkl_res['vertices'][i],
                                joints=pkl_res['joints'][i],
                                faces=pkl_res['faces'],
                                )

                else:
                    print(pkl_name, '++', time_dict[key_time]['path'])
                    print(value_time, '++', key_time)

    if 'mesh' in mosh_path:
        res_path = os.path.join(mosh_path, 'mesh_files')
        for obj_folder in os.listdir(res_path):
            value_time = time.mktime(time.strptime(obj_folder[-22:], "%Y-%m-%d-%I.%M.%S-%p"))
            key_time = min(time_dict.keys(), key=lambda x: abs(x - value_time))

            if abs(value_time - key_time) < 10:
                print(os.path.join(res_path, obj_folder), '-->', time_dict[key_time]['path'])

                obj_path = os.path.join(res_path, obj_folder, 'body_mesh')
                obj_list = os.listdir(obj_path)
                obj_list.sort(key=lambda x: int(x[:-4]))
                save_path = os.path.join(time_dict[key_time]['path'], 'mosh', 'obj')
                # save_path = os.path.join(time_dict[key_time]['path'], 'mosh', 'pkl')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                print(len(obj_list), len(time_dict[key_time]['opti_npz_list']))
                for i, obj in enumerate(obj_list):
                    shutil.copyfile(os.path.join(obj_path, obj), os.path.join(save_path, time_dict[key_time]['opti_npz_list'][i][:-4]+'.obj'))
            else:
                print(obj_folder, '++', time_dict[key_time]['path'])
                print(value_time, '++', key_time)

if __name__ == '__main__':
    root_path = '/home/nesc525/drivers/4/chen/moshpp'
    result_path = os.path.join(root_path, 'mosh_results')
    mesh_path = os.path.join(root_path, 'mesh')
    # for res in os.listdir(result_path):
    #     map_timestamp(res)
    target = 'SOMA_manulabeled_mpc_perple9'
    pkl_path = os.path.join(result_path, target)
    obj_path = os.path.join(mesh_path, target)
    map_timestamp(pkl_path)
    map_timestamp(obj_path)