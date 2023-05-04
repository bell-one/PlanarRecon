# This file is derived from [NeuralRecon](https://github.com/zju3dv/NeuralRecon).
# Originating Author: Yiming Xie, Jiaming Sun

# Original header:
# Copyright SenseTime. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
from tqdm import tqdm
import sys
from scipy.spatial.transform import Rotation as R
sys.path.append('.')

from tools.kp_reproject import *
from tools.sync_poses import *
from tools.colmap_read_model import read_cameras_binary, read_images_binary, read_points3d_binary


# params
project_path = '../room'


def colmap_images_to_pose(img_vals):
    # colmap quat has (w, x, y, z) and scipy needs (x, y, z, w)
    qvec = [0, 0, 0, 0]
    qvec[3] = img_vals.qvec[0]
    qvec[:3] = img_vals.qvec[1:]
    rot_mat = R.from_quat(qvec).as_matrix()
    cam_pose = np.eye(4)
    cam_pose[:3, :3] = rot_mat
    cam_pose[:3, 3] = img_vals.tvec.T

    return cam_pose

def colmap_cames_to_intrinsics(cam_vals, ori_size=(4032, 3024), size=(640, 480)):
    cam_intrinsic = np.eye(3)
    cam_intrinsic[0, 2] = cam_vals.params[1]  / (ori_size[0] / size[0])
    cam_intrinsic[1, 2] = cam_vals.params[2]  / (ori_size[1] / size[1])
    cam_intrinsic[0, 0] = cam_vals.params[0]  / (ori_size[0] / size[0])
    cam_intrinsic[1, 1] = cam_vals.params[0]  / (ori_size[1] / size[1])

    return cam_intrinsic

def process_data(data_path, data_source='COLMAP', window_size=9, min_angle=15, min_distance=0.1, ori_size=(4032, 3024), size=(640, 480)):
    image_path = os.path.join(data_path, 'images')

    # load intrin and extrin
    print('Load intrinsics and extrinsics')
    cam_intrinsic_dict = read_cameras_binary(os.path.join(data_path, "sparse/0/cameras.bin"))
    cam_pose_dict = read_images_binary(os.path.join(data_path, "sparse/0/images.bin"))
    # generate fragment
    fragments = []

    all_ids = []
    ids = []
    count = 0
    last_pose = None

    for id in tqdm(cam_pose_dict.keys()):
        img_vals = cam_pose_dict[id]
        cam_pose = colmap_images_to_pose(img_vals)

        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (min_angle / 180) * np.pi or dis > min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                count += 1
                if count == window_size:
                    all_ids.append(ids)
                    ids = []
                    count = 0




    # save fragments
    init = False

    for i, ids in enumerate(tqdm(all_ids, desc='Saving fragments file...')):
        poses = []
        intrinsics = []

        path = []
        for id in ids:
            # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
            if not init:
                trans = cam_pose_dict[id].tvec - [0, 0, 1.5]
                init = True
            img_vals = cam_pose_dict[id]
            cam_pose = colmap_images_to_pose(img_vals)
            cam_pose[:3, 3] = (img_vals.tvec - trans).T ## moving X-Y plane?
            poses.append(cam_pose)

            cam_intrinsic_vals = cam_intrinsic_dict[cam_pose_dict[id].camera_id]
            cam_intrinsic = colmap_cames_to_intrinsics(cam_intrinsic_vals, ori_size, size)

            intrinsics.append(cam_intrinsic)
            path.append(img_vals.name)
        fragments.append({
            'scene': data_path.split('/')[-1],
            'fragment_id': i,
            'image_ids': path,
            'extrinsics': poses,
            'intrinsics': intrinsics
        })

    with open(os.path.join(data_path, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

if __name__ == '__main__':
    process_data(project_path)