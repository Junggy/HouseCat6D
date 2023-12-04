import argparse
import pickle
import cv2
import os
import glob
import numpy as np
import pyrender
import trimesh
import h5py
import matplotlib.pyplot as plt

# id_base = [1,2,3,4,5,9,19,21,22,23,24,25,33,34,'test_5','val_1']
parser = argparse.ArgumentParser(description='Grasp visualization')
parser.add_argument('--split', type=str, help='train/val/test', default='train')
parser.add_argument('--scene', type=int, help='scene id', default=19)
parser.add_argument('--dimentional', type=str, help='2D/3D', default='3D')
parser.add_argument('--downsample', action='store_true', help='Downsample?')

args = parser.parse_args()

sequence_id = args.scene
option = args.dimentional
downsample_3D = (args.downsample and option == '3D')
downsample_2D = (args.downsample and option == '2D')
if args.split == 'train':
    assert sequence_id in [1,2,3,4,5,9,19,21,22,23,24,25,33,34]
    dataset_path = f"../scene{sequence_id:02d}"
elif args.split == 'val':
    assert sequence_id == 1
    dataset_path = f"../val_scene{sequence_id}"
else:
    assert sequence_id == 5
    dataset_path = f"../test_scene{sequence_id}"

cam_name = 'rgb'
grasp_base_path = os.path.join(dataset_path, "grasps", "grasps_info_base.h5")
assert os.path.exists(grasp_base_path)
grasp_frame_path = os.path.join(dataset_path, "grasps", 'polarization')
assert os.path.exists(grasp_frame_path)

def create_panda_marker(color=[0, 0, 255], tube_radius=0.002, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=tube_radius, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=tube_radius,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    # z axis to x axis
    # R = np.array([[0,0,1],[1,0,0],[0,1,0]]).reshape(3,3)
    # t =  np.array([0, 0, -1.12169998e-01]).reshape(3,1)
    #
    # T = np.r_[np.c_[np.eye(3), t], [[0, 0, 0, 1]]]
    # tmp.apply_transform(T)

    return tmp

def points2pixels(points, cam_matrix):
    """

    Parameters
    ----------
    points N * 3
    cam_matrix

    Returns
    pixel_coordinates on image N * 2 (WxH)
    -------

    """
    points = points.reshape(-1, 3).transpose()
    points /= points[2]
    projection = cam_matrix.dot(points)
    pixel_coordinates = np.round(projection.T, 0)[:, :2].astype('int')
    return pixel_coordinates

def plot_grasp(sequence_id, idx, img, grasps, K, downsample_2D=False, downsample_size = 300, visual=False):
    grasps_total = 0
    grasps_show = 0
    for key in grasps.keys():
        T = grasps[key]['grasps']
        s = np.array(grasps[key]['success']).reshape(-1)
        if T.shape[0] == 0:
            continue
        grasps_total += len(T)
        if downsample_2D:
            np.random.seed(10)
            slices = np.random.choice(np.arange(T.shape[0]), size=downsample_size, replace=False) if T.shape[0] >= downsample_size else np.arange(T.shape[0])
            T = T[slices]
            s = s[slices]
            grasps_show += len(slices)
        else:
            grasps_show = grasps_total

        b1 = np.array([0, 0, 0, 1]).reshape(-1, 1)
        b2 = np.array([0, 0, 6.59999996e-02, 1]).reshape(-1, 1)
        lb = np.array([4.10000000e-02, -7.27595772e-12, 6.59999996e-02, 1]).reshape(-1, 1)
        lt = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01, 1]).reshape(-1, 1)
        rb = np.array([-4.100000e-02, -7.27595772e-12, 6.59999996e-02, 1]).reshape(-1, 1)
        rt = np.array([-4.100000e-02, -7.27595772e-12, 1.12169998e-01, 1]).reshape(-1, 1)
        gripper_points = np.c_[b1, b2, lb, lt, rb, rt]  # 4*6
        gripper_points_cam = np.matmul(T, gripper_points)  # N*4*6
        for i in range(T.shape[0]):
            color = (int((1-s[i])), int(s[i]), 0)
            pixels = points2pixels(gripper_points_cam[i, 0:3, :].T, K)  # 4*4
            cv2.line(img, pixels[0], pixels[1], color, 1)
            cv2.line(img, pixels[2], pixels[3], color, 1)
            cv2.line(img, pixels[4], pixels[5], color, 1)
            cv2.line(img, pixels[2], pixels[4], color, 1)
            cv2.line(img, pixels[4], pixels[5], color, 1)
    print("show ", grasps_show, " grasps out of totally ", grasps_total, "grasps.")
    if visual:
        cv2.imshow('2D', img[:,:,::-1])
        cv2.waitKey(0)
        cv2.destroyWindow('2D')
    if img.max() <= 1:
        img = (img * 255).astype(np.float32)
    cv2.imwrite('scene_{0}_{1}.jpg'.format(str(sequence_id), str(idx)), img[:,:,::-1])
    cv2.imwrite('scene_{0}_{1}.jpg'.format(str(sequence_id), str(idx)), img[:, :, ::-1])
    return img[:,:,::-1]

def obj_info_frame(anno_path, demo=False):
    with open(anno_path, 'rb') as f:
        anno_frame = pickle.load(f)

    obj_names = anno_frame['model_list']
    obj_dict = {}
    for obj_idx in range(len(obj_names)):
        cam_R_m2c = np.array(anno_frame['rotations'][obj_idx])
        cam_t_m2c = np.array(anno_frame['translations'][obj_idx])
        cam_m2c = np.identity(4)
        cam_m2c[:3, :3] = cam_R_m2c
        cam_m2c[:3, 3] = cam_t_m2c
        class_ = anno_frame['class_ids'][obj_idx]
        instance_ = anno_frame['instance_ids'][obj_idx]
        obj_name = obj_names[obj_idx]
        class_name = obj_names[obj_idx].split('-')[0] if demo==False else 'demo'
        trimesh_obj = trimesh.load(os.path.join(dataset_path, '../', 'obj_models_small_size_final', class_name, obj_name+'.obj'))
        #trimesh_obj.show()

        if class_name in ['glass', 'cutlery']:
            trimesh_obj.visual = trimesh.visual.ColorVisuals()
            #trimesh_obj.show()
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        obj_dict[obj_name] = {"pose_cam": cam_m2c, "trimesh_mesh": trimesh_obj, "pyrender_mesh": mesh}

    return obj_dict

scene_gt_path = os.path.join(dataset_path,'labels')
label_files = glob.glob(os.path.join(scene_gt_path,'*.pkl'))
label_files.sort()
target_img_path = os.path.join(dataset_path,cam_name)
img_files = glob.glob(os.path.join(target_img_path,'*.png'))
img_files.sort()

real_intrinsics = np.loadtxt(os.path.join(dataset_path, 'intrinsics.txt')).reshape(3, 3)
cam = np.eye(3)
cam_fx, cam_fy, cam_cx, cam_cy = real_intrinsics[0, 0], real_intrinsics[1, 1], real_intrinsics[0, 2], real_intrinsics[1, 2]
cam[0, 0] = cam_fx
cam[1, 1] = cam_fy
cam[0, 2] = cam_cx
cam[1, 2] = cam_cy

scene = pyrender.Scene(bg_color=[0, 0, 0], ambient_light=[1, 1, 1])
for idx in range(len(label_files)):
    all_trimeshes = []
    grasp_trimeshes = []
    grasps_dict = {}
    grasp_meshes_dict = {}
    grasps_dict_cam = {}
    img = plt.imread(img_files[idx])[:,:,:3]
    h, w, _ = img.shape
    if img.max() > 1:
        img = (img / 255).astype(np.float32)

    obj_dict = obj_info_frame(label_files[idx],demo=False)

    data = h5py.File(os.path.join(grasp_frame_path, '%06d.h5' % idx), "r")
    for each_name in obj_dict.keys():
        each_key = each_name.replace('-', '_')
        if data[each_key]["transforms"].shape[0]:
            grasps_cam = np.array(data[each_key]["transforms"])
            success = np.array(data[each_key]["qualities/object_in_gripper"])

            grasps_dict_cam[each_key] = {"grasps": grasps_cam, "success": success}
        else:
            grasps_dict_cam[each_key] = {"grasps": np.array([]), "success": np.array([])}
        if option == '3D':
            obj_node = pyrender.Node(mesh=obj_dict[each_name]["pyrender_mesh"], name=each_key)
            scene.add_node(obj_node)
            scene.set_pose(obj_node, obj_dict[each_name]["pose_cam"])
            all_trimeshes.append(obj_dict[each_name]["trimesh_mesh"].apply_transform(obj_dict[each_name]["pose_cam"]))
            if downsample_3D:
                np.random.seed(10)
                slices = np.random.choice(np.arange(grasps_dict_cam[each_key]["grasps"].shape[0]), size=20, replace=False) if grasps_dict_cam[each_key]["grasps"].shape[0] >= 20 else np.arange(grasps_dict_cam[each_key]["grasps"].shape[0])
                grasps_dict_cam[each_key]["grasps"] = grasps_dict_cam[each_key]["grasps"][slices]
                grasps_dict_cam[each_key]["success"] = grasps_dict_cam[each_key]["success"][slices]
            grasp_meshes_dict[each_key] = [create_panda_marker([255 * (1 - grasps_dict_cam[each_key]["success"][i][0]),
                                                                    255 * grasps_dict_cam[each_key]["success"][i][0], 0],
                                                                   tube_radius=0.0005).apply_transform(grasps_dict_cam[each_key]["grasps"][i])
                                               for i in range(grasps_dict_cam[each_key]["grasps"].shape[0])]
            grasp_trimeshes.append(grasp_meshes_dict[each_key])

    #all_trimeshes[0].show()

    grasp_trimeshes = sum(grasp_trimeshes, [])

    if option == '2D':
        print('saving', 'scene_{0}_{1}.jpg'.format(str(sequence_id), str(idx)))
        plot_grasp(sequence_id, idx, img.copy(), grasps_dict_cam, cam, downsample_2D=downsample_2D, downsample_size = 5, visual=False)
    else:

        entire_scene = trimesh.Scene([all_trimeshes + grasp_trimeshes])
        print(len(grasp_trimeshes))
        entire_scene.show()
        entire_scene.export('scene_{}.glb'.format(str(sequence_id)))
        break



