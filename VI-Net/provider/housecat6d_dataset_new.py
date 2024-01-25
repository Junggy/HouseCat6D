import os
import math
import cv2
# import open3d as o3d
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image
# from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.data_utils import (
    load_housecat_depth,
    load_composed_depth,
    get_bbox,
    fill_missing,
    get_bbox_from_mask,
    rgb_add_noise,
    random_rotate,
    random_scale,
)

class HouseCat6DTrainingDataset(Dataset):
    def __init__(self,
            config, 
            dataset='housecat',
            mode='ts',
            resolution=64,
            ds_rate=2,
            seq_length=-1, # -1 means full
            img_length=-1, # -1 means full
    ):
        self.config = config
        self.dataset = dataset
        self.mode = mode
        self.resolution = resolution
        self.ds_rate = ds_rate
        self.sample_num = self.config.sample_num
        self.data_dir = config.data_dir
        self.train_scenes_rgb = glob.glob(os.path.join(self.data_dir,'scene*','rgb'))
        self.train_scenes_rgb.sort()
        self.train_scenes_rgb = self.train_scenes_rgb[:seq_length] if seq_length != -1 else self.train_scenes_rgb[:]
        self.real_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.train_scenes_rgb]
        self.meta_list = [os.path.join(scene, '..', 'meta.txt') for scene in self.train_scenes_rgb]
        self.min_num = 100
        for meta in self.meta_list:
            with open(meta, 'r') as file:
                content = file.read()
                num_count = content.count('\n') + 1
            self.min_num = num_count if num_count < self.min_num else self.min_num
        self.real_img_list = []
        for scene in self.train_scenes_rgb:
            img_paths = glob.glob(os.path.join(scene, '*.png'))
            img_paths.sort()
            img_paths = img_paths[:img_length] if img_length != -1 else img_paths[:]
            for img_path in img_paths:
                self.real_img_list.append(img_path)

        print(f'{len(self.train_scenes_rgb)} sequences, {img_length} images per sequence. Total {len(self.real_img_list)} images are found.')

        self.xmap = np.array([[i for i in range(1096)] for j in range(852)]) # 640x480
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.reset()


    def __len__(self):
        return len(self.real_img_list)

        
    def reset(self):
        num_real_img = len(self.real_img_list)
        self.img_index = np.arange(num_real_img)
        np.random.shuffle(self.img_index)

    def __getitem__(self, index):
        image_index = self.img_index[index]
        data_dict = self._read_data(image_index)
        assert data_dict is not None
        return data_dict
        
    def _read_data(self, image_index):
        img_type = 'real'
        img_path = os.path.join(self.data_dir, self.real_img_list[image_index])
        pol_path = img_path.replace('rgb','pol')
        real_intrinsics = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3,3)
        cam_fx, cam_fy, cam_cx, cam_cy = real_intrinsics[0,0], real_intrinsics[1,1], real_intrinsics[0,2], real_intrinsics[1,2]


        depth_ = load_housecat_depth(img_path)
        depth_ = fill_missing(depth_, self.norm_scale, 1)


        # mask
        with open(img_path.replace('rgb','labels').replace('.png','_label.pkl'), 'rb') as f:
            gts = cPickle.load(f)
        num_instance = len(gts['instance_ids'])
        assert(len(gts['class_ids'])==len(gts['instance_ids']))
        mask_ = cv2.imread(img_path.replace('rgb','instance'))[:, :, 2] # TODO 1096x852

        # rgb
        rgb_ = cv2.imread(img_path)[:, :, :3]  # TODO 1096x852
        rgb_ = rgb_[:, :, ::-1]  # 480*640*3

        # pol
        pol_ = cv2.imread(pol_path)[:, :, :3]
        pol_ = pol_[:, :, ::-1]
        pol_ = pol_ / 255.0
        h, w = pol_.shape[0], pol_.shape[1]
        half_h, half_w = int(h / 2), int(w / 2)
        p1, p2, p3, p4 = pol_[:half_h, :half_w], pol_[:half_h, half_w:w], pol_[half_h:h, :half_w], pol_[half_h:h, half_w:w]
        intensity = (p1 + p2 + p3 + p4) / 4 + 1e-6
        intensity = np.linalg.norm(intensity, axis=-1)[:, :, np.newaxis]
        dop_ = np.linalg.norm(np.sqrt(np.square(p1 - p3) + np.square(p2 - p4)) / intensity, axis=-1)
        aop_ = np.arctan2(p2 - p4, p1 - p3)
        aop_ += (aop_ < 0) * np.pi
        aop_ = np.linalg.norm(aop_, axis=-1)

        data_list = []
        idxs = np.random.choice(np.arange(num_instance), self.min_num, replace=False)
        for idx in idxs:
            cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
            rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx], img_width=852, img_length=1096)
            mask = np.equal(mask_, gts['instance_ids'][idx])
            mask = np.logical_and(mask , depth_ > 0)

            # choose
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose)<=0:
                return None
            elif len(choose) <= self.sample_num:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.sample_num, replace=False)
            choose = choose[choose_idx]

            # pts
            pts2 = depth_.copy()[rmin:rmax, cmin:cmax].reshape((-1))[choose] / self.norm_scale
            pts0 = (self.xmap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cx) * pts2 / cam_fx
            pts1 = (self.ymap[rmin:rmax, cmin:cmax].reshape((-1))[choose] - cam_cy) * pts2 / cam_fy
            pts_ = np.transpose(np.stack([pts0, pts1, pts2]), (1,0)).astype(np.float32) # 480*640*3
            pts = pts_ + np.clip(0.001*np.random.randn(pts_.shape[0], 3), -0.005, 0.005)

            # vis = o3d.visualization.Visualizer()
            # vis.create_window(window_name="test")
            # vis.get_render_option().point_size = 1
            # opt = vis.get_render_option()
            # opt.background_color = np.asarray([0, 0, 0])
            # pcd = o3d.open3d.geometry.PointCloud()
            # pcd.points = o3d.open3d.utility.Vector3dVector(pts)
            # pcd.paint_uniform_color([1, 1, 1])
            # vis.add_geometry(pcd)
            # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0]))
            #
            # vis.run()
            # vis.destroy_window()

            rgb = rgb_[rmin:rmax, cmin:cmax, :]
            # cv2.imshow('image', rgb)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
            rgb = np.array(rgb)
            rgb = rgb.astype(np.float32).reshape((-1,3))[choose, :] / 255.0

            dop = dop_[rmin:rmax, cmin:cmax].astype(np.float32)
            aop = aop_[rmin:rmax, cmin:cmax].astype(np.float32)

            # import matplotlib
            # matplotlib.use('TkAgg')
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(dop)
            # plt.subplot(1,3,2)
            # plt.imshow(rgb_)
            # plt.subplot(1,3,3)
            # plt.imshow(aop)
            # plt.show()

            #fused image feature
            rgb_pol = np.concatenate((rgb,dop.reshape(-1,1)[choose, :],aop.reshape(-1,1)[choose, :]), axis=-1)

            # gt
            translation = gts['translations'][idx].astype(np.float32)
            rotation = gts['rotations'][idx].astype(np.float32)
            size = gts['gt_scales'][idx].astype(np.float32)


            if hasattr(self.config, 'random_rotate') and self.config.random_rotate:
                pts, rotation = random_rotate(pts, rotation, translation, self.config.angle_range)

            ret_dict = {}
            if self.mode == 'ts':
                pts, size = random_scale(pts, size, rotation, translation)

                center = np.mean(pts, axis=0)
                pts = pts - center[np.newaxis, :]
                translation = translation - center

                noise_t = np.random.uniform(-0.02, 0.02, 3)
                pts = pts + noise_t[None, :]
                translation = translation + noise_t

                ret_dict['pts'] = torch.FloatTensor(pts)
                ret_dict['rgb'] = torch.FloatTensor(rgb)
                ret_dict['rgb_pol'] = torch.FloatTensor(rgb_pol)
                ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
                ret_dict['translation_label'] = torch.FloatTensor(translation)
                ret_dict['size_label'] = torch.FloatTensor(size)

            else:
                noise_t = np.random.uniform(-0.02, 0.02, 3)
                noise_s = np.random.uniform(0.8, 1.2, 1)
                pts = pts - translation[None, :] - noise_t[None, :]
                pts = pts / np.linalg.norm(size) * noise_s

                ## for symmetrical objects
                # theta_x = rotation[0, 0] + rotation[2, 2]
                # theta_y = rotation[0, 2] - rotation[2, 0]
                # r_norm = math.sqrt(theta_x**2 + theta_y**2)
                # s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                #                     [0.0,            1.0,  0.0           ],
                #                     [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                # rotation = rotation @ s_map
                # asym_flag = 0.0

                asym_flag = 1.0

                # transform ZXY system to XYZ system
                rotation = rotation[:, (2,0,1)] #TODO

                v = rotation[:,2] / (np.linalg.norm(rotation[:,2])+1e-8)
                rho = np.arctan2(v[1], v[0])
                if v[1]<0:
                    rho += 2*np.pi
                phi = np.arccos(v[2])

                vp_rotation = np.array([
                    [np.cos(rho),-np.sin(rho),0],
                    [np.sin(rho), np.cos(rho),0],
                    [0,0,1]
                ]) @ np.array([
                    [np.cos(phi),0,np.sin(phi)],
                    [0,1,0],
                    [-np.sin(phi),0,np.cos(phi)],
                ])
                ip_rotation = vp_rotation.T @ rotation

                rho_label = int(rho / (2*np.pi) * (self.resolution//self.ds_rate))
                phi_label = int(phi/np.pi*(self.resolution//self.ds_rate))

                ret_dict['rgb'] = torch.FloatTensor(rgb)
                ret_dict['rgb_pol'] = torch.FloatTensor(rgb_pol)
                ret_dict['pts'] = torch.FloatTensor(pts)
                ret_dict['category_label'] = torch.IntTensor([cat_id]).long()
                ret_dict['asym_flag'] = torch.FloatTensor([asym_flag])
                ret_dict['translation_label'] = torch.FloatTensor(translation)
                ret_dict['rotation_label'] = torch.FloatTensor(rotation)
                ret_dict['size_label'] = torch.FloatTensor(size)

                ret_dict['rho_label'] = torch.IntTensor([rho_label]).long()
                ret_dict['phi_label'] = torch.IntTensor([phi_label]).long()
                ret_dict['vp_rotation_label'] = torch.FloatTensor(vp_rotation)
                ret_dict['ip_rotation_label'] = torch.FloatTensor(ip_rotation)

            data_list.append(ret_dict)

        data_dict = {}

        for d in data_list:
            for key, value in d.items():
                value = value.unsqueeze(0)
                if key in data_dict:
                    data_dict[key] = torch.cat((data_dict[key], value), 0)
                else:
                    data_dict[key] = value

        return data_dict

    def collate_fn(self, batch):
        out = {}
        out['pts'] = []
        out['rgb'] = []
        out['rgb_pol'] = []
        out['category_label'] = []
        out['translation_label'] = []
        out['size_label'] = []
        if self.mode == 'r':
            out['asym_flag'] = []
            out['rotation_label'] = []
            out['rho_label'] = []
            out['phi_label'] = []
            out['vp_rotation_label'] = []
            out['ip_rotation_label'] = []
        for i in range(len(batch)):
            out['pts'].append(batch[i]['pts'])
            out['rgb'].append(batch[i]['rgb'])
            out['rgb_pol'].append(batch[i]['rgb_pol'])
            out['category_label'].append(batch[i]['category_label'])
            out['translation_label'].append(batch[i]['translation_label'])
            out['size_label'].append(batch[i]['size_label'])
            if self.mode == 'r':
                out['asym_flag'].append(batch[i]['asym_flag'])
                out['rotation_label'].append(batch[i]['rotation_label'])
                out['rho_label'].append(batch[i]['rho_label'])
                out['phi_label'].append(batch[i]['phi_label'])
                out['vp_rotation_label'].append(batch[i]['vp_rotation_label'])
                out['ip_rotation_label'].append(batch[i]['ip_rotation_label'])

        for key, value in out.items():
            out[key] = torch.cat(value)
        return out

    


class HouseCat6DTestDataset():
    def __init__(self, config, dataset='housecat', resolution=64):
        self.dataset = dataset
        self.resolution = resolution
        self.data_dir = config.data_dir
        self.sample_num = config.sample_num
        self.test_scenes_rgb = glob.glob(os.path.join(self.data_dir, 'test_scene*', 'rgb'))
        self.test_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.test_scenes_rgb]
        self.test_img_list = [img_path for scene in self.test_scenes_rgb for img_path in
                              glob.glob(os.path.join(scene, '*.png'))]

        n_image = len(self.test_img_list)
        print('no. of test images: {}\n'.format(n_image))

        self.xmap = np.array([[i for i in range(1096)] for j in range(852)])
        self.ymap = np.array([[j for i in range(1096)] for j in range(852)])
        self.norm_scale = 1000.0    # normalization scale

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, index):
        img_path = self.test_img_list[index]
        pol_path = img_path.replace('rgb', 'pol')
        with open(img_path.replace('rgb', 'labels').replace('.png', '_label.pkl'), 'rb') as f:
            gts = cPickle.load(f)

        # pred_mask = pred_data['pred_masks']
        mask_path = img_path.replace("rgb", "instance")
        mask = cv2.imread(mask_path)
        assert mask is not None
        mask = mask[:, :, 2]
        num_instance = len(gts['class_ids'])

        # rgb
        rgb = cv2.imread(img_path)[:, :, :3] # TODO 1096x852
        rgb = rgb[:, :, ::-1] #480*640*3

        # pol
        pol = cv2.imread(pol_path)[:, :, :3]
        pol = pol[:, :, ::-1]
        h,w = pol.shape[0], pol.shape[1]
        half_h, half_w = int(h / 2), int(w / 2)
        p1, p2, p3, p4 = pol[:half_h, :half_w], pol[:half_h, half_w:w], pol[half_h:h, :half_w], pol[half_h:h, half_w:w]
        intensity = (p1 + p2 + p3 + p4) / 4 + 1e-6
        # intensity = np.linalg.norm(intensity, axis=-1)
        dop = np.linalg.norm(np.sqrt(np.square(p1 - p3) + np.square(p2 - p4)) / intensity, axis=-1)
        aop = np.arctan2(p2 - p4, p1 - p3)
        aop += (aop < 0) * np.pi
        aop = np.linalg.norm(aop, axis=-1)
        pol_info = np.concatenate((dop[:, :, np.newaxis],dop[:, :, np.newaxis]), axis=-1)

        # pts
        intrinsics = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3, 3)
        cam_fx, cam_fy, cam_cx, cam_cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        depth = load_housecat_depth(img_path) #480*640 # TODO 1096x852
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3

        all_rgb = []
        all_rgb_pol = []
        all_pts = []
        all_center = []
        all_cat_ids = []
        flag_instance = torch.zeros(num_instance) == 1
        mask_target = mask.copy().astype(np.float32)
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(window_name="test")

        for j in range(num_instance):
            mask = np.equal(mask_target, gts['instance_ids'][j])
            inst_mask = 255 * mask.astype('uint8')
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            if np.sum(mask) > 16:
                rmin, rmax, cmin, cmax = get_bbox_from_mask(mask, img_width = 852, img_length = 1096)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                cat_id = gts['class_ids'][j] - 1 # convert to 0-indexed
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :].copy()


                # vis.get_render_option().point_size = 1
                # opt = vis.get_render_option()
                # opt.background_color = np.asarray([0, 0, 0])
                # pcd = o3d.open3d.geometry.PointCloud()
                # pcd.points = o3d.open3d.utility.Vector3dVector(instance_pts)
                # pcd.paint_uniform_color([1, 1, 1])
                # vis.add_geometry(pcd)
                # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0]))
                # # View Controll
                # ctr = vis.get_view_control()
                # ctr.set_front([0, 0, -1])
                # ctr.set_up([0, -1, 0])
                # # Updates
                # vis.update_geometry(pcd)
                # vis.poll_events()
                # vis.update_renderer()

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = np.array(instance_rgb).astype(np.float32).reshape((-1, 3))[choose, :] / 255.0

                instance_pol_info = pol_info[rmin:rmax, cmin:cmax, :].copy()

                center = np.mean(instance_pts, axis=0)
                instance_pts = instance_pts - center[np.newaxis, :]

                if instance_pts.shape[0] <= self.sample_num:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num)
                else:
                    choose_idx = np.random.choice(np.arange(instance_pts.shape[0]), self.sample_num, replace=False)
                instance_pts = instance_pts[choose_idx, :]
                instance_rgb = instance_rgb[choose_idx, :]
                instance_pol_info = instance_pol_info.reshape(-1,2)[choose_idx, :]
                instance_rgb_pol = np.concatenate((instance_rgb,instance_pol_info),axis=-1)

                # cv2.imshow('image', rgb[rmin:rmax, cmin:cmax, :])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_rgb_pol.append(torch.FloatTensor(instance_rgb_pol))
                all_center.append(torch.FloatTensor(center))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                flag_instance[j] = 1
        # vis.destroy_window()

        ret_dict = {}
        RTs = []
        for each_idx in range(len(gts["rotations"])):
            matrix = np.identity(4)
            matrix[:3, :3] = gts["rotations"][each_idx]
            matrix[:3, 3] = gts["translations"][each_idx]
            RTs.append(matrix)
        RTs = np.stack(RTs, 0)
        ret_dict['gt_class_ids'] = torch.tensor(gts['class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(gts["bboxes"])
        ret_dict['gt_RTs'] = torch.tensor(RTs)
        ret_dict['gt_scales'] = torch.tensor(gts["gt_scales"])
        ret_dict['index'] = index

        if len(all_pts) == 0:
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))
        else:
            ret_dict['pts'] = torch.stack(all_pts) # N*3
            ret_dict['rgb'] = torch.stack(all_rgb)
            ret_dict['rgb_pol'] = torch.stack(all_rgb_pol)
            ret_dict['center'] = torch.stack(all_center)
            ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])[flag_instance==1]
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])[flag_instance==1]
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))[flag_instance==1]

        return ret_dict

if __name__ == "__main__":
    import gorilla
    cfg = gorilla.Config.fromfile('../config/housecat.yaml')
    house = HouseCat6DTrainingDataset(cfg.train_dataset,
        'housecat',
        'r',
        resolution = cfg.resolution,
        ds_rate = cfg.ds_rate)
    print(len(house))
    a = house[0]

    house = HouseCat6DTestDataset(cfg.test,
                                      'housecat',
                                      resolution=cfg.resolution,
                                      )
    b = house[0]