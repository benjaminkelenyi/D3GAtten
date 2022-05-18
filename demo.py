import os
from pydoc import visiblename
import open3d as o3d
import argparse
import json
import logging
import torch
import copy
import numpy as np
from multiprocessing import Process, Manager
from easydict import EasyDict as edict
from models.architectures import KPFCNN
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_keypts, get_desc, get_scores, loadlog, build_correspondence

path = '/home/benji/projects/D3Feat.pytorch'


def execute_global_registration(src_keypts, tgt_keypts, src_desc, tgt_desc, distance_threshold):
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        src_keypts, tgt_keypts, src_desc, tgt_desc,
        distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def draw_registration_result(source, target, transformation, path_to_save, name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)

    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    o3d.visualization.draw_geometries(
        [source_temp, target_temp], "Registration results")
    o3d.io.write_point_cloud(
        f'{path_to_save}/{name}.pcd', source_temp + target_temp)


def register_one_scene(distance_threshold, save_path, scene):
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'{path}/geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(
        pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"

            source_keypts = get_keypts(keyptspath, cloud_bin_s)
            target_keypts = get_keypts(keyptspath, cloud_bin_t)
            source_desc = get_desc(descpath, cloud_bin_s, 'D3Feat')
            target_desc = get_desc(descpath, cloud_bin_t, 'D3Feat')
            source_score = get_scores(
                scorepath, cloud_bin_s, 'D3Feat').squeeze()
            target_score = get_scores(
                scorepath, cloud_bin_t, 'D3Feat').squeeze()
            source_desc = np.nan_to_num(source_desc)
            target_desc = np.nan_to_num(target_desc)

            # randomly select 5000 keypts
            if args.random_points:
                source_indices = np.random.choice(
                    range(source_keypts.shape[0]), args.num_points)
                target_indices = np.random.choice(
                    range(target_keypts.shape[0]), args.num_points)
            else:
                source_indices = np.argsort(
                    source_score)[-args.num_points:]
                target_indices = np.argsort(
                    target_score)[-args.num_points:]

                src_pts = source_keypts
                target_pts = target_keypts

                source_keypts = source_keypts[source_indices, :]
                source_desc = source_desc[source_indices, :]
                target_keypts = target_keypts[target_indices, :]
                target_desc = target_desc[target_indices, :]

                corr = build_correspondence(source_desc, target_desc)

                gt_trans = gtLog[key]
                frag1 = source_keypts[corr[:, 0]]
                frag2_pc = o3d.geometry.PointCloud()
                frag2_pc.points = o3d.utility.Vector3dVector(
                    target_keypts[corr[:, 1]])
                frag2_pc.transform(gt_trans)
                frag2 = np.asarray(frag2_pc.points)
                distance = np.sqrt(
                    np.sum(np.power(frag1 - frag2, 2), axis=1))

                src_features = o3d.registration.Feature()
                src_features.data = source_desc.T

                tgt_features = o3d.registration.Feature()
                tgt_features.data = target_desc.T

                ind_ok = []
                ind_nok = []

                for i in range(len(distance)):
                    if distance[i] < distance_threshold:
                        ind_ok.append(i)
                    else:
                        ind_nok.append(i)

                pcd_s = o3d.geometry.PointCloud()
                pcd_s.points = o3d.utility.Vector3dVector(src_pts)
                o3d.geometry.PointCloud.estimate_normals(pcd_s)
                pcd_s.paint_uniform_color([1, 0.706, 0])

                pcd_t = o3d.geometry.PointCloud()
                pcd_t.points = o3d.utility.Vector3dVector(target_pts)
                o3d.geometry.PointCloud.estimate_normals(pcd_t)
                pcd_t.paint_uniform_color([0, 0.651, 0.929])

                src_pts_s_ind_ok = o3d.geometry.PointCloud()
                src_pts_s_ind_ok.points = o3d.utility.Vector3dVector(
                    src_pts[corr[:, 0]][ind_ok])
                o3d.geometry.PointCloud.estimate_normals(src_pts_s_ind_ok)
                src_pts_s_ind_ok.paint_uniform_color([0, 1, 0])

                src_pts_t_ind_ok = o3d.geometry.PointCloud()
                src_pts_t_ind_ok.points = o3d.utility.Vector3dVector(
                    target_pts[corr[:, 1]][ind_ok])
                o3d.geometry.PointCloud.estimate_normals(src_pts_t_ind_ok)
                src_pts_t_ind_ok.paint_uniform_color([0, 1, 0])

                src_pts_s_ind_nok = o3d.geometry.PointCloud()
                src_pts_s_ind_nok.points = o3d.utility.Vector3dVector(
                    src_pts[corr[:, 0]][ind_nok])
                o3d.geometry.PointCloud.estimate_normals(src_pts_s_ind_nok)
                src_pts_s_ind_nok.paint_uniform_color([1, 0, 0])

                src_pts_t_ind_nok = o3d.geometry.PointCloud()
                src_pts_t_ind_nok.points = o3d.utility.Vector3dVector(
                    target_pts[corr[:, 1]][ind_nok])
                o3d.geometry.PointCloud.estimate_normals(src_pts_t_ind_nok)
                src_pts_t_ind_nok.paint_uniform_color([1, 0, 0])

                src_pcd = o3d.geometry.PointCloud()
                src_pcd.points = o3d.utility.Vector3dVector(source_keypts)

                tgt_pcd = o3d.geometry.PointCloud()
                tgt_pcd.points = o3d.utility.Vector3dVector(target_keypts)

                result_ransac = execute_global_registration(
                    src_pcd, tgt_pcd, src_features, tgt_features, 0.05)

                o3d.visualization.draw_geometries([pcd_s], "Input 1")
                o3d.visualization.draw_geometries([pcd_t], "Input 2")

                path_reg = f'{path}/results/registration/{scene}'
                path_dev = f'{path}/results/deviation/{scene}'
                path_pcd = f'{path}/results/pcd/{scene}'
                path_corr = f'{path}/results/corr/{scene}'
                name = f'{id1}+{id2}'

                if not os.path.exists(path_dev):
                    os.mkdir(path_dev)

                if not os.path.exists(path_pcd):
                    os.mkdir(path_pcd)

                if not os.path.exists(path_corr):
                    os.mkdir(path_corr)

                draw_registration_result(
                    pcd_s, pcd_t, result_ransac.transformation, path_reg, name)
                o3d.io.write_point_cloud(
                    f'{path_dev}/{name}.pcd', pcd_s + pcd_t)
                o3d.io.write_point_cloud(f'{path_pcd}/{id1}.pcd', pcd_s)
                o3d.io.write_point_cloud(f'{path_pcd}/{id2}.pcd', pcd_t)


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def generate_features(model, dloader, config, chosen_snapshot):
    dataloader_iter = dloader.__iter__()

    descriptor_path = f'{save_path}/descriptors'
    keypoint_path = f'{save_path}/keypoints'
    score_path = f'{save_path}/scores'
    if not os.path.exists(descriptor_path):
        os.mkdir(descriptor_path)
    if not os.path.exists(keypoint_path):
        os.mkdir(keypoint_path)
    if not os.path.exists(score_path):
        os.mkdir(score_path)

    # generate descriptors
    for scene in dset.scene_list:
        descriptor_path_scene = os.path.join(descriptor_path, scene)
        keypoint_path_scene = os.path.join(keypoint_path, scene)
        score_path_scene = os.path.join(score_path, scene)
        if not os.path.exists(descriptor_path_scene):
            os.mkdir(descriptor_path_scene)
        if not os.path.exists(keypoint_path_scene):
            os.mkdir(keypoint_path_scene)
        if not os.path.exists(score_path_scene):
            os.mkdir(score_path_scene)
        pcdpath = f"{config.root}/fragments/{scene}/"
        num_frag = len([filename for filename in os.listdir(
            pcdpath) if filename.endswith('ply')])
        # generate descriptors for each fragment
        for ids in range(num_frag):
            inputs = dataloader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()
            features, scores = model(inputs)
            pcd_size = inputs['stack_lengths'][0][0]
            pts = inputs['points'][0][:int(pcd_size)]
            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]

            # scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat',
                    features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}',
                    pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}',
                    scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot',
                        default='D3Feat03041342', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.20, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--generate_features',
                        default=False, action='store_true')
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='w',
                        format="")

    config_path = f'{path}/data/D3Feat/snapshot/{args.chosen_snapshot}/config.json'
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    # create model
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')

    model = KPFCNN(config)
    model.load_state_dict(torch.load(
        f'{path}/data/D3Feat/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'], strict=False)
    print(
        f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")

    model.eval()

    save_path = f'{path}/geometric_registration/{args.chosen_snapshot}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if args.generate_features:
        dset = ThreeDMatchTestset(root=config.root,
                                  downsample=config.downsample,
                                  config=config,
                                  last_scene=False,
                                  )
        dloader, _ = get_dataloader(dataset=dset,
                                    batch_size=config.batch_size,
                                    shuffle=False,
                                    num_workers=config.num_workers,
                                    )
        generate_features(model.cuda(), dloader, config, args.chosen_snapshot)

    # register each pair of fragments in scenes using multiprocessing.
    scene_list = ['own']

    jobs = []
    for scene in scene_list:
        p = Process(target=register_one_scene, args=(
            args.distance_threshold, save_path, scene))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
