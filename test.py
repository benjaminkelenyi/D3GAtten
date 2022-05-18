import os
from pydoc import visiblename
import open3d as o3d
import argparse
import json
import importlib
import logging
import torch
import copy
import numpy as np
from multiprocessing import Process, Manager
from functools import partial
from easydict import EasyDict as edict
from utils.pointcloud import make_point_cloud
from models.architectures import KPFCNN
from utils.timer import Timer, AverageMeter
from datasets.ThreeDMatch import ThreeDMatchTestset
from datasets.dataloader import get_dataloader
from geometric_registration.common import get_pcd, get_keypts, get_desc, get_scores, loadlog, build_correspondence
from utils.vis_TSNE import get_colored_point_cloud_feature

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.architectures import KPCNN

import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import proj3d

global vis
vis = True

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

    o3d.visualization.draw_geometries([source_temp, target_temp], "reg_res")
    o3d.io.write_point_cloud(f'{path_to_save}/{name}.pcd', source_temp + target_temp)


def register_one_scene(inlier_ratio_threshold, distance_threshold, save_path, return_dict, scene):
    gt_matches = 0
    pred_matches = 0
    keyptspath = f"{save_path}/keypoints/{scene}"
    descpath = f"{save_path}/descriptors/{scene}"
    scorepath = f"{save_path}/scores/{scene}"
    gtpath = f'/home/benji/projects/D3Feat.pytorch/geometric_registration/gt_result/{scene}-evaluation/'
    gtLog = loadlog(gtpath)
    inlier_num_meter, inlier_ratio_meter = AverageMeter(), AverageMeter()
    pcdpath = f"{config.root}/fragments/{scene}/"
    num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
    for id1 in range(num_frag):
        for id2 in range(id1 + 1, num_frag):
            cloud_bin_s = f'cloud_bin_{id1}'
            cloud_bin_t = f'cloud_bin_{id2}'
            key = f"{id1}_{id2}"
            if key not in gtLog.keys():
                # skip the pairs that have less than 30% overlap.
                num_inliers = 0
                inlier_ratio = 0
                gt_flag = 0
            else:
                source_keypts = get_keypts(keyptspath, cloud_bin_s)
                target_keypts = get_keypts(keyptspath, cloud_bin_t)
                source_desc = get_desc(descpath, cloud_bin_s, 'D3Feat')
                target_desc = get_desc(descpath, cloud_bin_t, 'D3Feat')
                source_score = get_scores(scorepath, cloud_bin_s, 'D3Feat').squeeze()
                target_score = get_scores(scorepath, cloud_bin_t, 'D3Feat').squeeze()
                source_desc = np.nan_to_num(source_desc)
                target_desc = np.nan_to_num(target_desc)

                # randomly select 5000 keypts
                if args.random_points:
                    source_indices = np.random.choice(range(source_keypts.shape[0]), args.num_points)
                    target_indices = np.random.choice(range(target_keypts.shape[0]), args.num_points)
                else:
                    source_indices = np.argsort(source_score)[-args.num_points:]
                    target_indices = np.argsort(target_score)[-args.num_points:]

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
                    frag2_pc.points = o3d.utility.Vector3dVector(target_keypts[corr[:, 1]])
                    frag2_pc.transform(gt_trans)
                    frag2 = np.asarray(frag2_pc.points)
                    distance = np.sqrt(np.sum(np.power(frag1 - frag2, 2), axis=1))
                
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
                    src_pts_s_ind_ok.points = o3d.utility.Vector3dVector(src_pts[corr[:, 0]][ind_ok])
                    o3d.geometry.PointCloud.estimate_normals(src_pts_s_ind_ok)
                    src_pts_s_ind_ok.paint_uniform_color([0, 1, 0])

                    src_pts_t_ind_ok = o3d.geometry.PointCloud()
                    src_pts_t_ind_ok.points = o3d.utility.Vector3dVector(target_pts[corr[:, 1]][ind_ok])
                    o3d.geometry.PointCloud.estimate_normals(src_pts_t_ind_ok)
                    src_pts_t_ind_ok.paint_uniform_color([0, 1, 0])

                    src_pts_s_ind_nok = o3d.geometry.PointCloud()
                    src_pts_s_ind_nok.points = o3d.utility.Vector3dVector(src_pts[corr[:, 0]][ind_nok])
                    o3d.geometry.PointCloud.estimate_normals(src_pts_s_ind_nok)
                    src_pts_s_ind_nok.paint_uniform_color([1, 0, 0])

                    src_pts_t_ind_nok = o3d.geometry.PointCloud()
                    src_pts_t_ind_nok.points = o3d.utility.Vector3dVector(target_pts[corr[:, 1]][ind_nok])
                    o3d.geometry.PointCloud.estimate_normals(src_pts_t_ind_nok)
                    src_pts_t_ind_nok.paint_uniform_color([1, 0, 0])

                    pcd_s_corr = src_pts_s_ind_ok + src_pts_s_ind_nok
                    pcd_t_corr = src_pts_t_ind_ok + src_pts_t_ind_nok

                    src_pcd = o3d.geometry.PointCloud()
                    src_pcd.points = o3d.utility.Vector3dVector(source_keypts)

                    tgt_pcd = o3d.geometry.PointCloud()
                    tgt_pcd.points = o3d.utility.Vector3dVector(target_keypts)

                    result_ransac = execute_global_registration(src_pcd, tgt_pcd, src_features, tgt_features, 0.05)
                    
                    import numpy
                    pcd_t_trans = pcd_t
                    pcd_corr = pcd_s + pcd_t_trans #translate([4,0,0])

                    lines_ok = np.zeros((len(corr[ind_ok]), 2))
                    lines_ok[:,0] = corr[:,0][ind_ok]
                    lines_ok[:,1] = corr[:, 1][ind_ok] + len(pcd_s.points)

                    lines_nok = np.zeros((len(corr[ind_nok]), 2))
                    lines_nok[:,0] = corr[:,0][ind_nok]
                    lines_nok[:,1] = corr[:, 1][ind_nok] + len(pcd_s.points)

                    colors_nok = [[1, 0, 0] for i in range(len(lines_nok))]
                    colors_ok = [[0, 1, 0] for i in range(len(lines_ok))]

                    line_set_ok = o3d.geometry.LineSet()
                    line_set_nok = o3d.geometry.LineSet()

                    line_set_ok.points = o3d.utility.Vector3dVector(numpy.asarray(pcd_corr.points))
                    line_set_nok.points = o3d.utility.Vector3dVector(numpy.asarray(pcd_corr.points))

                    line_set_ok.lines = o3d.utility.Vector2iVector(lines_ok)
                    line_set_ok.colors = o3d.utility.Vector3dVector(colors_ok)

                    line_set_nok.lines = o3d.utility.Vector2iVector(lines_nok)
                    line_set_nok.colors = o3d.utility.Vector3dVector(colors_nok)

                    # o3d.visualization.draw_geometries([pcd_corr, line_set_ok , line_set_nok])
                    
                    # logpath = f"/home/benji/projects/D3Feat.pytorch/geometric_registration/log_result-{args.num_points}/{scene}-evaluation"
                    # if not os.path.exists(logpath):
                    #     os.mkdir(logpath)

                    # # write the transformation matrix into .log file for evaluation.
                    # with open(os.path.join(logpath, f'{args.chosen_snapshot}.log'), 'a+') as f:
                    #     trans = result_ransac.transformation
                    #     trans = np.linalg.inv(trans)
                    #     s1 = f'{id1}\t {id2}\t  37\n'
                    #     f.write(s1)
                    #     f.write(f"{trans[0,0]}\t {trans[0,1]}\t {trans[0,2]}\t {trans[0,3]}\t \n")
                    #     f.write(f"{trans[1,0]}\t {trans[1,1]}\t {trans[1,2]}\t {trans[1,3]}\t \n")
                    #     f.write(f"{trans[2,0]}\t {trans[2,1]}\t {trans[2,2]}\t {trans[2,3]}\t \n")
                    #     f.write(f"{trans[3,0]}\t {trans[3,1]}\t {trans[3,2]}\t {trans[3,3]}\t \n")

                if vis == True:
                    
                    # o3d.visualization.draw_geometries([pcd_s, pcd_t])
                    path_reg = f'/home/benji/projects/D3Feat.pytorch/results/registration/{scene}'
                    path_dev = f'/home/benji/projects/D3Feat.pytorch/results/deviation/{scene}'
                    path_pcd = f'/home/benji/projects/D3Feat.pytorch/results/pcd/{scene}'
                    path_corr = f'/home/benji/projects/D3Feat.pytorch/results/corr/{scene}'
                    name = f'{id1}+{id2}'

                    if not os.path.exists(path_dev):
                        os.mkdir(path_dev)

                    if not os.path.exists(path_pcd):
                        os.mkdir(path_pcd)

                    if not os.path.exists(path_corr):
                        os.mkdir(path_corr)

                    draw_registration_result(pcd_s, pcd_t, result_ransac.transformation, path_reg, name)
                    # o3d.io.write_point_cloud(f'{path_dev}/{name}.pcd', pcd_s + pcd_t)
                    # o3d.io.write_point_cloud(f'{path_pcd}/{id1}.pcd', pcd_s)
                    # o3d.io.write_point_cloud(f'{path_pcd}/{id2}.pcd', pcd_t)
                    # o3d.io.write_line_set(f'{path_corr}/{name}.ply', line_set_ok + line_set_nok, print_progress=True)

                    # o3d.visualization.draw_geometries([pcd_s, pcd_s_corr])
                    # o3d.visualization.draw_geometries([pcd_t, pcd_t_corr])
                    # o3d.visualization.draw_geometries([pcd_s, pcd_s_corr, pcd_t, pcd_t_corr])

                    # o3d.visualization.draw_geometries([pcd_s, pcd_s_corr])
                    # o3d.visualization.draw_geometries([pcd_t, pcd_t_corr])

                num_inliers = np.sum(distance < distance_threshold)
                inlier_ratio = num_inliers / len(distance)
                if inlier_ratio > inlier_ratio_threshold:
                    pred_matches += 1
                gt_matches += 1
                inlier_num_meter.update(num_inliers)
                inlier_ratio_meter.update(inlier_ratio)

    # t-NSE
    # vis_pcd = get_colored_point_cloud_feature(src_keypts, features, 0.025)
    # o3d.visualization.draw_geometries([vis_pcd])

    recall = pred_matches * 100.0 / gt_matches
    return_dict[scene] = [recall, inlier_num_meter.avg, inlier_ratio_meter.avg]
    logging.info(f"{scene}: Recall={recall:.2f}%, inlier ratio={inlier_ratio_meter.avg*100:.2f}%, inlier num={inlier_num_meter.avg:.2f}")
    return recall, inlier_num_meter.avg, inlier_ratio_meter.avg


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
    recall_list = []
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
        num_frag = len([filename for filename in os.listdir(pcdpath) if filename.endswith('ply')])
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
            pts_all = inputs['points'][0]
            scores_all = scores
            features, scores = features[:int(pcd_size)], scores[:int(pcd_size)]
            
            # scores = torch.ones_like(features[:, 0:1])
            np.save(f'{descriptor_path_scene}/cloud_bin_{ids}.D3Feat', features.detach().cpu().numpy().astype(np.float32))
            np.save(f'{keypoint_path_scene}/cloud_bin_{ids}', pts.detach().cpu().numpy().astype(np.float32))
            np.save(f'{score_path_scene}/cloud_bin_{ids}', scores.detach().cpu().numpy().astype(np.float32))
            print(f"Generate cloud_bin_{ids} for {scene}")

            # ids = 0 -> pdc 0
            # if scene is "7-scenes-redkitchen":

            # # Visualize the detected keypts on src_pcd and tgt_pcd
            # box_list = []
            # # scores = np.load('new_score.npy')
            # top_k = np.argsort(scores.detach().cpu().numpy(), axis=0)[-100:]
            # for i in range(100):
            #     src_pcd = pts.detach().cpu().numpy().astype(np.float32)
            #     mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            #     mesh_box.translate(src_pcd[top_k[i]].reshape([3, 1]))
            #     mesh_box.paint_uniform_color([1, 0, 0])
            #     box_list.append(mesh_box)
                
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(src_pcd)
            #     o3d.geometry.PointCloud.estimate_normals(pcd)
            #     if ids % 2 is not 0:
            #         pcd.paint_uniform_color([1, 0.706, 0])
            #     else:
            #         pcd.paint_uniform_color([0, 0.651, 0.929])
                
            # o3d.visualization.draw_geometries([pcd] + box_list)

            # visualize keypoints based on score
            data = np.load(f'{keypoint_path_scene}/cloud_bin_{ids}.npy')
            score = np.load(f'{score_path_scene}/cloud_bin_{ids}.npy')

            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]

            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(projection='3d')
            norm = matplotlib.colors.Normalize(vmin = np.min(score), vmax = np.max(score), clip = False)
            img = ax.scatter(x, y, z, c=score, cmap="bwr", marker=".", norm=norm, alpha=0.5)
            fig.colorbar(img)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            plt.show()

            # t-SNE
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(inputs['points'][0][:int(pcd_size)].detach().cpu().numpy())
            # vis_pcd = get_colored_point_cloud_feature(pcd, features.detach().cpu().numpy(), 0.02)
            # # o3d.visualization.draw_geometries([vis_pcd])
            # custom_draw_geometry_with_custom_fov(vis_pcd, 90.0)
            # # o3d.io.write_triangle_mesh("/home/benji/projects/", [vis_pcd])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chosen_snapshot', default='D3Feat03041342', type=str, help='snapshot dir')
    parser.add_argument('--inlier_ratio_threshold', default=0.05, type=float)
    parser.add_argument('--distance_threshold', default=0.20, type=float)
    parser.add_argument('--random_points', default=False, action='store_true')
    parser.add_argument('--num_points', default=5000, type=int)
    parser.add_argument('--generate_features', default=False, action='store_true')
    args = parser.parse_args()
    if args.random_points:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-rand-{args.num_points}.log'
    else:
        log_filename = f'geometric_registration/{args.chosen_snapshot}-pred-{args.num_points}.log'
    logging.basicConfig(level=logging.INFO, 
        filename=log_filename, 
        filemode='w', 
        format="")


    config_path = f'/home/benji/projects/D3Feat.pytorch/data/D3Feat/snapshot/{args.chosen_snapshot}/config.json'
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

    # # dynamically load the model from snapshot
    # module_file_path = f'snapshot/{chosen_snap}/model.py'
    # module_name = 'model'
    # module_spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    # module = importlib.util.module_from_spec(module_spec)
    # module_spec.loader.exec_module(module)
    # model = module.KPFCNN(config)
    
    # if test on datasets with different scale
    # config.first_subsampling_dl = [new voxel size for first layer]
    
    model = KPFCNN(config)
    model.load_state_dict(torch.load(f'/home/benji/projects/D3Feat.pytorch/data/D3Feat/snapshot/{args.chosen_snapshot}/models/model_best_acc.pth')['state_dict'], strict=False)
    print(f"Load weight from snapshot/{args.chosen_snapshot}/models/model_best_acc.pth")

    # dummy_input = Variable(torch.randn(1)) 

    # torch.onnx.export(model, dummy_input, "d3feat_233.onnx")

    model.eval()

    save_path = f'/home/benji/projects/D3Feat.pytorch/geometric_registration/{args.chosen_snapshot}'
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
    scene_list = [
        'own'
        # '7-scenes-redkitchen',
        # 'sun3d-home_at-home_at_scan1_2013_jan_1',
        # 'sun3d-home_md-home_md_scan9_2012_sep_30',
        # 'sun3d-hotel_uc-scan3',
        # 'sun3d-hotel_umd-maryland_hotel1',
        # 'sun3d-hotel_umd-maryland_hotel3',
        # 'sun3d-mit_76_studyroom-76-1studyroom2',
        # 'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika'
    ]
    return_dict = Manager().dict()
    # register_one_scene(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene_list[0])
    jobs = []
    for scene in scene_list:
        p = Process(target=register_one_scene, args=(args.inlier_ratio_threshold, args.distance_threshold, save_path, return_dict, scene))
        jobs.append(p)
        p.start()
    
    for proc in jobs:
        proc.join()

    recalls = [v[0] for k, v in return_dict.items()]
    inlier_nums = [v[1] for k, v in return_dict.items()]
    inlier_ratios = [v[2] for k, v in return_dict.items()]

    logging.info("*" * 40)
    logging.info(recalls)
    logging.info(f"All 8 scene, average recall: {np.mean(recalls):.2f}%")
    logging.info(f"All 8 scene, average num inliers: {np.mean(inlier_nums):.2f}")
    logging.info(f"All 8 scene, average num inliers ratio: {np.mean(inlier_ratios)*100:.2f}%")

    print("Nr. points: ", args.num_points)
    print("All 8 scene, average recall: ", np.mean(recalls))
    print("All 8 scene, average num inliers: ", np.mean(inlier_nums))
    print("All 8 scene, average num inliers ratio: ", np.mean(inlier_ratios))

