import open3d as o3d



pcd_reg = o3d.io.read_point_cloud('/home/benji/projects/D3Feat.pytorch/results/registration/7-scenes-redkitchen/0+1.pcd')
pcd_dev = o3d.io.read_point_cloud('/home/benji/projects/D3Feat.pytorch/results/deviation/7-scenes-redkitchen/0+1.pcd')
pcd_s = o3d.io.read_point_cloud('/home/benji/projects/D3Feat.pytorch/results/pcd/7-scenes-redkitchen/0.pcd')
pcd_t = o3d.io.read_point_cloud('/home/benji/projects/D3Feat.pytorch/results/pcd/7-scenes-redkitchen/1.pcd')
pcd_corr = o3d.io.read_point_cloud('/home/benji/projects/D3Feat.pytorch/results/corr/7-scenes-redkitchen/0+1.ply')

# pcd_res= pcd_s + pcd_t + pcd_corr

o3d.visualization.draw_geometries([pcd_reg], 'Registration')
o3d.visualization.draw_geometries([pcd_dev], 'Deviation')
o3d.visualization.draw_geometries([pcd_s], 'PCD_S')
o3d.visualization.draw_geometries([pcd_t], 'PCD_T')
o3d.visualization.draw_geometries([pcd_s, pcd_t], 'PCD_CORR')
