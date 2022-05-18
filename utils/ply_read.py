import numpy as np

keypoints_1 = np.load("/home/benji/projects/D3Feat.pytorch/geometric_registration/D3Feat02241142/keypoints/7-scenes-redkitchen/cloud_bin_10.npy")
keypoints_2 = np.load("/home/benji/projects/D3Feat.pytorch/geometric_registration/D3Feat02241143/keypoints/7-scenes-redkitchen/cloud_bin_10.npy")
keypoints_3 = np.load("/home/benji/projects/D3Feat.pytorch/geometric_registration/D3Feat02241233/keypoints/7-scenes-redkitchen/cloud_bin_10.npy")
keypoints_4 = np.load("/home/benji/projects/D3Feat.pytorch/geometric_registration/D3Feat03041342/keypoints/7-scenes-redkitchen/cloud_bin_10.npy")

print(np.array_equal(keypoints_1, keypoints_2))
print(np.array_equal(keypoints_2, keypoints_3))
print(np.array_equal(keypoints_3, keypoints_4))