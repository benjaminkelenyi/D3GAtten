import numpy as np

# read scores
score_7_scenes_redkitchen_1 = np.load("/home/benji/projects/D3Feat/geometric_registration/D3Feat_contralo-54-pred/scores/7-scenes-redkitchen/cloud_bin_1.npy")
score_7_scenes_redkitchen_2 = np.load("/home/benji/projects/D3Feat.pytorch/geometric_registration/D3Feat03041342/scores/7-scenes-redkitchen/cloud_bin_1.npy")

# calcultate delta on score
new_score = abs(score_7_scenes_redkitchen_1 - score_7_scenes_redkitchen_2)

# save as .npy
np.save("new_score", new_score)

# read back the saved scores
np.load("new_score.npy")

# check if ==
print(np.array_equal(score_7_scenes_redkitchen_1, score_7_scenes_redkitchen_2))