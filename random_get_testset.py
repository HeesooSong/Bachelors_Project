import os
import shutil
import random

org_dir = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4_AL4/unlabeled/img'
dir = "/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1"

video_list = {'vid01':[], 'vid02':[], 'vid03':[], 'vid05':[], 'vid06':[], 'vid07':[]}
for image in os.listdir(org_dir):
    for video in video_list.keys():
        if image.count(video) >= 1:
            value_list = video_list.get(video)
            value_list.append(image)
            video_list[video] = value_list

for video in video_list:
    value_list = video_list.get(video)
    print(len(value_list))
    random_images = random.sample(value_list, k=500)
    dstn_path = dir + '/test_' + video
    for image in random_images:
        img_dir = org_dir + '/'+ image
        shutil.copy(img_dir, dstn_path)

print('finished')
