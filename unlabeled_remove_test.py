import os
import shutil

unlabeled_dir = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4/unlabeled/img'
dstn_path = '/home/ba4_project/ba4_Hee/Trypanosome/training_dataset4/unlabeled_testvid_excl/img'
dir = "/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1/"
video_list = ['test_vid01/pos','test_vid02/pos','test_vid03/pos','test_vid05/pos','test_vid06/pos','test_vid07/pos',
              'test_vid01/neg','test_vid02/neg','test_vid03/neg','test_vid05/neg','test_vid06/neg','test_vid07/neg']

unlabeled_imgs = os.listdir(unlabeled_dir)

test_img_list = []
for video in video_list:
    for image in os.listdir(dir + video):
        test_img_list.append(image)

for unlabeled_image in unlabeled_imgs:
    match = 0
    for test_img in test_img_list:
        if unlabeled_image == test_img:
            match = 1
    if match == 0:
        unlabeled_image_dir = unlabeled_dir + '/' + unlabeled_image
        shutil.copy(unlabeled_image_dir, dstn_path)
