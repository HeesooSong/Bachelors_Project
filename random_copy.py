import os
import shutil
import random

org_dir = '/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1/test2/pos'
dir = "/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1/ultimate_test/pos"

images = os.listdir(org_dir)
random_images = random.sample(images, k=100)
for image in random_images:
    image_path = org_dir+'/'+ image
    shutil.copy(image_path, dir)

org_dir = '/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1/test2/neg'
dir = "/home/ba4_project/ba4_Hee/timothy_workstation/data/training_dataset1/ultimate_test/neg"

images = os.listdir(org_dir)
random_images = random.sample(images, k=100)
for image in random_images:
    image_path = org_dir+'/'+ image
    shutil.copy(image_path, dir)