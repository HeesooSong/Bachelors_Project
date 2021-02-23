import glob
import cv2
import numpy as np
import shutil

path = glob.glob(r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\data\train\mask\crop\*.png')
for path_to_img in path:
    filename = path_to_img[-17:-4]
    print(filename)
    img = cv2.imread(path_to_img, 0)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)

    # Count white pixels
    histogram, bin_edges = np.histogram(img, bins=2, range=(0, 256))
    white_pixels = histogram[1]

    # Redistribute original images according to existence of white pixels
    if white_pixels != 0:
        path_to_original_image = r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\data\train\original\crop\{}.png'.format(filename)
        path_to_positive_folder = r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\data\train\pos'
        shutil.move(path_to_original_image, path_to_positive_folder)
    else:
        path_to_original_image = r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\data\train\original\crop\{}.png'.format(filename)
        path_to_negative_folder = r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\data\train\neg'
        shutil.move(path_to_original_image, path_to_negative_folder)
