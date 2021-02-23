import cv2
import os
import glob

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

path = glob.glob(r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\other_videos\*.png')
for path_to_img in path:
    img = cv2.imread(path_to_img, 0)    # Read images in grayscale
    img_h, img_w = img.shape
    split_width = 224
    split_height = 224
    X_points = start_points(img_w, split_width, 0.3)
    Y_points = start_points(img_h, split_height, 0.3)

    count = 0
    savepath = r'C:\Users\cleos\Desktop\BachelorProject\Trypanosome\other_videos\crop'
    name = path_to_img[-13:-4]
    frmt = 'png'

    for i in Y_points:
        for j in X_points:
            split = img[i:i + split_height, j:j + split_width]
            cv2.imwrite(os.path.join(savepath, '{}_{:03}.{}'.format(name, count, frmt)), split)
            count += 1



