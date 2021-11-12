import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import os


img = cv2.imread('0001.png')
Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)
# /home/siit/navi/data/input_data/dn_DIV2K/k_03

in_path = '/home/siit/navi/data/input_data/DIV2K/DIV2K_train_HR'
out_root = '/home/siit/navi/data/input_data/dn_DIV2K/'

k_list = [8] #, 16]

if not os.path.exists(os.path.join(out_root, 'path')):
    os.makedirs(os.path.join(out_root, 'path'))

for k in k_list:
    if not os.path.exists(os.path.join(out_root, 'k_{:02d}'.format(k))):
        os.makedirs(os.path.join(out_root, 'k_{:02d}'.format(k)))


# define criteria, number of clusters(K) and apply kmeans()
for file in tqdm(glob(os.path.join(in_path, '*.png'))):
    img = cv2.imread(file)
    file_name = os.path.basename(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    Z = img.reshape((-1))
    Z = np.float32(Z)
    
    cv2.imwrite(os.path.join(out_root, 'gray', file_name), img)

    
    for k in k_list:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = k
        ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        cv2.imwrite(os.path.join(out_root, 'k_{:02d}'.format(k), file_name), res2)
