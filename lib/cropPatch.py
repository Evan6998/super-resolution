# encoding=utf-8
from __future__ import print_function
import cv2
import numpy as np
import os

def getVectorizedPatches():
    lr_patches = []
    hr_patches = []
    path = "../data/Train"
    images = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and 'BSD' in i]
    for i in range(len(images)):
        print('Loading patches: '+`float(i)/len(images)*100`[:4]+'%', end='\r')
        img = cv2.imread(os.path.join(path, images[i]))
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
        lrs, hrs = cropImage(img_yuv)
        lr_patches += lrs
        hr_patches += hrs
    print('', end='\n')
    print('Loading patches finished, Total: ' + `len(lr_patches)` + ' patches')
    return lr_patches[:2000000], hr_patches[:2000000]

def cropImage(img):
    lr_res, hr_res = [], []

    img_gauss = cv2.GaussianBlur(img, (11,11), 1.7)
    img_lr = cv2.resize(img_gauss, (img_gauss.shape[1]/3, img_gauss.shape[0]/3))
    m, n = img_lr.shape

    for i in range(3, m, 2):
        for j in range(3, n):
            lr_patch = img_lr[i-3:i+4,j-3:j+4].astype(np.int)
            hr_patch = img[i*3-6:i*3+6, j*3-6:j*3+6].astype(np.int)
            if lr_patch.size == 49 and hr_patch.size == 144: 

                lr_patch_vec = lr_patch.flatten('C')
                for k in [48,42,6,0]: lr_patch_vec = np.delete(lr_patch_vec, k)
                lr_patch_vec -= int(lr_patch_vec.mean())

                hr_patch_vec = hr_patch.flatten('C') - int(lr_patch[2:5,2:5].mean())

                lr_res.append(lr_patch_vec)
                hr_res.append(hr_patch_vec)
    return lr_res, hr_res
    
def main():
    a = getVectorizedPatches()
    print(a[0][40])
    print(a[1][40])

if __name__ == '__main__':
    main()