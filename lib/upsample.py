import numpy as np
import cv2
import pickle
from PIL import Image

def main():
    coef = pickle.load(open('../model/coef_pinv_mini', 'r'))
    model = pickle.load(open('../model/class512_mini', 'r'))
    print "Finish loading model..."
    
    img = cv2.imread('../data/Test/Set14/baboon.bmp')
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    img_gau = cv2.GaussianBlur(img, (11,11), 1.7)

    img_lr_rgb = cv2.resize(img_gau, (img_gau.shape[1]/3, img_gau.shape[0]/3), interpolation=cv2.INTER_CUBIC)
    img_lr_yuv = cv2.cvtColor(img_lr_rgb, cv2.COLOR_BGR2YUV)
    img_lr = img_lr_yuv[:,:,0]

    res = np.ones((img.shape[0], img.shape[1]))*-1
    result = np.ones((img.shape))

    m, n = img_lr.shape
    for i in range(3, m-4):
        for j in range(3, n-4):
            lr_patch_vec = img_lr[i-3:i+4, j-3:j+4].flatten('C').astype(int)
            hr_mean = img_lr[i-3:i+4, j-3:j+4][2:5, 2:5].mean()
            for k in [48,42,6,0]: lr_patch_vec = np.delete(lr_patch_vec, k)
            lr_patch_vec_feature = lr_patch_vec - int(lr_patch_vec.mean())
            center = model.predict([lr_patch_vec_feature])[0]
            hr_patch_vec_feature = (np.append(lr_patch_vec_feature, [1])).reshape((1,46)).dot(coef[center])
            hr_patch_vec = hr_patch_vec_feature + int(hr_mean)
            hr_patch = hr_patch_vec.reshape((12,12))

            for x in range(12):
                for y in range(12):
                    u, v = 3*i-6+x, 3*j-6+y
                    if u < res.shape[0] and v < res.shape[1]:
                        if res[u, v] == -1:
                            res[u, v] = hr_patch[x, y]
                        else:
                            res[u, v] = (res[u,v] + hr_patch[x,y])/2

    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i,j] *= (res[i,j]>0)
            res[i,j] = res[i,j] * (res[i,j] <= 255) + 255 * (res[i,j] > 255)
    res = res.astype(np.uint8)

    result[:,:,0] = res
    result[:,:,1] = cv2.resize(img_lr_yuv[:,:,1], (img_gau.shape[1], img_gau.shape[0]), interpolation=cv2.INTER_CUBIC)
    result[:,:,2] = cv2.resize(img_lr_yuv[:,:,2], (img_gau.shape[1], img_gau.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_YUV2BGR)
    
    cv2.imshow('result', result)
    cv2.imwrite('../data/Result/baboon.jpg', result)
    cv2.waitKey()


if __name__ == '__main__':
    main()