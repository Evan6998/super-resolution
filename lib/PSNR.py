import numpy as np
import math
import cv2
import bic

def PSNR(img1, img2):
    img1 = np.array(img1, dtype=np.float64)
    img2 = np.array(img2,dtype=np.float64)
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    return 20 * np.log10(255. / np.sqrt(mse))

def SSIM(img1, img2):
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    u1 = np.mean(img1)
    u2 = np.mean(img2)
    v1 = np.var(img1)
    v2 = np.var(img2)
    co = np.cov(img1.flatten('C'), img2.flatten('C'))[0,1]
    k1, k2 = 0.01, 0.03
    c1 = (k1*255)**2.
    c2 = (k2*255)**2.
    return ((2*u1*u2+c1)*(2*co+c2))/((u1**2.+u2**2.+c1)*(v1+v2+c2))

def main():
    import os
    images = os.listdir('../data/Test/Set14')
    psnrs = []
    ssims = []
    
    for img_ in images:
        img = cv2.imread('../data/Test/Set14/'+img_)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
        m, n = img.shape
        
        bicubic_img = bic.bicubic(img, m/3, n/3)
        bicubic_img_ = bic.bicubic(bicubic_img, m, n)

        
        psnr = PSNR(img, bicubic_img_)
        ssim = SSIM(img, bicubic_img_)

        psnrs.append(psnr)
        ssims.append(ssim)

        print img_ + ' PSNR: ' + `psnr`
        print img_ + ' SSIM: ' + `ssim`

    print 'average psnr: ' + `(sum(psnrs)/len(psnrs))`
    print 'average ssim: ' + `(sum(ssims)/len(ssims))`

if __name__ == '__main__':
    main()