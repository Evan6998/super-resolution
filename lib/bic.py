from __future__ import division
import cv2
import numpy as np
import PSNR

def W(x):
    x = np.abs(x)
    if 0 < x <= 1: return 1-2.5*x**2+1.5*x**3
    if 1 < x <= 2: return 2-4*x+2.5*x**2-0.5*x**3
    else: return 1

def bicubic(img, m, n):
    h, w = img.shape
    result = np.zeros((m, n), np.uint8)
    scale_factor_h = m/h
    scale_factor_w = n/w
    for i in range(m):
        for j in range(n):
            x = i/scale_factor_h
            y = j/scale_factor_w
            p = x - int(x)
            q = y - int(y)
            x = int(x)
            y = int(y)
            if x <= 1 or y <= 1 or x >= h - 2 or y >= w - 2: result[i,j] = img[x, y]
            else:
                A = np.asarray([[W(1+p), W(p), W(1-p), W(2-p)]])
                B = np.asarray([[W(1+q)], [W(q)], [W(1-q)], [W(2-q)]])
                P = np.asarray([
                    [img[x-1, y-1], img[x-1, y], img[x-1, y+1], img[x-1, y+2]],
                    [img[x, y-1], img[x, y], img[x, y+1], img[x, y+2]],
                    [img[x+1, y-1], img[x+1, y], img[x+1, y+1], img[x+1, y+2]],
                    [img[x+2, y-1], img[x+2, y], img[x+2, y+1], img[x+2, y+2]]
                ])
                result[i, j] = np.dot(np.dot(A, P), B)[0,0]
    return result

def main():
    img = cv2.imread('../data/Test/Set14/baboon.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]
    m, n = img.shape

    # convert to matlab ycbcr
    # for i in range(m):
    #     for j in range(n):
    #         img[i,j] = 16 + 219.*(img[i,j]/255.)

    img_lr = bicubic(img, int(m/3), int(n/3))

    img_hr = bicubic(img_lr, m, n)
    # img_cv = cv2.resize(img, (int(n/3), int(m/3)), interpolation=cv2.INTER_CUBIC)
    # img_cv = cv2.resize(img_cv, (n, m), interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('bicubic', img_hr)
    # cv2.imshow('cv2', img_cv)
    # cv2.waitKey()
    print PSNR.PSNR(img, img_hr)
    # print PSNR.PSNR(img, img_cv)

if __name__ == '__main__':
    main()

            