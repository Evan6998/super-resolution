import cv2
import numpy
import PSNR

def main():
    ori = cv2.imread('../data/Test/Set14/baboon.bmp')
    upsampling = cv2.imread('../data/Result/baboon.jpg')

    m, n = ori.shape[0], ori.shape[1]
    ori = ori[9:(m/3-4)*3, 9:(n/3-4)*3]
    upsampling = upsampling[9:(m/3-4)*3, 9:(n/3-4)*3]

    test = cv2.resize(ori, (n/3,m/3), interpolation=cv2.INTER_CUBIC)
    test = cv2.resize(test, (n, m), interpolation=cv2.INTER_CUBIC)[9:(m/3-4)*3, 9:(n/3-4)*3]

    ori_y = cv2.cvtColor(ori, cv2.COLOR_BGR2YUV)
    upsampling_y = cv2.cvtColor(upsampling, cv2.COLOR_BGR2YUV)
    cubic_y = cv2.cvtColor(test, cv2.COLOR_BGR2YUV)

    print cv2.PSNR(ori_y, upsampling_y)
    # print PSNR.SSIM(ori_y, upsampling_y)

    print cv2.PSNR(ori_y, cubic_y)
    # print cv2.SSIM(ori_y, cubic_y)


if __name__ == '__main__':
    main()