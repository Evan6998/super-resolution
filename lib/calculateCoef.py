import numpy as np
import pickle

def calcuMatrices():
    lr_patches = pickle.load(open('../model/lr_patches_mini', 'r'))
    hr_patches = pickle.load(open('../model/hr_patches_mini', 'r'))
    model = pickle.load(open('../model/class512_mini', 'r'))
    print 'load finished...'
    centerNum = model.n_clusters
    labels_ = model.labels_

    lr_patch_each_center = [[] for i in range(centerNum)]
    hr_patch_each_center = [[] for i in range(centerNum)]
    coef                 = [[] for i in range(centerNum)]
    
    for i in range(len(labels_)):
        lr_patch_each_center[labels_[i]].append(lr_patches[i])
        hr_patch_each_center[labels_[i]].append(hr_patches[i])
    
    for i in range(centerNum):
        lr_patch_this_center = np.append(lr_patch_each_center[i], np.ones((len(lr_patch_each_center[i]), 1)), 1)
        lr_patch_this_center = np.array(lr_patch_this_center)
        hr_patch_this_center = np.array(hr_patch_each_center[i])
        coef[i] = np.linalg.pinv(lr_patch_this_center.T.dot(lr_patch_this_center)).dot(lr_patch_this_center.T).dot(hr_patch_this_center)

    pickle.dump(coef, open('coef_pinv_mini', 'w'))


def main():
    calcuMatrices()

if __name__ == '__main__':
    main()