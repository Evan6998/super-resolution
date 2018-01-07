# encoding=utf-8
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import cropPatch
import pickle

def main():
    lr_patches, hr_patches = cropPatch.getVectorizedPatches()
    learn = MiniBatchKMeans(n_clusters=2048, n_init=5, verbose=1, init_size=6144, reassignment_ratio=0, batch_size=2000)

    y_predict = learn.fit_predict(lr_patches)

    pickle.dump(learn, open('../model/class512_mini', 'w'))
    pickle.dump(lr_patches, open('../model/lr_patches_mini', 'w'))
    pickle.dump(hr_patches, open('../model/hr_patches_mini', 'w'))


if __name__ == '__main__':
    main()
