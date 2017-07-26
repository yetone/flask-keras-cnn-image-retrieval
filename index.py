# -*- coding: utf-8 -*-
# Author: yongyuan.name
import os
import h5py
import numpy as np
import argparse
from multiprocessing import cpu_count

from extract_cnn_vgg16_keras import extract_feat

from concurrent.futures import ProcessPoolExecutor


ap = argparse.ArgumentParser()
ap.add_argument('-database', required=True,
                help='Path to database which contains images to be indexed')
ap.add_argument('-index', required=True,
                help='Name of index file')
args = vars(ap.parse_args())


def get_imlist(path):
    '''
    Returns a list of filenames for all jpg images in a directory.
    '''
    return [
        os.path.join(path, f)
        for f in os.listdir(path) if f.endswith('.jpg')
    ]


'''
 Extract features and index the images
'''
if __name__ == '__main__':

    db = args['database']
    img_list = get_imlist(db)

    print '--------------------------------------------------'
    print '         feature extraction starts'
    print '--------------------------------------------------'

    def extract(arg):
        i, img_path = arg
        norm_feat = extract_feat(img_path)
        img_name = os.path.split(img_path)[1]
        print (
            'extracting feature from image No. %d , %d images in total' % (
                (i + 1), len(img_list)
            )
        )
        return norm_feat, img_name

    with ProcessPoolExecutor(max_workers=cpu_count()) as exe:
        r = exe.map(extract, enumerate(img_list))

    feats = []
    names = []

    for feat, name in r:
        feats.append(feat)
        names.append(name)

    feats = np.array(feats)

    # directory for storing extracted features
    output = args['index']

    print '--------------------------------------------------'
    print '      writing feature extraction results ...'
    print '--------------------------------------------------'

    h5f = h5py.File(output, 'w')
    h5f.create_dataset('dataset_1', data=feats)
    h5f.create_dataset('dataset_2', data=names)
    h5f.close()
