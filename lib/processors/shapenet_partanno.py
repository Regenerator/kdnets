import os
import json

import numpy as np
import pandas as pd
import h5py as h5


def prepare(path2data, path2save):
    cats = sorted(os.listdir(path2data + '/data/'))

    cat2label = {}
    label2name = {}
    with open(path2data + '/synsetoffset2category.txt', 'rb') as fin:
        for i, line in enumerate(fin):
            info = line.split('\t')
            cat2label[info[1][:-1]] = np.int8(i)
            label2name[np.int8(i)] = info[0]
            
    with open(path2data + '/train_test_split/shuffled_train_file_list.json', 'rb') as fin:
        train_filenames = json.load(fin)
        
    with open(path2data + '/train_test_split/shuffled_val_file_list.json', 'rb') as fin:
        val_filenames = json.load(fin)
        
    with open(path2data + '/train_test_split/shuffled_test_file_list.json', 'rb') as fin:
        test_filenames = json.load(fin)

    train_labels = np.empty((0,), dtype=np.int8)
    train_borders = np.zeros((1,), dtype=np.int32)
    train_points = np.empty((0, 3), dtype=np.float32)
    train_point_labels = np.empty((0,), dtype=np.int8)

    for i, shapefile in enumerate(train_filenames):
        cat = shapefile[11:19]
        shapename = shapefile[20:]

        points = pd.read_csv(path2data + '/data/' + cat + '/points/' + shapename + '.pts', 
                             sep=' ', header=None).as_matrix().astype(np.float32)
        point_labels = pd.read_csv(path2data + '/data/' + cat + '/points_label/' + shapename + '.seg', 
                                   sep=' ', header=None).as_matrix()[:, 0].astype(np.int8)
        
        points -= points.mean(axis=0, keepdims=True)
        points /= np.fabs(points).max()
        tmp = points[1].copy()
        points[1] = points[2]
        points[2] = tmp
        
        train_labels = np.hstack((train_labels, cat2label[cat]))
        train_borders = np.hstack((train_borders, train_borders[-1] + len(points)))
        train_points = np.vstack((train_points, points))
        train_point_labels = np.hstack((train_point_labels, point_labels))
        
    train_point_labels -= 1

    val_labels = np.empty((0,), dtype=np.int8)
    val_borders = np.zeros((1,), dtype=np.int32)
    val_points = np.empty((0, 3), dtype=np.float32)
    val_point_labels = np.empty((0,), dtype=np.int8)

    for i, shapefile in enumerate(val_filenames):
        cat = shapefile[11:19]
        shapename = shapefile[20:]
        
        points = pd.read_csv(path2data + '/data/' + cat + '/points/' + shapename + '.pts', 
                             sep=' ', header=None).as_matrix().astype(np.float32)
        point_labels = pd.read_csv(path2data + '/data/' + cat + '/points_label/' + shapename + '.seg', 
                                   sep=' ', header=None).as_matrix()[:, 0].astype(np.int8)
        
        points -= points.mean(axis=0, keepdims=True)
        points /= np.fabs(points).max()
        tmp = points[1].copy()
        points[1] = points[2]
        points[2] = tmp
        
        val_labels = np.hstack((val_labels, cat2label[cat]))
        val_borders = np.hstack((val_borders, val_borders[-1] + len(points)))
        val_points = np.vstack((val_points, points))
        val_point_labels = np.hstack((val_point_labels, point_labels))
        
    val_point_labels -= 1

    test_labels = np.empty((0,), dtype=np.int8)
    test_borders = np.zeros((1,), dtype=np.int32)
    test_points = np.empty((0, 3), dtype=np.float32)
    test_point_labels = np.empty((0,), dtype=np.int8)

    for i, shapefile in enumerate(test_filenames):
        cat = shapefile[11:19]
        shapename = shapefile[20:]
        
        points = pd.read_csv(path2data + '/data/' + cat + '/points/' + shapename + '.pts', 
                             sep=' ', header=None).as_matrix().astype(np.float32)
        point_labels = pd.read_csv(path2data + '/data/' + cat + '/points_label/' + shapename + '.seg', 
                                   sep=' ', header=None).as_matrix()[:, 0].astype(np.int8)
        
        points -= points.mean(axis=0, keepdims=True)
        points /= np.fabs(points).max()
        tmp = points[1].copy()
        points[1] = points[2]
        points[2] = tmp
        
        test_labels = np.hstack((test_labels, cat2label[cat]))
        test_borders = np.hstack((test_borders, test_borders[-1] + len(points)))
        test_points = np.vstack((test_points, points))
        test_point_labels = np.hstack((test_point_labels, point_labels))
        
    test_point_labels -= 1

    label2offset = {0: np.int8(0)}
    label2point_labels = {}
    for label in xrange(max(label2name.keys()) + 1):
        unique_plabels = set()
        for shape_ind in (train_labels == label).nonzero()[0]:
            unique_plabels |= set(train_point_labels[train_borders[shape_ind]:train_borders[shape_ind+1]])
        
        if label < max(label2name.keys()):
            label2offset[label+1] = label2offset[label] + np.int8(len(unique_plabels))
        label2point_labels[label] = np.array(list(unique_plabels), dtype=np.int8) + label2offset[label]
        
    for i, label in enumerate(train_labels):
        train_point_labels[train_borders[i]:train_borders[i+1]] += label2offset[label]
        
    for i, label in enumerate(val_labels):
        val_point_labels[val_borders[i]:val_borders[i+1]] += label2offset[label]
        
    for i, label in enumerate(test_labels):
        test_point_labels[test_borders[i]:test_borders[i+1]] += label2offset[label]

    with h5.File(path2save + '/shapenet_partanno.h5', 'w') as fout:
        fout.create_dataset('train_labels', data=train_labels, dtype=np.int8)
        fout.create_dataset('train_borders', data=train_borders, dtype=np.int32)
        fout.create_dataset('train_points', data=train_points, dtype=np.float32)
        fout.create_dataset('train_point_labels', data=train_point_labels, dtype=np.int8)

        fout.create_dataset('val_labels', data=val_labels, dtype=np.int8)
        fout.create_dataset('val_borders', data=val_borders, dtype=np.int32)
        fout.create_dataset('val_points', data=val_points, dtype=np.float32)
        fout.create_dataset('val_point_labels', data=val_point_labels, dtype=np.int8)

        fout.create_dataset('test_labels', data=test_labels, dtype=np.int8)
        fout.create_dataset('test_borders', data=test_borders, dtype=np.int32)
        fout.create_dataset('test_points', data=test_points, dtype=np.float32)
        fout.create_dataset('test_point_labels', data=test_point_labels, dtype=np.int8)

    print('Data is processed and saved to ' + path2save + '/shapenet_partanno.h5')
    print('\nlabel2name = {')
    for i, key in enumerate(sorted(label2name.keys())):
        print('    {}: "{}"'.format(key, label2name[key]) + (',' if i < len(label2name) - 1 else ''))
    print('}')
    print('\nlabel2point_labels = {')
    for i, key in enumerate(sorted(label2point_labels.keys())):
        print('    {}: {}'.format(key, label2point_labels[key]) + (',' if i < len(label2point_labels) - 1 else ''))
    print('}')
