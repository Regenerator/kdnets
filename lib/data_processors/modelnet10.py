import os

import numpy as np
import h5py as h5


def prepare(path2data, path2save):
    train_nFaces = np.zeros((1,), dtype=np.int32)
    train_faces = np.empty((0, 3), dtype=np.int32)
    train_vertices = np.empty((0, 3), dtype=np.float16)
    train_labels = np.empty((0,), dtype=np.int8)

    test_nFaces = np.zeros((1,), dtype=np.int32)
    test_faces = np.empty((0, 3), dtype=np.int32)
    test_vertices = np.empty((0, 3), dtype=np.float16)
    test_labels = np.empty((0,), dtype=np.int8)

    classes = sorted(os.listdir(path2data))
    class2label = {}

    for i, cl in enumerate(classes):
        class2label[cl] = np.int8(i)
        
        cl_train_filenames = sorted(os.listdir(path2data + '/' + cl + '/train/'))
        cl_test_filenames = sorted(os.listdir(path2data + '/' + cl + '/test/'))

        for j, shapefile in enumerate(cl_train_filenames):
            with open(path2data + '/' + cl + '/train/' + shapefile, 'rb') as fobj:
                for k, line in enumerate(fobj):
                    if k == 0 and line != 'OFF\n':
                        numVertices, numFaces, numEdges = map(np.int32, line[3:].split())
                        break
                    if k == 1:
                        numVertices, numFaces, numEdges = map(np.int32, line.split())
                        break

                vrtcs = np.empty((numVertices, 3), dtype=np.float16)
                for k, line in enumerate(fobj):
                    vrtcs[k] = map(np.float16, line.split())
                    if k == numVertices - 1:
                        break

                fcs = np.empty((numFaces, 3), dtype=np.int32)
                for k, line in enumerate(fobj):
                    fcs[k] = map(np.int32, line.split())[1:]
                    if k == numFaces - 1:
                        break

            train_nFaces = np.hstack((train_nFaces, numFaces + train_nFaces[-1]))
            train_faces = np.vstack((train_faces, fcs + len(train_vertices)))
            train_vertices = np.vstack((train_vertices, vrtcs))
        train_labels = np.hstack((train_labels, class2label[cl]*np.ones(len(cl_train_filenames), dtype=np.int8)))

        for j, shapefile in enumerate(cl_test_filenames):
            with open(path2data + '/' + cl + '/test/' + shapefile, 'rb') as fobj:
                for k, line in enumerate(fobj):
                    if k == 0 and line != 'OFF\n':
                        numVertices, numFaces, numEdges = map(np.int32, line[3:].split())
                        break
                    if k == 1:
                        numVertices, numFaces, numEdges = map(np.int32, line.split())
                        break

                vrtcs = np.empty((numVertices, 3), dtype=np.float16)
                for k, line in enumerate(fobj):
                    vrtcs[k] = map(np.float16, line.split())
                    if k == numVertices - 1:
                        break

                fcs = np.empty((numFaces, 3), dtype=np.int32)
                for k, line in enumerate(fobj):
                    fcs[k] = map(np.int32, line.split())[1:]
                    if k == numFaces - 1:
                        break

            test_nFaces = np.hstack((test_nFaces, numFaces + test_nFaces[-1]))
            test_faces = np.vstack((test_faces, fcs + len(test_vertices)))
            test_vertices = np.vstack((test_vertices, vrtcs))
        test_labels = np.hstack((test_labels, class2label[cl]*np.ones(len(cl_test_filenames), dtype=np.int8)))
        
        print('{} - processed'.format(cl))

    with h5.File(path2save + '/modelnet10.h5', 'w') as hf:
        hf.create_dataset('train_nFaces', data=train_nFaces)
        hf.create_dataset('train_faces', data=train_faces)
        hf.create_dataset('train_vertices', data=train_vertices)
        hf.create_dataset('train_labels', data=train_labels)
        hf.create_dataset('test_nFaces', data=test_nFaces)
        hf.create_dataset('test_faces', data=test_faces)
        hf.create_dataset('test_vertices', data=test_vertices)
        hf.create_dataset('test_labels', data=test_labels)

    print('\nData in ' + path2data + ' is processed and saved to ' + path2save + '/modelnet10.h5')
    print('\nclass2label = {')
    for i, cl in enumerate(sorted(class2label.keys())):
        print('    "{}": {}'.format(cl, class2label[cl]) + (',' if i+1 < len(class2label) else ''))
    print('}')
    