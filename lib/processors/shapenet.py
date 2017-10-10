import os

import numpy as np
import pandas as pd
import h5py as h5


def prepare(path2data, path2save, pose='normal'):
    raw_label_data = pd.read_csv(path2data + '/train.csv').append(pd.read_csv(path2data + '/val.csv'), ignore_index=True)

    label_idx = 0
    labelId2label = {}
    for labelId in raw_label_data['synsetId'].values:
        if not labelId in labelId2label:
            labelId2label[labelId] = label_idx
            label_idx += 1

    sublabel_idx = 0
    sublabelId2sublabel = {}
    for sublabelId in raw_label_data['subSynsetId'].values:
        if not sublabelId in sublabelId2sublabel:
            sublabelId2sublabel[sublabelId] = sublabel_idx
            sublabel_idx += 1

    shapeId2label = {}
    shapeId2sublabel = {}
    for i, shapeId in enumerate(raw_label_data['id'].values):
        shapeId2label[shapeId] = np.int8(labelId2label[raw_label_data['synsetId'].values[i]])
        shapeId2sublabel[shapeId] = np.int8(sublabelId2sublabel[raw_label_data['subSynsetId'].values[i]])

    train_filenames = sorted(os.listdir(path2data + '/train_' + pose))
    val_filenames = sorted(os.listdir(path2data + '/val_' + pose))
    test_filenames = sorted(os.listdir(path2data + '/test_' + pose))

    train_vertices_cnt = 0
    train_faces_cnt = 0
    val_vertices_cnt = 0
    val_faces_cnt = 0
    test_vertices_cnt = 0
    test_faces_cnt = 0

    for i, shapefile in enumerate(train_filenames):
        with open(path2data + '/train_' + pose + '/' + shapefile, 'rb') as fobj:
            for line in fobj:
                if 'v' in line:
                    train_vertices_cnt += 1
                if 'f' in line:
                    train_faces_cnt += 1

    for i, shapefile in enumerate(val_filenames):
        with open(path2data + '/val_' + pose + '/' + shapefile, 'rb') as fobj:
            for line in fobj:
                if 'v' in line:
                    val_vertices_cnt += 1
                if 'f' in line:
                    val_faces_cnt += 1

    for i, shapefile in enumerate(test_filenames):
        with open(path2data + '/test_' + pose + '/' + shapefile, 'rb') as fobj:
            for line in fobj:
                if 'v' in line:
                    test_vertices_cnt += 1
                if 'f' in line:
                    test_faces_cnt += 1

    hf = h5.File(path2save + '/shapenet_' + pose + '.h5', 'w')

    train_nFaces = hf.create_dataset('train_nFaces', (1 + len(train_filenames),), dtype=np.int32)
    train_faces = hf.create_dataset('train_faces', (train_faces_cnt, 3), dtype=np.int32)
    train_vertices = hf.create_dataset('train_vertices', (train_vertices_cnt, 3), dtype=np.float16)
    train_labels = hf.create_dataset('train_labels', (len(train_filenames),), dtype=np.int8)
    train_sublabels = hf.create_dataset('train_sublabels', (len(train_filenames),), dtype=np.int8)

    val_nFaces = hf.create_dataset('val_nFaces', (1 + len(val_filenames),), dtype=np.int32)
    val_faces = hf.create_dataset('val_faces', (val_faces_cnt, 3), dtype=np.int32)
    val_vertices = hf.create_dataset('val_vertices', (val_vertices_cnt, 3), dtype=np.float16)
    val_labels = hf.create_dataset('val_labels', (len(val_filenames),), dtype=np.int8)
    val_sublabels = hf.create_dataset('val_sublabels', (len(val_filenames),), dtype=np.int8)

    test_nFaces = hf.create_dataset('test_nFaces', (1 + len(test_filenames),), dtype=np.int32)
    test_faces = hf.create_dataset('test_faces', (test_faces_cnt, 3), dtype=np.int32)
    test_vertices = hf.create_dataset('test_vertices', (test_vertices_cnt, 3), dtype=np.float16)
    
    train_nFaces[0] = 0
    val_nFaces[0] = 0
    test_nFaces[0] = 0

    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(train_filenames):
        shape_name = np.int32(shapefile[:6])
        shape_vertices = []
        shape_faces = []
        with open(path2data + '/train_' + pose + '/' + shapefile, 'rb') as fobj:
            for j, line in enumerate(fobj):
                tmp = line.split(' ')
                if tmp[0] == 'v':
                    shape_vertices.append(list(map(np.float16, tmp[1:])))
                elif tmp[0] == 'f':
                    shape_faces.append(list(map(lambda x: np.int32(x.split('/')[0]), tmp[1:])))

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        train_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        train_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        train_nFaces[i+1] = faces_pos + faces_offset
        train_labels[i] = shapeId2label[shape_name]
        train_sublabels[i] = shapeId2sublabel[shape_name]

        vertices_pos += vertices_offset
        faces_pos += faces_offset

    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(val_filenames):
        shape_name = np.int32(shapefile[:6])
        shape_vertices = []
        shape_faces = []
        with open(path2data + '/val_' + pose + '/' + shapefile, 'rb') as fobj:
            for j, line in enumerate(fobj):
                tmp = line.split(' ')
                if tmp[0] == 'v':
                    shape_vertices.append(list(map(np.float16, tmp[1:])))
                elif tmp[0] == 'f':
                    shape_faces.append(list(map(lambda x: np.int32(x.split('/')[0]), tmp[1:])))

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        val_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        val_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        val_nFaces[i+1] = faces_pos + faces_offset
        val_labels[i] = shapeId2label[shape_name]
        val_sublabels[i] = shapeId2sublabel[shape_name]

        vertices_pos += vertices_offset
        faces_pos += faces_offset

    vertices_pos = 0
    faces_pos = 0
    for i, shapefile in enumerate(test_filenames):
        shape_name = np.int32(shapefile[:6])
        shape_vertices = []
        shape_faces = []
        with open(path2data + '/test_' + pose + '/' + shapefile, 'rb') as fobj:
            for j, line in enumerate(fobj):
                tmp = line.split(' ')
                if tmp[0] == 'v':
                    shape_vertices.append(list(map(np.float16, tmp[1:])))
                elif tmp[0] == 'f':
                    shape_faces.append(list(map(lambda x: np.int32(x.split('/')[0]), tmp[1:])))

        shape_vertices = np.array(shape_vertices)
        buf = shape_vertices[:, 1].copy()
        shape_vertices[:, 1] = shape_vertices[:, 2]
        shape_vertices[:, 2] = buf
        shape_faces = np.array(shape_faces) - 1

        vertices_offset = shape_vertices.shape[0]
        faces_offset = shape_faces.shape[0]

        test_vertices[vertices_pos:vertices_pos+vertices_offset] = shape_vertices
        test_faces[faces_pos:faces_pos+faces_offset] = vertices_pos + shape_faces
        test_nFaces[i+1] = faces_pos + faces_offset

        vertices_pos += vertices_offset
        faces_pos += faces_offset

    hf.close()
    print('\nData in ' + path2data + ' ('+ pose + ') ' + 'is processed and saved to ' + path2save + '/shapenet_' + pose + '.h5')
    