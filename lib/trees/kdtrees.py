import numpy as np
from copy import copy


def KDTrees(point_clouds, dim=3, steps=5, lim=1, det=False, gamma=10.):
    min_nide_size = point_clouds.shape[2]
    srt0 = [np.tile(np.arange(2**steps).reshape(1, -1), (len(point_clouds), 1))]
    step = 0
    Ns = []

    while min_nide_size > lim:
        if step == steps:
            break
            
        srt1 = []
        ns = np.empty((len(point_clouds), 3, 2**step), dtype=np.float32)
        
        buf_srt0 = np.tile(np.arange(point_clouds.shape[0]).reshape(-1, 1, 1),
                           (1, point_clouds.shape[1], 2**(steps - step)))
        buf_srt1 = np.tile(np.arange(point_clouds.shape[1]).reshape(1, -1, 1),
                           (point_clouds.shape[0], 1, 2**(steps - step)))
        
        for i, subsrt in enumerate(srt0):
            buf_srt2 = np.tile(subsrt.reshape(subsrt.shape[0], 1, subsrt.shape[1]), 
                               (1, point_clouds.shape[1], 1))
            
            crd = point_clouds[buf_srt0, buf_srt1, buf_srt2][:, :dim, :].copy()
            rng = (crd.max(axis=2) - crd.min(axis=2))
            rng /= rng.sum(axis=1).reshape(-1, 1)
            
            if det:
                split_ax = rng.argmax(axis=1)
            else:
                prob = np.exp(gamma*rng)
                prob /= prob.sum(axis=1).reshape(-1, 1)
                if dim == 2:
                    bndr = np.zeros((len(prob), 3))
                    bndr[:, 1] = prob[:, 0]
                    bndr[:, 2] = 1.
                if dim == 3:
                    bndr = np.zeros((len(prob), 4))
                    bndr[:, 1] = prob[:, 0]
                    bndr[:, 2] = prob[:, 0] + prob[:, 1]
                    bndr[:, 3] = 1.
                split_ax = (bndr >= np.random.random((prob.shape[0], 1))).argmax(axis=1) - 1
                
            norms = np.zeros((len(point_clouds), 3), dtype=np.float32)
            norms[np.arange(len(point_clouds)), split_ax] = 1.
                
            srt_buf = crd[np.arange(len(crd)), split_ax, :].argsort(axis=1)
            srt1.append(subsrt[np.tile(np.arange(subsrt.shape[0]).reshape(-1, 1), (1, subsrt.shape[1]/2)),
                               srt_buf[:, :srt_buf.shape[1]/2]])
            srt1.append(subsrt[np.tile(np.arange(subsrt.shape[0]).reshape(-1, 1), (1, (subsrt.shape[1]+1)/2)), 
                               srt_buf[:, srt_buf.shape[1]/2:]])
            
            ns[:, :, i] = norms.copy()
        
        lens = []
        for group in srt1:
            lens.append(group.shape[1])
            
        min_nide_size = min(lens)
        srt0 = copy(srt1)
        step += 1
        Ns.append(ns)
        
    sortings = np.empty((point_clouds.shape[0], 2**steps), dtype=np.int32)
    for i, subsrt in enumerate(srt0):
        if subsrt.shape[1] == 1:
            sortings[:, i] = subsrt[:, 0]
        else:
            print 'Error!'
            
    trees_data = {'sortings': sortings, 'normals': Ns}
    return trees_data
