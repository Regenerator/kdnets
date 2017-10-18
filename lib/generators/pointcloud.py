import numpy as np


def generate_clouds(idx, steps, coordinates, borders):
    lb, ub = borders[idx], borders[idx+1]
    smpl = lb.reshape(-1, 1) + np.int32(np.random.random((len(idx), 2**steps))*(ub - lb).reshape(-1, 1))
    sn = 2e-4*(np.random.random((len(idx), 3, 2**steps)) - 0.5)
    return np.float32(np.transpose(coordinates[smpl], (0, 2, 1)) + sn), smpl
