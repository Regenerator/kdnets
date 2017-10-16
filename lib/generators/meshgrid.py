import numpy as np


def generate_clouds(idx, steps, vertices, faces, nFaces):
    clouds = np.empty((len(idx), 3, 2**steps), dtype=np.float32)
    
    for i, ind in enumerate(idx):
        triangles = np.float64(vertices[faces[nFaces[ind]:nFaces[ind+1]]])
        
        probs = np.sqrt((np.cross(triangles[:, 1, :] - triangles[:, 0, :],
                                  triangles[:, 2, :] - triangles[:, 0, :])**2).sum(axis=1))/2.
        probs /= probs.sum()
        
        sample = np.random.choice(np.arange(len(triangles)), size=2**steps, p=probs)
        samples_0 = np.random.random((2**steps, 1))
        samples_1 = np.random.random((2**steps, 1))
        cond = (samples_0[:, 0] + samples_1[:, 0]) > 1.
        samples_0[cond, 0] = 1. - samples_0[cond, 0]
        samples_1[cond, 0] = 1. - samples_1[cond, 0]
        clouds[i] = np.float32(triangles[sample, 0, :] +
                               samples_0*(triangles[sample, 1, :] - triangles[sample, 0, :]) +
                               samples_1*(triangles[sample, 2, :] - triangles[sample, 0, :])).T
    
    clouds -= clouds.mean(axis=2, keepdims=True)
    clouds /= np.fabs(clouds).max(axis=(1, 2), keepdims=True)
    
    return clouds
