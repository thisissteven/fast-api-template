import pickle
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def compute_lbp(arr, r=3, n=8):
    """Find LBP of all pixels.
    Also perform Vectorization/Normalization to get feature vector.
    """
    # LBP function params
    radius = r
    n_points = n * radius
    n_bins = n_points + 1
    lbp = local_binary_pattern(arr, n_points, radius, 'uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, n_bins), density=True)
    return hist


def compute_glcm(arr, distances=[1, 2, 3], angles=[0, np.pi/2, np.pi/4, np.pi*3/4]):
    return graycomatrix(arr, distances=distances, angles=angles, normed=True), len(distances), len(angles)


def predict(arr, mode='combined'):
    lbp_params = [
        {'r': 1, 'n': 32},
        {'r': 2, 'n': 32},
        {'r': 3, 'n': 32},
        {'r': 4, 'n': 32},
        {'r': 5, 'n': 32},
        {'r': 6, 'n': 32}
    ]

    lbp = np.concatenate([
        compute_lbp(arr, **lbp_params[0]),
        compute_lbp(arr, **lbp_params[1]),
        compute_lbp(arr, **lbp_params[2]),
        compute_lbp(arr, **lbp_params[3]),
        compute_lbp(arr, **lbp_params[4]),
        compute_lbp(arr, **lbp_params[5]),
    ])

    glcm_res, d_dim, a_dim = compute_glcm(arr)

    glcm = np.array([
        graycoprops(glcm_res, prop='correlation'),
        graycoprops(glcm_res, prop='contrast'),
        graycoprops(glcm_res, prop='homogeneity'),
        graycoprops(glcm_res, prop='energy'),
        graycoprops(glcm_res, prop='dissimilarity'),
        graycoprops(glcm_res, prop='ASM'),

    ])

    lbp = lbp.reshape(1, sum(param['r'] * param['n'] for param in lbp_params))
    glcm = glcm.reshape(1, d_dim * a_dim * 6)

    env_dir = 'models'
    if mode == 'combined':
        with open(f'{env_dir}/combined.pkl', 'rb') as file:
            model = pickle.load(file)

        return model.predict(np.concatenate((lbp, glcm), axis=1))
    elif mode == 'lbp':
        lbp_params = [
        {'r': 1, 'n': 32},
        ]
        lbp = np.array([compute_lbp(arr, **param) for param in lbp_params])
        lbp = lbp.reshape(1, sum(param['r'] * param['n'] for param in lbp_params))
        with open(f'{env_dir}/model.pkl', 'rb') as file:
            model = pickle.load(file)

        return model.predict(lbp)

    elif mode == 'glcm':
        # Replace with the actual directory path
        with open(f'{env_dir}/glcm_only.pkl', 'rb') as file:
            model = pickle.load(file)

        return model.predict(glcm)
