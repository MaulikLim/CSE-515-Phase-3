from imageLoader import load_images_from_folder
import numpy as np


class SVD:

    def __init__(self, k):
        self.name = "svd"
        self.k = k

    def compute_semantics(self, data):
        data = np.array(data)
        data_t = data.transpose()
        r = np.matmul(data_t, data)
        r_eig = np.linalg.eig(r)
        r_eig_values = np.around(r_eig[0].real, 2)
        sorted_r_indices = np.flip(np.argsort(r_eig_values))
        r_eig_vectors = []
        for i in range(min(self.k,len(r_eig_values))):
            r_eig_vectors.append(r_eig[1][:, sorted_r_indices[i]].real.tolist())
        return r_eig_vectors
