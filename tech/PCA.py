from imageLoader import load_images_from_folder
import numpy as np


class PCA:

    def __init__(self, k):
        self.name = "svd"
        self.k = k

    def compute_semantics(self, data):
        data = np.array(data)
        covar = np.cov(data, rowvar=False, bias=True)
        eig = np.linalg.eig(covar)
        eig_values = np.around(eig[0].real, 2)
        sorted_indices = np.flip(np.argsort(eig_values))
        eig_mat = []
        eig_vectors = []
        for i in range(len(sorted_indices)):
            eig_mat.append(eig_values[sorted_indices[i]])
            eig_vectors.append(eig[1][:, sorted_indices[i]].real)
            if len(eig_mat) == self.k:
                break
        #eig_mat = np.sqrt(np.diag(np.array(eig_mat)))
        print(np.array(eig_vectors).shape)
        return [np.array(eig_vectors).transpose().tolist(), eig_mat, np.array(eig_vectors).tolist()]
