import numpy as np
from skimage.feature import local_binary_pattern
from scipy.spatial import distance
from scipy.stats import wasserstein_distance as wd


class ELBP:

    # Initializes the model with the given specifications
    def __init__(self, p=16, r=2):
        self.name = "elbp"
        self.p = p
        self.r = r
        self.method = 'uniform'

    # Computes feature vectors for the given image matrix
    def compute_features(self, image):
        height, width = image.shape[0], image.shape[1]
        feature = []
        # Computes the LBP matrix for the given image
        lbp = local_binary_pattern(image, self.p, self.r, self.method)
        return np.histogram(lbp.ravel(), bins=self.p+2, range=(0, self.p + 1))[0]
    
    def compute_elbp_features(self, image):
        elbp = self.get_lbp(image)
        var = self.get_var_mat(image,10)
        feature = np.zeros([self.p+2, 10])
        x, y = elbp.shape[0], elbp.shape[1]
        for i in range(x):
            for j in range(y):
                feature[int(elbp[i][j])][int(var[i][j])] += 1
        return feature.flatten()

    def get_lbp(self, image):
        return local_binary_pattern(image, self.p, self.r, self.method)

    def get_var(self, mat, i, j):
        res = []
        for x in range(i-1,i+2):
            for y in range(j-1,j+2):
                if(x>=0 and y>=0 and x<len(mat) and y<len(mat)):
                    if(x!=i or y!=j):
                        res.append(mat[x][y])
        return np.std(res)

    def get_var_mat(self, image, bins=10):
        elbp = []
        for i in range(0,len(image)):
            tmp = []
            for j in range(0,len(image)):
                var = self.get_var(image,i,j)
                tmp.append(var)
            elbp.append(np.array(tmp))
        elbp = np.array(elbp)
        min_var = np.min(elbp)
        max_var = np.max(elbp)
        elbp = ((elbp-min_var)*bins)//(max_var-min_var)
        elbp = np.where(elbp > (bins-1), bins-1, elbp)
        return elbp

    

    # Computes and visualizes feature vector for a given image
    # def visualize_feature(self, image):
    #     height, width = image.shape[0], image.shape[1]
    #     features = self.compute_features(image)
    #     index = 0
    #     for y in range(0, height, self.windowY):
    #         for x in range(0, width, self.windowX):
    #             print(features[index], end =" ")
    #             index += 1
    #         print()

    # Computes feature vectors for all the given image matrices
    def compute_features_for_images(self, images):
        return np.array([self.compute_features(x) for x in images])

    # def flattenFeature(self, feature):
    #     return feature.ravel()
    
    # # Computes intersection similarity between histograms of a pair of cells
    # def computeIntersectionSimilarity(self, vector1, vector2):
    #     minSum = 0
    #     maxSum = 0
    #     for x in range(len(vector1)):
    #         minSum += min(vector1[x], vector2[x])
    #         maxSum += max(vector1[x], vector2[x])
    #     return minSum/maxSum

    # # Calculates a similarity score between feature vectors of two images 
    # # by first computing intersection similarity for a pair of 
    # # corresponding histograms and then taking a L2 norm to provide a cumulative similarity score
    # def computeL2Similarity(self, feature1, feature2):
    #     n = len(feature1)
    #     similarities = np.zeros(n)
    #     for x in range(n):
    #         similarities[x] = self.computeIntersectionSimilarity(
    #             feature1[x], feature2[x])
    #     return np.linalg.norm(similarities)

    # # Calculates similarity scores between the query image and every other image provided
    # def computeL2Similarities(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     similarities = np.zeros(n)
    #     for x in range(n):
    #         similarities[x] = self.computeL2Similarity(query_image,
    #                                                    features[x])
    #     return similarities

    # def getIndex(self, labels, file_id):
    #     return labels.tolist().index(file_id)

    # # Retrieves k images which are most similar to the query image, based on their similarity scores
    # def get_top_k(self, labels, features, file_id, k):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.flipud(np.argsort(similarities)[-k-1:])
    #     return [(labels[x], similarities[x]) for x in ans]

    # # Gets normalized similarity scores of every other image when compared to the query image
    # def get_normalized_similarities(self, labels, features, file_id):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     return similarities / np.sum(similarities)

    # Below commented code are some other configurations I used to check model performance

    # def computeWassersteinDistance(self, vector1, vector2):
    #     return wd(vector1, vector2)

    # def computeDistance(self, feature1, feature2):
    #     n = feature1.shape[0]
    #     vectorDistances = []
    #     for x in range(n):
    #         vectorDistances.append(np.linalg.norm(feature1[x] - feature2[x]))
    #     return np.linalg.norm(np.array(vectorDistances))
    
    # def computeDistance(self, feature1, feature2):
    #     return np.linalg.norm(feature1.ravel() - feature2.ravel())

    # def computeDistances(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     # print(query_image)
    #     distances = np.zeros(n)
    #     for x in range(n):
    #         distances[x] = self.computeDistance(
    #             query_image, features[x])
    #     return distances

    # def computeCosineSimilarity(self, vector1, vector2):
    #     return 1 - distance.cosine(vector1, vector2)

    # def getNormalizedDistances(self, labels, features, file_id, k):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     similarities = 1 - similarities/np.sum(similarities)
    #     return similarities/np.sum(similarities)

    # def computeSimilarityArray(self, feature1, feature2):
    #     n = len(feature1)
    #     similarities = np.zeros(n)
    #     for x in range(n):
    #         similarities[x] = self.computeIntersectionSimilarity(
    #             feature1[x], feature2[x])
    #     return similarities



    # def compute_elbp_intersection_similarity(self, feature1, feature2):
    #     return np.mean(self.computeIntersectionSimilarity(feature1.ravel(), feature2.ravel()))

    # def compute_elbp_intersection_similarities(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     similarities = np.zeros(n)
    #     for x in range(n):
    #         similarities[x] = self.compute_elbp_intersection_similarity(query_image,
    #                                                    features[x])
    #     return similarities

    

    # def get_top_k(self, labels, features, file_id, k):
    #     distances = self.computeDistances(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.argsort(distances)[0:k+1]
    #     return [(labels[x], distances[x]) for x in ans]
