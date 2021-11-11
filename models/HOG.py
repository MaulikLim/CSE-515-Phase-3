import numpy as np
from skimage.feature import hog
from scipy.stats import wasserstein_distance as wd


class HOG:

    # Initializes the model with the given specifications
    def __init__(self, windowX=16, windowY=16):
        self.name = "hog"
        self.windowX = windowX
        self.windowY = windowY
        self.orientations = 9
        self.pixels_per_cell = (windowY, windowX)
        self.cells_per_block = (2, 2)

    # Computes the histogram features and reshapes it to make them accessible blockwise
    def compute_features(self, image):
        return np.array(hog(image, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell,
                            cells_per_block=self.cells_per_block, visualize=False, multichannel=False))

    # Computes and visualizes the feature vector for the given image matrix
    # def visualize_feature(self, image):
    #     features = self.compute_features(image)
    #     for x in features:
    #         print(x)

    # Generates feature vectors for all the provided image matrices
    def compute_features_for_images(self, images):
        return np.array([self.compute_features(x) for x in images])
    
    # # Computes intersection similarity between histograms for a block
    # def computeIntersectionSimilarity(self, vector1, vector2):
    #     minSum = 0
    #     maxSum = 0
    #     for x in range(len(vector1)):
    #         minSum += min(vector1[x], vector2[x])
    #         maxSum += max(vector1[x], vector2[x])
    #     return minSum/maxSum

    # # Calculates a cumulative similarity score between feature vectors of two different images
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

    # # Retrieves top k similar images based on the calculated similarity scores
    # def get_top_k(self, labels, features, file_id, k):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.flipud(np.argsort(similarities)[-k-1:])
    #     return [(labels[x], similarities[x]) for x in ans]

    # # Retruns normalized similarity scores for every image when compared to the query image
    # def get_normalized_similarities(self, labels, features, file_id):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     return similarities / np.sum(similarities)

    # def computeWassersteinDistance(self, vector1, vector2):
    #     return wd(vector1, vector2)

    # def computeDistance(self, feature1, feature2):
    #     n = feature1.shape[0]
    #     vectorDistances = []
    #     for x in range(n):
    #         vectorDistances.append(
    #             self.computeWassersteinDistance(feature1[x], feature2[x]))
    #     return np.mean(np.array(vectorDistances))

    # def computeDistances(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     distances = np.zeros(n)
    #     for x in range(n):
    #         distances[x] = self.computeDistance(
    #             query_image, features[x])
    #     return distances

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

    # def get_top_k(self, labels, features, file_id, k):
    #     distances = self.computeDistances(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.argsort(distances)[0:k+1]
    #     return [(labels[x], distances[x]) for x in ans]
