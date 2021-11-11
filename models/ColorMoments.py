import numpy as np
from scipy.spatial import distance
from scipy.stats import skew
from scipy.stats import wasserstein_distance as wd


class ColorMoments:

    # Initializes the model with the given specifications
    def __init__(self, windowX=8, windowY=8):
        self.name = "cm"
        self.windowX = windowX
        self.windowY = windowY

    # Computes feature vectors for the given image matrix
    def compute_features(self, image):
        height, width = image.shape[0], image.shape[1]
        feature = []
        window_size = self.windowX * self.windowY
        # Extracts the windows, computes color moments for each of them and concatenates them sequentially
        for y in range(0, height, self.windowY):
            currR = image[y:min(height, y+self.windowY), :]
            for x in range(0, width, self.windowX):
                window = currR[:, x:min(width, x+self.windowX)]
                mean = np.mean(window)
                std = np.std(window)
                skewness = skew(window.flatten())
                feature.extend([mean, std, skewness])
        return np.array(feature)

    # # Computes and visualizes feature vectors for a given image
    # def visualize_feature(self, image):
    #     height, width = image.shape[0], image.shape[1]
    #     features = self.compute_features(image)
    #     index = 0
    #     for y in range(0, height, self.windowY):
    #         for x in range(0, width, self.windowX):
    #             print(features[index], end=" ")
    #             index += 1
    #         print()

    # Computes feature vectors for all image matrices provided
    def compute_features_for_images(self, images):
        return np.array([self.compute_features(x) for x in images])

    # Calculates intersection similarity between a pair of color moments
    def computeIntersectionSimilarity(self, vector1, vector2):
        minSum = 0
        maxSum = 0
        for x in range(len(vector1)):
            minSum += min(vector1[x], vector2[x])
            maxSum += max(vector1[x], vector2[x])
        return minSum/maxSum

    # Calculates similarity score between feature vector of two images by first computing 
    # individual intersection similarities between color moments and then taking a L2 norm
    def computeL2Similarity(self, feature1, feature2):
        
        return np.sum(np.absolute(feature1-feature2))

    # Calculates similarity scores between the query image and every other image
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

    # Calculates normalized similarity scores between the query image and every other image
    # def get_normalized_similarities(self, labels, features, file_id):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     return similarities / np.sum(similarities)

    # Retrieves k most similar images from the given features based on the computed similarity scores
    # def get_top_k(self, labels, features, file_id, k):
    #     similarities = self.computeL2Similarities(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.flipud(np.argsort(similarities)[-k-1:])
    #     return [(labels[x], similarities[x]) for x in ans]

    # Following code are some other distance functions and configurations tested out

    # def computeSimilarityArray(self, feature1, feature2):
    #     n = len(feature1)
    #     similarities = np.zeros(n)
    #     for x in range(n):
    #         similarities[x] = self.computeIntersectionSimilarity(
    #             feature1[x], feature2[x])
    #     return similarities

    # def computeIndividualDistance(self, vector1, vector2):
    #     return np.linalg.norm(vector1 - vector2)

    # def computeWassersteinDistance(self, vector1, vector2):
    #     return wd(vector1, vector2)

    # def computeL2Distance(self, feature1, feature2):
    #     n = feature1.shape[0]
    #     vectorDistances = []
    #     for x in range(n):
    #         vectorDistances.append(
    #             self.computeWassersteinDistance(feature1[x], feature2[x]))
    #     return np.mean(np.array(vectorDistances))

    # def computeL2Distances(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     distances = np.zeros(n)
    #     for x in range(n):
    #         distances[x] = self.computeL2Distance(
    #             query_image, features[x])
    #     return distances

    # def computeMahalanobisDistance(self, feature1, feature2):
    #     distribution1 = np.mean(feature1, axis=0)
    #     distribution2 = np.mean(feature2, axis=0)
    #     inv_var = np.linalg.inv(np.cov(feature1.T))
    #     return distance.mahalanobis(distribution1, distribution2, inv_var)

    # def computeMahalanobisDistances(self, features, index):
    #     n = len(features)
    #     query_image = features[index]
    #     distances = np.zeros(n)
    #     for x in range(n):
    #         distances[x] = self.computeMahalanobisDistance(query_image,
    #                                                        features[x])
    #     return distances

    # def getTopK(self, labels, features, file_id, k):
    #     distances = self.computeL2Distances(
    #         features, self.getIndex(labels, file_id))
    #     ans = np.argsort(distances)[0:k+1]
    #     return [(labels[x], distances[x]) for x in ans]

    # def getNormalizedDistances(self, labels, features, file_id, k):
    #     distances = self.computeL2Distances(
    #         features, self.getIndex(labels, file_id))
    #     return distances/np.sum(distances)
