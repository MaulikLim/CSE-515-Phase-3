import featureLoader, featureGenerator
from tech.SVD import SVD
import numpy as np
import os
import imageLoader
import modelFactory


def compute_latent_features(folder_path, feature_model, k):
    file_name = "latent_semantics_" + feature_model + "_" + str(k) + ".json"
    latent_features_descriptors = featureLoader.load_json(os.path.join(folder_path, file_name))
    data = featureLoader.load_features_for_model(folder_path, feature_model)
    labels = data[0]
    features = data[1]
    if k<0:
        return features,labels
    if latent_features_descriptors is None:
        pca = SVD(k)
        latent_features_descriptors = [pca.compute_semantics(features), labels.tolist()]
        featureGenerator.save_features_to_json(folder_path, latent_features_descriptors, file_name)
    if latent_features_descriptors is not None:
        return np.matmul(features, np.array(latent_features_descriptors[0]).T), labels
    return [None, None]

def compute_latent_feature(folder_path, image_path, feature_model, k):
    file_name = "latent_semantics_" + feature_model + "_" + str(k) + ".json"
    latent_features_descriptors = featureLoader.load_json(os.path.join(folder_path, file_name))
    query_feature = modelFactory.get_model(feature_model).compute_features(imageLoader.load_image(image_path))
    if k<0:
        return query_feature
    if latent_features_descriptors is None:
        data = featureLoader.load_features_for_model(folder_path, feature_model)
        labels = data[0]
        features = data[1]
        pca = SVD(k)
        latent_features_descriptors = [pca.compute_semantics(features), labels.tolist()]
        featureGenerator.save_features_to_json(folder_path, latent_features_descriptors, file_name)
    if latent_features_descriptors is not None:
        return np.matmul(query_feature, np.array(latent_features_descriptors[0]).T)
    return None