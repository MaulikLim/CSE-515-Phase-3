import featureLoader, featureGenerator;
from tech.PCA import PCA;
import numpy as np;
def compute_latent_features(folder_path, feature_model, k):
    file_name = "latent_semantics_" + feature_model +  "_" + str(k) + ".json"
    latent_features_descriptors = featureLoader.load_json(folder_path + '/' + file_name)
    data = featureLoader.load_features_for_model(folder_path, feature_model)
    labels = data[0]
    features = data[1]
    if k<0:
        return features,labels
    if latent_features_descriptors is None:
        pca = PCA(k)
        latent_features_descriptors = [pca.compute_semantics(features), labels.tolist()]
        featureGenerator.save_features_to_json(folder_path,latent_features_descriptors,file_name)
    if latent_features_descriptors is not None:
        return np.matmul(features, np.array(latent_features_descriptors[0][0])),labels
    return [None, None]