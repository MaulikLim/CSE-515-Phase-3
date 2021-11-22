import numpy as np
import os
import json
import featureGenerator
import imageLoader
import pickle

# Loads json file of feature descriptor
def load_json(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as feature_descriptors:
            return json.load(feature_descriptors)
    return None
def load_object(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as inp:
            obj = pickle.load(inp)
            return obj
    return None


# Retrieves numpy array of labels and their corresponding feature vectors for the given model name
def load_features_for_model(folder_path, model_name, file_name='feature_descriptors'):
    file_name = file_name + "_" + model_name + ".json"
    print('Loading features from path ' + folder_path)
    features_descriptors = load_json(folder_path + '/' + file_name)
    if features_descriptors is None:
        labels, images = imageLoader.load_images_from_folder(folder_path)
        featureGenerator.generate_and_save_features(labels, images, model_name, folder_path, file_name)
        features_descriptors = load_json(folder_path + '/' + file_name)
    if features_descriptors is not None:
        return np.array(features_descriptors['labels']), np.array(features_descriptors[model_name])
    return [None, None]
