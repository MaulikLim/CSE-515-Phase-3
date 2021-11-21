import modelFactory
import numpy as np
import os
import json
import pickle

# Saves the feature descriptors to a json file
def save_features_to_json(folder_path, feature_descriptors, file_name):
    json_feature_descriptors = json.dumps(feature_descriptors, indent=4)
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
            os.remove(file_path)
    with open(file_path, "w") as out_file:
        out_file.write(json_feature_descriptors)

def save_object(folder_path, obj, filename):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

# Generates feature vectors for each model for all images and saves them to a json file
def generate_and_save_features(labels, images, model_name, folder_path, file_name='feature_descriptors.json'):
    model = modelFactory.get_model(model_name)
    feature_descriptors = {}
    feature_descriptors['labels'] = labels
    features = model.compute_features_for_images(images)
    feature_descriptors[model.name] = features.tolist()
    save_features_to_json(folder_path, feature_descriptors, file_name)
