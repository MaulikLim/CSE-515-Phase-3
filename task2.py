from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb
import featureLoader
import latentFeatureGenerator;

from tech.PCA import PCA
from utilities import print_semantics_sub, print_semantics_type

parser = argparse.ArgumentParser(description="Task 2")
parser.add_argument(
    "-fp",
    "--folder_path",
    type=str,
    required=True,
)
parser.add_argument(
    "-f",
    "--feature_model",
    type=str,
    required=True,
)
parser.add_argument(
    "-k",
    "--k",
    type=int,
    required=True,
)

parser.add_argument(
    "-qf",
    "--query_folder",
    type=str,
    required=True,
)

parser.add_argument(
    "-c",
    "--classifier",
    type=str,
    required=True,
)

args = parser.parse_args()

data = latentFeatureGenerator.compute_latent_features(args.folder_path, args.feature_model, args.k)
if data is not None:
    #features will be latent features of the images in the given folder
    features = data[0]
    #each labels will correspond to each feature row in the features matrix
    labels = data[1]
    #train classifier as given in the input
    classifier = args.classifier
    if classifier == 'ppr':
        #Train PPR
        pass
    elif classifier == 'svm':
        #Train SVM
        pass
    else:
        #Train decision tree
        pass
    #load query data to which we are supposed to assign labels
    query_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
    #assign subject using the classifier above