from classifier.svm import SVM
from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb
import featureLoader
import latentFeatureGenerator;
from metrics_utils import print_matrices

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
    # features will be latent features of the images in the given folder
    features = data[0]
    # each labels will correspond to each feature row in the features matrix
    labels = [x.split("-")[2] for x in data[1]]
    # train classifier as given in the input
    classifier = args.classifier.lower()
    if classifier == 'ppr':
        # Train PPR
        pass
    elif classifier == 'svm':
        # Train SVM
        svm = SVM()
        labels = [int(x)-1 for x in labels]
        svm.train(np.array(features), np.array(labels), 10000, 1e-5, 1e-6, verbose=True)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [int(x.split("-")[2]) - 1 for x in test_data[1]]
        test_predictions = svm.predict(test_features)
        print_matrices(test_labels, test_predictions)

    else:
        # Train decision tree
        pass
    # load query data to which we are supposed to assign labels
    query_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
    # assign subject using the classifier above
