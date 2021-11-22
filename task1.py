from classifiers.DecisionTree import DecisionTree
from classifier.svm import SVM
from metrics_utils import print_matrices
from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb
import featureLoader
import latentFeatureGenerator

from arg_parser_util import Parser
from tech.PCA import PCA
from utilities import print_semantics_sub, print_semantics_type

# parser = argparse.ArgumentParser(description="Task 1")
parser = Parser("Task 1")
parser.add_args("-fp", "--folder_path", str, True)
parser.add_args("-f", "--feature_model", str, True)
parser.add_args("-k", "--k", int, True)
parser.add_args("-qf", "--query_folder", str, True)
parser.add_args("-c", "--classifier", str, True)

args = parser.parse_args()

data = latentFeatureGenerator.compute_latent_features(args.folder_path, args.feature_model, args.k)
if data is not None:
    # features will be latent features of the images in the given folder
    features = data[0]
    # each labels will correspond to each feature row in the features matrix
    labels = [x.split("-")[1] for x in data[1]]
    # train classifier as given in the input
    classifier = args.classifier.lower()
    if classifier == 'ppr':
        # Train PPR
        pass
    elif classifier == 'svm':
        # Train SVM
        svm = SVM()
        labels = [int(x) - 1 for x in labels]
        svm.train(np.array(features), np.array(labels), 10000, 1e-5, 1e-6, verbose=True)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [int(x.split("-")[2]) - 1 for x in test_data[1]]
        test_predictions = svm.predict(test_features)
        print_matrices(test_labels, test_predictions)

    else:
        # Train decision tree
        dt = DecisionTree(features, labels)
        dt.train()
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_lables = [x.split("-")[2] for x in test_data[1]]
        i = 0
        for test_feature in test_features:
            lab = dt.predict(test_feature)
            print(lab, test_lables[i])
            i += 1
        pass
    # print(labels.shape)
    # load query data to which we are supposed to assign labels
    # query_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
    # assign types using the classifier above
