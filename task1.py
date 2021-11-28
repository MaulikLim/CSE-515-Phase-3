from sklearn.preprocessing import MinMaxScaler

from classifiers.DecisionTree import DecisionTree
from classifier.svm import SVM
from classifier.ppr import Personalised_Page_Rank
from featureGenerator import save_features_to_json
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb
import featureLoader
import latentFeatureGenerator
from metrics_utils import print_matrices
from sklearn import tree
import math
from arg_parser_util import Parser
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
    # k=args.k
    if classifier == 'ppr':
        # Train PPR
        types_of_labels = list(set(labels))
        ppr = Personalised_Page_Rank(20, types_of_labels)
        ppr.fit(features, labels)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [x.split("-")[1] for x in test_data[1]]
        test_predicted_labels = ppr.predict(test_features, test_labels)
        # print(ppr.accuracy(test_predicted_labels, test_labels))
        print_matrices(test_labels, np.array(test_predicted_labels))
        # for test_image,test_predicted_label in test_predicted_labels.items():
        #     print(test_image,"->",test_predicted_label)

        pass
    elif classifier == 'svm':
        # Train SVM
        svm = SVM()
        label_map = {}
        labels_set = list(set(labels))
        for i, label in enumerate(labels_set):
            label_map[label] = i
        labels = [label_map[x] for x in labels]
        min_max_scalar = MinMaxScaler()
        features = min_max_scalar.fit_transform(features)
        svm.train(np.array(features), np.array(labels), -1, 10000, 1e-3, 1e-5, verbose=False)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [label_map[x.split("-")[1]] for x in test_data[1]]
        test_features = min_max_scalar.transform(test_features)
        test_predictions = svm.predict(test_features)
        # print(list(label_map.keys()))
        print_matrices(test_labels, test_predictions)

    else:
        # Train decision tree
        label_map = {}
        labels_set = list(set(labels))
        reverse_map = {}
        for i, label in enumerate(labels_set):
            label_map[label] = i
            reverse_map[i] = label
        labels = [label_map[x] for x in labels]
        dt = DecisionTree(features, np.array(labels),int(3*math.log(len(labels))))
        dt.train()

        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [x.split("-")[1] for x in test_data[1]]

        predict_labels = []
        for i, feature in enumerate(test_features):
            predict_labels.append(reverse_map[dt.predict(feature)])
        print_matrices(test_labels, np.array(predict_labels))
