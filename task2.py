from classifier.svm import SVM
from sklearn.preprocessing import MinMaxScaler
from classifiers.DecisionTree import DecisionTree
from featureGenerator import save_features_to_json
from classifier.ppr import Personalised_Page_Rank
import imageLoader
import modelFactory
import argparse
import json
import numpy as np
import pdb
import featureLoader
import latentFeatureGenerator;
from metrics_utils import print_matrices
import math
from utilities import print_semantics_sub, print_semantics_type
import time

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
        # Train PPR
        types_of_labels = list(set(labels))
        ppr = Personalised_Page_Rank(20, types_of_labels)
        ppr.fit(features, labels)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [x.split("-")[2] for x in test_data[1]]
        test_predicted_labels = ppr.predict(test_features, test_labels)
        # print(ppr.accuracy(test_predicted_labels, test_labels))
        print(types_of_labels)
        print_matrices(test_labels, np.array(test_predicted_labels))
        # for test_image,test_predicted_label in test_predicted_labels.items():
        #     print(test_image,"->",test_predicted_label)
        pass
    elif classifier == 'svm':
        # Train SVM
        svm = SVM()
        labels = [int(x)-1 for x in labels]
        min_max_scalar = MinMaxScaler()
        features = min_max_scalar.fit_transform(features)
        svm.train(np.array(features), np.array(labels), -1, 10000, 5e-2, 1e-5, verbose=False)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [int(x.split("-")[2]) - 1 for x in test_data[1]]
        test_features = min_max_scalar.transform(test_features)
        test_predictions = svm.predict(test_features)
        print(set([x+1 for x in labels]))
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
        start_time = time.time()
        dt = DecisionTree(features,np.array(labels), int(3*math.log(len(labels))))
        dt.train()

        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [x.split("-")[2] for x in test_data[1]]

        predict_labels = []
        for i,feature in enumerate(test_features):
            predict_labels.append(reverse_map[dt.predict(feature)])
        print(labels_set)
        print_matrices(test_labels, np.array(predict_labels))
        # print("--- %s seconds ---" % (time.time() - start_time))
        pass
    # load query data to which we are supposed to assign labels
    # query_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
    # assign subject using the classifier above
