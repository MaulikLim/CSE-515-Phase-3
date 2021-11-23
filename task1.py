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
import latentFeatureGenerator;

from tech.PCA import PCA
from utilities import print_semantics_sub, print_semantics_type

parser = argparse.ArgumentParser(description="Task 1")
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
    labels = [x.split("-")[1] for x in data[1]]
    # train classifier as given in the input
    classifier = args.classifier.lower()
    #k=args.k
    if classifier == 'ppr':
        # Train PPR
        types_of_labels=["cc","jitter","neg","con","emboss","noise01","noise02","original","poster","rot","smooth","stipple"]
        ppr = Personalised_Page_Rank(20,types_of_labels)
        ppr.fit(features,labels)
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_labels = [x.split("-")[1] for x in test_data[1]]
        test_predicted_labels = ppr.predict(test_features,test_labels)
        print(ppr.accuracy(test_predicted_labels,test_labels))
        # for test_image,test_predicted_label in test_predicted_labels.items():
        #     print(test_image,"->",test_predicted_label)
        
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
        print(sum(test_predictions == test_labels)/len(test_labels))
    else:
        #Train decision tree
        dt = DecisionTree(features,labels, args.folder_path, args.feature_model, args.k)
        dt.train()
        test_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
        test_features = test_data[0]
        test_lables = [x.split("-")[1] for x in test_data[1]]
        i=0
        acc = 0
        for test_feature in test_features:
            lab = dt.predict(test_feature)
            if lab==test_lables[i]:
                acc+=1
            # print(lab, test_lables[i])
            i+=1
        print(acc*100/i)
        pass
    # print(labels.shape)
    # load query data to which we are supposed to assign labels
    # query_data = latentFeatureGenerator.compute_latent_features(args.query_folder, args.feature_model, args.k)
    # assign types using the classifier above
