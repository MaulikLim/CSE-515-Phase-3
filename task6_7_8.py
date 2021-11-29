import os
from classifier.svm import SVM
from classifiers.DecisionTree import DecisionTree
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from featureLoader import load_json
from indexes.LSH import LSH
import indexes.vaUtility as va_utility
import latentFeatureGenerator
import imageLoader

rel_imgs = {}
irrel_imgs = {}
min_max_scalar = None

index_folder_path = input("Index folder path\n")
index_file_name = input("Index file name\n")
query_image = input("Query Image path\n")
t = int(input("t\n"))

def getImageFeature(index_file_name, index_folder_path, query_image_name):
    cmp = index_file_name.split("_")
    feature_model = cmp[-3]
    k = int(cmp[-2])
    return latentFeatureGenerator.compute_latent_feature(index_folder_path, query_image_name, feature_model, k)

def getImageFeatureLSH(index_file_name, index_folder_path, query_image_name):
    cmp = index_file_name.split("_")
    feature_model = cmp[2]
    k = int(cmp[3])
    return latentFeatureGenerator.compute_latent_feature(index_folder_path, query_image_name, feature_model, k)

def take_feedback(features, names):
    global min_max_scalar
    relavent_images = input("Which images do you find relavent? (e.g. '1,4,5')\n")
    irrelavent_images = input("Which images do you find irrelavent? (e.g. '2,3')\n")
    model = input("which classifier you want to use? (e.g. DT or SVM)\n")
    rel = []
    if relavent_images != '':
        rel = relavent_images.split(",")
    irrel = []
    if irrelavent_images != '':
        irrel = irrelavent_images.split(",")
    cur_rel = {}
    cur_irrel = {}
    for ind in rel:
        ind = int(ind) - 1
        cur_rel[names[ind]] = features[ind]
    for ind in irrel:
        ind = int(ind) - 1
        cur_irrel[names[ind]] = features[ind]
    rel_imgs.update(cur_rel)
    for name in cur_irrel.keys():
        if name in rel_imgs:
            rel_imgs.pop(name, None)
    irrel_imgs.update(cur_irrel)
    for name in cur_rel.keys():
        if name in irrel_imgs:
            irrel_imgs.pop(name, None)
    training_features = []
    training_labels = []
    for key in rel_imgs.keys():
        training_features.append(rel_imgs[key])
        training_labels.append(1)
    for key in irrel_imgs.keys():
        training_features.append(irrel_imgs[key])
        training_labels.append(0)
    # print(rel_imgs.keys())
    # print(irrel_imgs.keys())

    if model == 'DT':
        # call task 6 and return classifier
        dt = DecisionTree(training_features, np.array(training_labels))
        dt.train()
        return dt, model
    else:
        # call task 7 and return classifier
        if not min_max_scalar:
            min_max_scalar = MinMaxScaler()
            features = min_max_scalar.fit_transform(training_features)
        else:
            features = min_max_scalar.transform(training_features)
        svm = SVM()
        svm.train(np.array(features), np.array(training_labels), 2, 10000, 1e-3, 1e-5, verbose=False)
        return svm, model

if index_file_name.startswith("index_va"):
    # call task 5
    query = getImageFeature(index_file_name, index_folder_path, query_image)
    feature_model = index_file_name.split("_")[-3]
    result = va_utility.get_top_t(index_folder_path, index_file_name, query, t,feature_model)
    features = []
    labels = []
    for res in result:
        features.append(res[2])
        labels.append(res[0])
    # show images in labels list
    i = 1
    for img in labels:
        print(str(i) + ".", img)
        imageLoader.show_image(os.path.join(index_folder_path, img))
        i += 1
    j = 1
    while True:
        if input("Do you want to provide feedback? Y/N\n") == 'N':
            break
        classifier, model = take_feedback(features, labels)
        result = va_utility.get_top_t(index_folder_path, index_file_name, query, j * 10 * t, feature_model)
        # print(np.array(result)[:,0])
        features = []
        labels = []
        # call task 5 again with 5*t
        # classify the result using above classifier and output top t images
        new_result = []
        irrel_predictions = []
        irrel_predictions_features = []
        for res in result:
            lab = 0
            if model == 'DT':
                lab = classifier.predict(res[2])
            else:
                lab = classifier.predict(min_max_scalar.transform([res[2]]))
                # print(lab)
            if lab == 1:
                features.append(res[2])
                labels.append(res[0])
                new_result.append(res[0])
                if len(new_result) == t:
                    break
            else:
                irrel_predictions.append(res[0])
                irrel_predictions_features.append(res[2])
        # show new results
        i = 1
        for img in new_result:
            print(str(i) + ".", img)
            imageLoader.show_image(os.path.join(index_folder_path, img))
            i += 1
        for j,img in enumerate(irrel_predictions):
            if i>t:
                break
            print(str(i) + ".", img)
            labels.append(img)
            features.append(irrel_predictions_features[j])
            imageLoader.show_image(os.path.join(index_folder_path, img))
            i+=1
        j += 1
else:
    query = getImageFeatureLSH(index_file_name, index_folder_path, query_image)
    cmp = index_file_name.split("_")
    l = cmp[5]
    k = int(cmp[4])
    lsh = LSH(l,k)
    lsh_index_file = load_json(os.path.join(index_folder_path, index_file_name))
    lsh.restore(lsh_index_file,l,k)
    # change below lines for lsh
    result = lsh.get_top_t( query, t,os.path.join(index_folder_path, index_file_name))
    features = []
    labels = []
    for res in result:
        features.append(lsh_index_file["datavectors"][res])
        labels.append(res)
    # show images in labels list
    i = 1
    for img in labels:
        print(str(i) + ".", img)
        imageLoader.show_image(os.path.join(index_folder_path, img))
        i += 1
    j = 1
    while True:
        if input("Do you want to provide feedback? Y/N\n") == 'N':
            break
        classifier, model = take_feedback(features, labels)
        result = lsh.get_top_t( query, j*10*t, os.path.join(index_folder_path, index_file_name))
        # print(np.array(result)[:,0])
        features = []
        labels = []
        # call task 5 again with 5*t
        # classify the result using above classifier and output top t images
        new_result = []
        irrel_predictions = []
        irrel_predictions_features = []
        for res in result:
            lab = 0
            if model == 'DT':
                lab = classifier.predict(lsh_index_file["datavectors"][res])
            else:
                lab = classifier.predict(min_max_scalar.transform([lsh_index_file["datavectors"][res]]))
                # print(lab)
            if lab == 1:
                features.append(lsh_index_file["datavectors"][res])
                labels.append(res)
                new_result.append(res)
                if len(new_result) == t:
                    break
            else:
                irrel_predictions.append(res)
                irrel_predictions_features.append(lsh_index_file["datavectors"][res])
        # show new results
        i = 1
        for img in new_result:
            print(str(i) + ".", img)
            imageLoader.show_image(os.path.join(index_folder_path, img))
            i += 1
        for j,img in enumerate(irrel_predictions):
            if i>t:
                break
            print(str(i) + ".", img)
            labels.append(img)
            features.append(irrel_predictions_features[j])
            imageLoader.show_image(os.path.join(index_folder_path, img))
            i+=1
        j += 1
