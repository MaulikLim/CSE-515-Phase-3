import os
from classifier.svm import SVM
from classifiers.DecisionTree import DecisionTree
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import indexes.vaUtility as va_utility
import latentFeatureGenerator
import imageLoader

rel_imgs = {}
irrel_imgs = {}
min_max_scalar = MinMaxScaler()
def getImageFeature(index_file_name, index_folder_path, query_image_name):
    cmp = index_file_name.split("_")
    feature_model = cmp[-3]
    k = int(cmp[-2])
    return latentFeatureGenerator.compute_latent_feature(index_folder_path, query_image_name,feature_model,k)

def take_feedback(features, names):
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
        ind = int(ind)-1
        cur_rel[names[ind]] = features[ind]
    for ind in irrel:
        ind = int(ind)-1
        cur_irrel[names[ind]] = features[ind]
    rel_imgs.update(cur_rel)
    for name in cur_irrel.keys():
        if name in rel_imgs:
            rel_imgs.pop(name,None)
    irrel_imgs.update(cur_irrel)
    for name in cur_rel.keys():
        if name in irrel_imgs:
            irrel_imgs.pop(name,None)
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

    if model== 'DT':
        # call task 6 and return classifier
        dt = DecisionTree(training_features,np.array(training_labels))
        dt.train()
        return dt, model
    else:
        # call task 7 and return classifier
        features = min_max_scalar.fit_transform(training_features)
        svm = SVM()
        svm.train(np.array(features), np.array(training_labels), 2, 10000, 1e-3, 1e-5, verbose=False)
        return svm, model

index_folder_path = input("Index folder path\n")
index_file_name = input("Index file name\n")
query_image = input("Query Image path\n")
t = int(input("t\n"))
if index_file_name.startswith("index_va"):
    # call task 5
    query = getImageFeature(index_file_name, index_folder_path, query_image)
    result = va_utility.get_top_t(index_folder_path, index_file_name,query,t)
    features = []
    labels = []
    for res in result:
        features.append(res[2])
        labels.append(res[0])
    #show images in labels list
    i=1
    for img in labels:
        print(str(i)+".",img)
        imageLoader.show_image(os.path.join(index_folder_path,img))
        i+=1
    j=1
    while True:
        classifier, model = take_feedback(features,labels)
        result = va_utility.get_top_t(index_folder_path, index_file_name,query,j*10*t)
        # print(np.array(result)[:,0])
        features = []
        labels = []
        # call task 5 again with 5*t
        # classify the result using above classifier and output top t images
        new_result = []
        for res in result:
            lab = 0
            if model == 'DT':
                lab = classifier.predict(res[2])
            else:
                lab = classifier.predict(min_max_scalar.fit_transform([res[2]]))
                print(lab)
            if lab==1:
                features.append(res[2])
                labels.append(res[0])
                new_result.append(res[0])
                if len(new_result)==t:
                    break
        #show new results
        i=1
        for img in new_result:
            print(str(i)+".",img)
            imageLoader.show_image(os.path.join(index_folder_path,img))
            i+=1
        quit = input("Do you want to continue? Y/N\n")
        if quit=='N':
            break
        j+=1
else:
    query = getImageFeature(index_file_name, index_folder_path, query_image)
    #change below lines for lsh
    result = va_utility.get_top_t(index_folder_path, index_file_name,query,t)
    features = []
    labels = []
    for res in result:
        features.append(res[0])
        labels.append(res[2])
    #show images in labels list
    classifier = take_feedback(features)
    result = va_utility.get_top_t(index_file_name,query,5*t)
    # call task 5 again with 5*t
    # classify the result using above classifier and output top t images
    new_result = []
    for res in result:
        lab = classifier.predict(res[0])
        if lab==1:
            new_result.append(res[2])
            if len(new_result)==t:
                break
    #show new results