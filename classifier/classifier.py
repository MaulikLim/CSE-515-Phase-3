class Classifiers:

    def classifier(classifier,**args):
        #from .svm import SVM
        #from .decisionTree import Decision_Tree
        #from .ppr import PPR
        classifiers = {"svm":SVM,"decision_tree":Decision_Tree,"ppr":Personalised_Page_Rank}
        if classifier.lower() in classifiers:
            return classifiers[classifier](**args)
        else:
            raise Exception("Classifier not present")