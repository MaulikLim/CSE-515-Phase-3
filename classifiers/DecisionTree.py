import numpy as np
from featureGenerator import save_features_to_json, save_object

from featureLoader import load_json, load_object
class Node:
    def __init__(self, index = 0, val = 0, prediction = None) -> None:
        self.index = index
        self.val = val
        self.left = None
        self.right = None
        self.prediction = prediction

class DecisionTree:
    
    def __init__(self, features, labels, path, model, k):
        self.data = np.c_[features,labels]
        self.max_depth = 50
        self.max_size = 100
        self.head = None
        self.classes = set(labels)
        self.path = path
        self.model = model
        self.k=k

    def split(self, data, val, ind):
        l = []
        r = []
        for row in data:
            if row[ind]<val:
                l.append(row)
            else:
                r.append(row)
        return l,r
    
    def getGiniWeasly(self, l, r):
        lSize = len(l)
        rSize = len(r)
        total_samples = lSize+rSize
        lScore = 0.0
        gini = 0.0
        rScore = 0.0
        labCnt = {}
        for row in l:
            if row[-1] not in labCnt:
                labCnt[row[-1]]=0
            labCnt[row[-1]]+=1
        if lSize!=0:
            for label in self.classes:
                if label in labCnt:
                    p = labCnt[label]/lSize
                    lScore += p * p
        labCnt = {}
        for row in r:
            if row[-1] not in labCnt:
                labCnt[row[-1]]=0
            labCnt[row[-1]]+=1
        if rSize!=0:
            for label in self.classes:
                if label in labCnt:
                    p = labCnt[label]/rSize
                    rScore += p * p
        gini = (1-lScore)*(lSize/total_samples)
        gini += (1-rScore)*(rSize/total_samples)
        return gini
    
    def getSplit(self, data):
        cur_gini = 1e9
        index, val = None, None
        _l, _r = [], []
        for i in range(len(data[0])-1):
            for row in data:
                l,r = self.split(data,row[i],i)
                gini = self.getGiniWeasly(l,r)
                if gini < cur_gini:
                    cur_gini = gini
                    _l = l
                    _r = r
                    index = i
                    val = float(row[i])
        # print(head.val, end=", ")
        return index,val,_l,_r

    def buildTree(self, data, d):
        node = Node()
        if d >= self.max_depth or len(data) < self.max_size:
            labs = [row[-1] for row in data]
            node.prediction = max(set(labs),key = labs.count)
            return node
        index, val, l, r = self.getSplit(data)
        node.val = val
        node.index = index
        node.left = self.buildTree(l,d+1)
        node.right = self.buildTree(r,d+1)
        return node


    def train(self):
        # self.head = self.getNode(self.data)
        # print("head done.")
        file_name = self.model + "_" + str(self.k) + ".pkl"
        # print('Loading features from path '+folder_path)
        self.head = load_object(self.path + '/' + file_name)
        if self.head is None:
            self.head = self.buildTree(self.data,1)
            save_object(self.path,self.head,file_name)
    
    def isLeaf(self, node):
        return node.left!=None and node.right!=None

    def predict(self, testFeature):
        root = self.head
        # while not self.isLeaf(root):
        while root.prediction == None:
            # print(testFeature[root.index],root.index,root.val,end = ", ")
            if testFeature[root.index]<root.val:
                root = root.left
            else:
                root = root.right
        return root.prediction





