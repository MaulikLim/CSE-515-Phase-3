import numpy as np

class Node:
    def __init__(self, index = 0, val = 0, prediction = None) -> None:
        self.index = index
        self.val = val
        self.left = None
        self.right = None
        self.prediction = prediction

class DecisionTree:
    
    def __init__(self, features, labels, max_depth=15):
        self.data = np.c_[features,labels]
        self.max_depth = max_depth
        self.head = None
        self.classes = len(set(labels))
    
    def getGiniWeasly(self, mat, size):
        return 1.0 - sum((mat[x]/size) ** 2 for x in range(self.classes))
    
    def getSplit(self, data):
        size = len(data)
        r_mat_p = [np.sum(data[:,-1]==c) for c in range(self.classes)]
        cur_gini = self.getGiniWeasly(r_mat_p,size)
        index, val = None, None
        inds = np.random.randint(len(data[0])-1,size = len(data[0])//4)
        for ind in inds:
            lim, cl = zip(*sorted(zip(data[:,ind],data[:,-1])))
            l_mat = [0] * self.classes
            r_mat = r_mat_p.copy()
            for i in range(1, size):
                c = int(cl[i-1])
                l_mat[c]+=1
                r_mat[c]-=1
                l_gini = self.getGiniWeasly(l_mat,i)
                r_gini = self.getGiniWeasly(r_mat,size-i)
                gini = (i*l_gini + (size-i)*r_gini)/size
                if lim[i]==lim[i-1]:
                    continue
                if gini<cur_gini:
                    cur_gini = gini
                    index = ind
                    val = (lim[i-1]+lim[i])/2
        return index,val

    def buildTree(self, data, d):
        node = Node()
        node.prediction = np.argmax([np.sum(data[:,-1]==i) for i in range(self.classes)])
        if d<self.max_depth and len(data)>1:
            index, val = self.getSplit(data)
            if index is not None:
                node.val = val
                node.index = index
                left = data[:,index]<val
                l = data[left]
                r = data[~left]
                node.left = self.buildTree(l,d+1)
                node.right = self.buildTree(r,d+1)
        return node


    def train(self):
        self.head = self.buildTree(self.data,0)

    def predict(self, testFeature):
        root = self.head
        while root.left:
            if testFeature[root.index]<root.val:
                root = root.left
            else:
                root = root.right
        return root.prediction





