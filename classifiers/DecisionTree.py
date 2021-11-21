import numpy as np


class Node:
    def __init__(self, index=0, val=0, l=None, r=None) -> None:
        self.index = index
        self.val = val
        self.l = l if l else []
        self.r = r if r else []
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, features, labels):
        self.data = np.c_[features, labels]
        self.max_depth = 500
        self.max_size = 1
        self.head = None

    def split(self, data, val, ind):
        l = []
        r = []
        for row in data:
            if row[ind] <= val:
                l.append(row)
            else:
                r.append(row)
        return l, r

    def getGiniWeasly(self, l, r, labels):
        labels = set(labels)
        lSize = len(l)
        rSize = len(r)
        total_samples = lSize + rSize
        lScore = 0.0
        gini = 0.0
        rScore = 0.0
        for label in labels:
            if lSize != 0:
                p = [row[-1] for row in l].count(label) / lSize
                lScore += p * p
            if rSize != 0:
                p = [row[-1] for row in r].count(label) / rSize
                rScore += p * p
        gini = (1 - lScore) * (lSize / total_samples)
        gini += (1 - rScore) * (rSize / total_samples)
        return gini

    def getNode(self, data):
        labels = [x[-1] for x in data]
        head = Node()
        cur_gini = 1e9
        for i in range(len(data[0]) - 1):
            for row in data:
                l, r = self.split(data, row[i], i)
                gini = self.getGiniWeasly(l, r, labels)
                if gini < cur_gini:
                    cur_gini = gini
                    head.l = l
                    head.r = r
                    head.index = int(i)
                    head.val = float(row[i])
        return head

    def buildTree(self, root, d):
        l = root.l
        r = root.r
        if d > self.max_depth:
            return
        if len(l) <= self.max_size:
            pass
        else:
            root.left = self.getNode(l)
            self.buildTree(root.left, d + 1)
        if len(r) <= self.max_size:
            pass
        else:
            root.right = self.getNode(r)
            self.buildTree(root.right, d + 1)

    def train(self):
        self.head = self.getNode(self.data)
        print("head done.")
        self.buildTree(self.head, 1)

    def isLeaf(self, node):
        return node.left is not None and node.right is not None

    def predict(self, testFeature):
        root = self.head
        # while not self.isLeaf(root):
        while root is not None:
            if testFeature[root.index] <= root.val:
                if root.left is None:
                    labs = [x[-1] for x in root.l] + [x[-1] for x in root.r]
                    # return max(set(labs),key = labs.count)
                    return set(labs)
                else:
                    root = root.left
            else:
                if root.right is None:
                    labs = [x[-1] for x in root.l] + [x[-1] for x in root.r]
                    # return max(set(labs),key = labs.count)
                    return set(labs)
                else:
                    root = root.right
        return "Something unexpected happened!"
