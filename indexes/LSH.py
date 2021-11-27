# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 09:52:14 2021

@author: kbarhanp
"""

import numpy as np
import heapq
import sys
import pdb
import operator
from collections import OrderedDict
from collections import defaultdict
import os
import json

class LSH:

    def __init__(self, L,k):
        self.L = L
        self.k = k
        self.finalfile = {}

    def populate_index(self, labels, data):
        self.labels = labels
        self.data = data
        savefile = []
        num_rows, num_cols = data.shape
        for i in range(self.L):
            randv = np.random.randn(num_cols, self.k)
            binary_repr = data.dot(randv) >= 0
            table = defaultdict(list)
            for binaryArr,labl in zip(binary_repr.astype(int), labels):
                binaryrepr = (str.encode(''.join(str(binaryArr))).decode("utf-8"))
                table[binaryrepr[1:-1]].append(labl)
            savetable = {"randomvectors" : randv.tolist(), "table" : table}
            savefile.append(savetable)
        label_data = {}
        for label,data in zip(labels, data):
            label_data[label] = data.tolist()
        self.finalfile["datavectors"] = label_data
        self.finalfile["lshdata"] = savefile
    
    def restore(self, index_json,l,k):
        self.l = l
        self.k = k
        self.finalfile = index_json
    
    def cal_l2_distance(self, query, target):
        return np.linalg.norm(query - target)
    
    def get_top_t(self, query, t, index_file_path=""):
        if(len(index_file_path) > 0):
            if os.path.isfile(index_file_path):
                with open(index_file_path, 'r') as index_json:
                    self.finalfile = json.load(index_json)
        retrieval = set()
        n_buckets = 0
        n_candidates = 0
        permLevel = 0
        while (len(retrieval) < t):
            for hash_table in self.finalfile["lshdata"]:
                table = hash_table['table']
                random_vectors = np.array(hash_table['randomvectors'])
                binary_repr = query.dot(random_vectors) >= 0
                binary_idx = (str.encode(''.join(str(binary_repr.astype(int)))).decode("utf-8"))[1:-1]
                if (permLevel > 0):
                    for hash_table_idx in table.keys():
                        xorNum = bin(int(binary_idx, 2) ^ int(hash_table_idx, 2))[2:]
                        bitChanges = xorNum.encode().count(b'1')
                        if (bitChanges == permLevel):
                            n_buckets += 1
                            retrieval.update(table[hash_table_idx])
                        if (len(retrieval) >= t):
                            break
                    if (len(retrieval) >= t):
                        break
                else:
                    if (table[binary_idx]):
                        n_buckets += 1
                        retrieval.update(table[binary_idx])
            permLevel += 1
            if (permLevel == self.k):
                break
        retrieval = list(retrieval)
        false_pos = len(retrieval)-t
        datavectors = self.finalfile["datavectors"]
        distancedict = {}
        for id in retrieval:
            distancedict[id] = self.cal_l2_distance(query, np.array(datavectors[id]))
        sorted_d = dict( sorted(distancedict.items(), key=operator.itemgetter(1)))
        sorted_dict = OrderedDict()
        for k, v in sorted_d.items():
            sorted_dict[k] = v
        sorted_list = list(sorted_dict.keys())
        labels = list(datavectors.keys())
        data = list(datavectors.values())
        distances = np.array([self.cal_l2_distance(query, data[i])
                              for i in range(len(labels))])
        ans = np.argsort(distances)[:t]
        expected = [labels[x] for x in ans]
        miss = 0
        index_results = sorted_list[:t]
        for x in expected:
            if x not in index_results:
                miss += 1
        print("Number of buckets searched: "+str(n_buckets))
        print("False positive: " + str(false_pos))
        print("Miss: "+str(miss))
        return index_results
            
