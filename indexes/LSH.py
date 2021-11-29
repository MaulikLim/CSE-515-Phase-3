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
            binary = data.dot(randv) >= 0
            table = defaultdict(list)
            for binaryArr,labl in zip(binary.astype(int), labels):
                binaryrepr = (str.encode(''.join(str(binaryArr))).decode("utf-8"))
                table[binaryrepr[1:-1]].append(labl)
            savetable = {"randomvectors" : randv.tolist(), "table" : table}
            savefile.append(savetable)
        label_data = {}
        for label,data in zip(labels, data):
            label_data[label] = data.tolist()
        self.finalfile["datavectors"] = label_data
        self.finalfile["lshdata"] = savefile
        data_size = labels.nbytes + data.nbytes
        print("Size of original data: " + str(data_size))
        print("Total index size: " + str(np.array(savefile).nbytes))
    
    def restore(self, index_json,l,k):
        self.l = l
        self.k = k
        self.finalfile = index_json
    
    def cal_l2_distance(self, query, target):
        return np.linalg.norm(query - target)
    
    def cal_distance(self, query, target, feature_model):
            return np.sum(np.absolute(query - target))

    def get_top_t(self, query, t, index_file_path=""):
        if(len(index_file_path) > 0):
            if os.path.isfile(index_file_path):
                with open(index_file_path, 'r') as index_json:
                    self.finalfile = json.load(index_json)
        ret = set()
        buckets = 0
        Level = 0
        while (len(ret) < t):
            for hash_table in self.finalfile["lshdata"]:
                table = hash_table['table']
                random_vectors = np.array(hash_table['randomvectors'])
                binary_repr = query.dot(random_vectors) >= 0
                binary_idx = (str.encode(''.join(str(binary_repr.astype(int)))).decode("utf-8"))[1:-1]
                if (Level > 0):
                    for hash_table_idx in table.keys():
                        xor = bin(int(binary_idx, 2) ^ int(hash_table_idx, 2))[2:]
                        Change = xor.encode().count(b'1')
                        if (Change == Level):
                            buckets += 1
                            ret.update(table[hash_table_idx])
                        if (len(ret) >= t):
                            break
                    if (len(ret) >= t):
                        break
                else:
                    if (table[binary_idx]):
                        buckets += 1
                        ret.update(table[binary_idx])
            Level += 1
            if (Level == self.k):
                break
        retrieval = list(ret)
        false_pos = (len(retrieval)-t)/len(retrieval)
        datavectors = self.finalfile["datavectors"]
        distancedict = {}
        pdb.set_trace()
        for id in retrieval:
            if(index_file_path.split("index_lsh_")[1][:2] != "cm"):
                distancedict[id] = self.cal_l2_distance(query, np.array(datavectors[id]))
            else:
                distancedict[id] = self.cal_distance(query, np.array(datavectors[id]))
        sorted_d = dict( sorted(distancedict.items(), key=operator.itemgetter(1)))
        sorted_dict = OrderedDict()
        for k, v in sorted_d.items():
            sorted_dict[k] = v
        sorted_list = list(sorted_dict.keys())
        labels = list(datavectors.keys())
        data = list(datavectors.values())
        if(index_file_path.split("index_lsh_")[1][:2] != "cm"):
            distances = np.array([self.cal_l2_distance(query, data[i])
                              for i in range(len(labels))])
        else:
            distances = np.array([self.cal_distance(query, data[i])
                              for i in range(len(labels))])
        ans = np.argsort(distances)[:t]
        expected = [labels[x] for x in ans]
        miss = 0
        index_results = sorted_list[:t]
        for x in expected:
            if x not in index_results:
                miss += 1
        print("Number of unique and overall images considered: " + str(len(retrieval)))
        print("Number of buckets searched: "+str(buckets))
        print("False positive: " + str(false_pos))
        print("Miss: "+str(miss))
        return index_results