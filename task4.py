# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 10:32:07 2021

@author: kbarhanp
"""

import numpy as np
import featureLoader
import imageLoader
from indexes.LSH import LSH
import modelFactory
from arg_parser_util import Parser
import latentFeatureGenerator
import os
import json
import pdb

parser = Parser("Task 4")
parser.add_args("-fp", "--folder_path", str, True)
parser.add_args("-f", "--feature_model", str, True)
<<<<<<< HEAD
=======
# parser.add_args("-qp", "--query_path", str, True)
>>>>>>> 10d9020c83765f60d45eaed4fa71870359a6f359
parser.add_args("-l", "--l", int, True)
parser.add_args("-k", "--k", int, True)
parser.add_args("-kh", "--khash", int, True)
<<<<<<< HEAD
=======
# parser.add_args("-t", "--top_t", int, True)
>>>>>>> 10d9020c83765f60d45eaed4fa71870359a6f359
args = parser.parse_args()

data, labels = latentFeatureGenerator.compute_latent_features(args.folder_path, args.feature_model, k)
if data is not None:
    file_name = 'index_lsh_' + args.feature_model+'_'+ str(k) + '_' + str(args.khash) + '_' + str(args.l) + '.json'
    file_path = os.path.join(args.folder_path, file_name)
    index = LSH(args.l,args.khash)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as index_json:
            index.restore(json.load(index_json),args.l,args.khash)
    else:    
        index.populate_index(labels, data)
        index_json = json.dumps(index.finalfile, indent=4)
        if os.path.isfile(file_path):
            os.remove(file_path)
        with open(file_path, "w") as out_file:
            out_file.write(index_json)