import numpy as np
import featureLoader
import imageLoader
import modelFactory
from indexes.va import VA
from arg_parser_util import Parser

parser = Parser("Task 1")
parser.add_args("-fp", "--folder_path", str, True)
parser.add_args("-f", "--feature_model", str, True)
parser.add_args("-b", "--bits_per_dimension", int, True)
parser.add_args("-q", "--query_path", str, True)
parser.add_args("-t", "--t", int, True)
args = parser.parse_args()

labels, data = featureLoader.load_features_for_model(args.folder_path, args.feature_model)

if data is not None:
    index = VA(args.bits_per_dimension)
    index.populate_index(labels, data)
    query = modelFactory.get_model(args.feature_model).compute_features(imageLoader.load_image(args.query_path))
    index.compare_top_k(query, args.t)
    
