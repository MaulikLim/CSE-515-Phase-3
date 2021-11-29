import numpy as np
import featureLoader
import imageLoader
import modelFactory
import indexes.vaUtility as va_utility
from indexes.va import VA
from arg_parser_util import Parser
import latentFeatureGenerator
import os

parser = Parser("Task 1")
parser.add_args("-fp", "--folder_path", str, True)
parser.add_args("-f", "--feature_model", str, True)
parser.add_args("-k", "--k", int, True)
parser.add_args("-b", "--bits_per_dimension", int, True)
# parser.add_args("-q", "--query_path", str, True)
# parser.add_args("-t", "--t", int, True)
args = parser.parse_args()

# labels, data = featureLoader.load_features_for_model(args.folder_path, args.feature_model)
# k = -1
# if args.k != 'all':
#     k = int(args.k)
data, labels = latentFeatureGenerator.compute_latent_features(args.folder_path, args.feature_model, args.k)

if data is not None:
    index = VA(args.bits_per_dimension)
    index.populate_index(labels, data)
    file_name = 'index_va_' + args.feature_model + '_' + str(args.k) + '_' + str(args.bits_per_dimension) + '.json'
    va_utility.save_index(index, args.folder_path, file_name)
    # index = va_utility.load_va_index(os.path.join(args.folder_path, file_name))
    # query = modelFactory.get_model(args.feature_model).compute_features(imageLoader.load_image(args.query_path))
    # va_utility.get_top_t(args.folder_path, file_name, query, args.t, args.feature_model)
