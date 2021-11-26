import json
import os

from imageLoader import load_image
from indexes.va import VA


def save_index(index, folder_path, file_name):
    index_details = {}
    index_details['b'] = index.b
    index_details['partition_per_dim'] = index.partition_per_dim
    index_details['labels'] = index.labels.tolist()
    index_details['data'] = index.data.tolist()
    index_details['min_values'] = index.min_values.tolist()
    index_details['max_values'] = index.max_values.tolist()
    index_details['lower_dim_data'] = index.lower_dim_data.tolist()
    index_json = json.dumps(index_details, indent=4)
    file_path = os.path.join(folder_path, file_name)
    if os.path.isfile(file_path):
        os.remove(file_path)
    with open(file_path, "w") as out_file:
        out_file.write(index_json)

def load_json(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as index_json:
            return json.load(index_json)
    return None

def load_va_index(file_path):
    index_json = load_json(file_path)
    index = VA(1)
    index.restore(index_json)
    return index

def get_top_t(folder_path, file_name, query, t):
    file_path = os.path.join(folder_path, file_name)
    index = load_va_index(file_path)
    return index.compare_top_k(query, t)
