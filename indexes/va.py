import numpy as np
import heapq
import sys


class VA:

    def __init__(self, b):
        self.b = b
        self.partition_per_dim = 2 ** b

    def populate_index(self, labels, data):
        self.labels = labels
        self.data = data
        self.min_values, self.max_values = self.get_min_max()
        self.lower_dim_data = self.transform_data()
        # print(self.lower_dim_data)
        data_size = sys.getsizeof(self.labels) + data.nbytes
        print("Size of original data: " + str(data_size))
        lower_dim_size = self.b * len(self.data) * len(self.data[0]) / 8
        print("Size of lower dim data: " + str(lower_dim_size))
        metadata_size = self.min_values.nbytes + self.max_values.nbytes
        print("Metadata size: " + str(metadata_size))
        print("Total index size: " +
              str(data_size + lower_dim_size + metadata_size))

    def get_min_max(self):
        min_data = np.amin(self.data, axis=0)
        max_data = np.amax(self.data, axis=0)
        return [min_data, max_data]

    def transform_data(self):
        transformed_data = []
        for x in self.data:
            transformed_data.append(self.get_lower_dim_rep(x))
        return np.array(transformed_data)

    def get_lower_dim_rep(self, vector):
        rep = []
        for x in range(len(vector)):
            rep.append(self.get_dim_partition(
                vector[x], self.min_values[x], self.max_values[x]))
        return rep

    def get_dim_partition(self, dim_value, min_val, max_val):
        if dim_value < min_val:
            return 0
        elif dim_value >= max_val:
            return self.partition_per_dim - 1

        step = (max_val - min_val)/self.partition_per_dim
        rel_val = dim_value - min_val
        return rel_val // step
        # curr_step = min_val + step
        # partition = 0
        # while dim_value > curr_step and partition < (self.partition_per_dim - 1):
        #     curr_step += step
        #     partition += 1
        # return partition

    def cal_lower_bound(self, query_vector, query_low_dim, target_low_dim):
        arr = np.array([self.cal_dim_lower_bound(query_vector[x], query_low_dim[x],
                                                 target_low_dim[x], self.min_values[x], self.max_values[x]) for x in range(len(query_vector))])
        return np.linalg.norm(arr)

    def cal_dim_lower_bound(self, query_dim_value, query_dim_part, target_low_dim_part, dim_min_value, dim_max_value):
        if query_dim_part == target_low_dim_part:
            return 0
        step = (dim_max_value - dim_min_value)/self.partition_per_dim
        temp = step * target_low_dim_part + dim_min_value

        if query_dim_part > target_low_dim_part:
            return query_dim_value - step - temp
        else:
            return temp - query_dim_value

    def cal_l2_distance(self, query, target):
        return np.linalg.norm(query - target)

    def get_top_k(self, query, k):
        distances = np.array([self.cal_l2_distance(query, self.data[i])
                              for i in range(len(self.labels))])
        ans = np.argsort(distances)[:k]
        return [self.labels[x] for x in ans]

    def get_top_k_low_dim(self, query, k, actual_results):
        query_low_dim = self.get_lower_dim_rep(query)
        result = []
        candidates = 0
        false_pos = 0
        for i in range(len(self.labels)):
            if len(result) < k:
                if self.labels[i] not in actual_results:
                    false_pos += 1
                heapq.heappush(
                    result, (-self.cal_l2_distance(query, self.data[i]), self.labels[i]))
                candidates += 1
            else:
                curr_largest = result[0]
                lower_bound = self.cal_lower_bound(
                    query, query_low_dim, self.lower_dim_data[i])
                if lower_bound < -curr_largest[0]:
                    candidates += 1
                    if self.labels[i] not in actual_results:
                        false_pos += 1
                    dist = self.cal_l2_distance(query, self.data[i])
                    if dist < -curr_largest[0]:
                        heapq.heappop(result)
                        heapq.heappush(result, (-dist, self.labels[i]))
        print("Number of unique and overall images considered: " + str(candidates))
        print("False positive: " + str(false_pos))
        print(result)
        result = [(result[x][1], -result[x][0])
                  for x in range(-1, -len(result)-1, -1)]
        index_results = [x[0] for x in result]
        miss = 0
        for x in actual_results:
            if x not in index_results:
                miss += 1
        print("Miss: "+str(miss))
        result.sort(key=lambda x: x[1])
        print(result)

    def compare_top_k(self, query, k):
        actual_results = self.get_top_k(query, k)
        print(actual_results)
        self.get_top_k_low_dim(query, k, actual_results)
