import numpy as np
import time, math, heapq, tables
from collections import deque
from sklearn.metrics.pairwise import euclidean_distances
from six import string_types
import grouping_utils as gutils
import constants as cnt

valid_fn_names = ['max_min_mid_split', 'max_min_median_split', 'max_variance_split']

def max_min_mid_split(vectors):
    maxs = np.max(vectors, axis=0)
    mins = np.min(vectors, axis=0)

    split_axis = np.argmax(maxs-mins)
    split_val = 0.5 * (maxs[split_axis] + mins[split_axis])
    
    return split_axis, split_val


def max_min_median_split(vectors):
    maxs = np.max(vectors, axis=0)
    mins = np.min(vectors, axis=0)

    split_axis = np.argmax(maxs-mins)
    split_val = np.median(vectors[:,split_axis])
    
    return split_axis, split_val


def max_variance_split(vectors):
    variances = np.var(vectors, axis=0)

    split_axis = np.argmax(variances)
    split_val = np.median(vectors[:,split_axis])
    
    return split_axis, split_val


def get_split(vectors, algorithm='max_min_median_split'):
    if isinstance(algorithm, string_types) and algorithm in valid_fn_names:
        return eval(algorithm)(vectors)
    return None, None


class Node(object):
    def __init__(self, split_axis=None, split_val=None):
        self.split_axis = split_axis
        self.split_val = split_val
        self.left, self.right = None, None
        
        
class LeafNode(object):
    def __init__(self, indices):
        self.indices = indices
        
        
class KDTree(object):
    def __init__(self, vectors, leafsize=10, algorithm='max_min_median_split'):
        self.leaf_size = leafsize
        self.tree = None
        self.vectors = vectors
        self.algorithm = algorithm
        
    def construct(self):
        root_indices = np.arange(self.vectors.shape[0])

        if self.vectors.shape[0] <= self.leaf_size:
            self.tree = LeafNode(root_indices)
        else:
            self.tree = Node()
            queue_obj = deque([(self.tree, root_indices, None, None)])

            while len(queue_obj) > 0:
                curr_obj, indices, parent_obj, direction = queue_obj.popleft()

                if isinstance(curr_obj, Node):
                    split_axis, split_val = get_split(self.vectors[indices,:], self.algorithm)
                    
                    if split_axis is None:
                        return "Incorrect splitting algorithm specified"
                    
                    vec = self.vectors[indices, split_axis]

                    l_indices = indices[np.nonzero(vec <= split_val)[0]]
                    r_indices = indices[np.nonzero(vec > split_val)[0]]
                    
                    if len(r_indices) == 0 or len(l_indices) == 0:
                        if parent_obj is not None:
                            if direction == 0:
                                parent_obj.left = LeafNode(indices)
                            else:
                                parent_obj.right = LeafNode(indices)
                        else:
                            self.tree = LeafNode(indices)
                            break
                        
                    else:
                        curr_obj.split_axis = split_axis
                        curr_obj.split_val = split_val

                        if len(l_indices) <= self.leaf_size:
                            l_node_obj = LeafNode(l_indices)
                        else:
                            l_node_obj = Node()

                        if len(r_indices) <= self.leaf_size:
                            r_node_obj = LeafNode(r_indices)
                        else:
                            r_node_obj = Node()

                        curr_obj.left, curr_obj.right = l_node_obj, r_node_obj

                        queue_obj.append((l_node_obj, l_indices, curr_obj, 0))
                        queue_obj.append((r_node_obj, r_indices, curr_obj, 1))
                    
    
    def query_count(self, query_vector, k=5):
        max_heap, visited = [], set()
        node_stack = [self.tree]

        while len(node_stack) > 0:
            curr_obj = node_stack[-1]

            if isinstance(curr_obj, LeafNode):
                distances = euclidean_distances([query_vector], self.vectors[curr_obj.indices,:])[0]
                for dist, idx in zip(distances, curr_obj.indices):
                    if len(max_heap) < k:
                        heapq.heappush(max_heap, (-dist, idx))
                    else:
                        if dist < -max_heap[0][0]:
                            heapq.heappop(max_heap)
                            heapq.heappush(max_heap, (-dist, idx))

                visited.add(curr_obj)
                node_stack.pop()

            else:
                split_axis, split_val = curr_obj.split_axis, curr_obj.split_val

                if query_vector[split_axis] <= split_val:
                    if curr_obj.left not in visited:
                        node_stack.append(curr_obj.left)
                    else:
                        max_dist = -max_heap[0][0]
                        if max_dist > abs(query_vector[split_axis]-split_val) and curr_obj.right not in visited:
                            node_stack.append(curr_obj.right)
                        else:
                            visited.add(curr_obj)
                            node_stack.pop()

                else:
                    if curr_obj.right not in visited:
                        node_stack.append(curr_obj.right)
                    else:
                        max_dist = -max_heap[0][0]
                        if max_dist > abs(query_vector[split_axis]-split_val) and curr_obj.left not in visited:
                            node_stack.append(curr_obj.left)
                        else:
                            visited.add(curr_obj)
                            node_stack.pop()

        output = [(-d, i) for d, i in max_heap]
        return output
    
    def query_radius(self, query_vector, radius=0.1):
        output, visited = [], set()
        node_stack = [self.tree]

        while len(node_stack) > 0:
            curr_obj = node_stack[-1]

            if isinstance(curr_obj, LeafNode):
                distances = euclidean_distances([query_vector], self.vectors[curr_obj.indices,:])[0]
                for dist, idx in zip(distances, curr_obj.indices):
                    if dist <= radius:
                        output.append((dist, idx))

                visited.add(curr_obj)
                node_stack.pop()

            else:
                split_axis, split_val = curr_obj.split_axis, curr_obj.split_val

                if query_vector[split_axis] <= split_val:
                    if curr_obj.left not in visited:
                        node_stack.append(curr_obj.left)
                    else:
                        if radius >= abs(query_vector[split_axis]-split_val) and curr_obj.right not in visited:
                            node_stack.append(curr_obj.right)
                        else:
                            visited.add(curr_obj)
                            node_stack.pop()

                else:
                    if curr_obj.right not in visited:
                        node_stack.append(curr_obj.right)
                    else:
                        if radius >= abs(query_vector[split_axis]-split_val) and curr_obj.left not in visited:
                            node_stack.append(curr_obj.left)
                        else:
                            visited.add(curr_obj)
                            node_stack.pop()
        return output
            
            