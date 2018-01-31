import collections

class Node(object):
    def __init__(self, val, children):
        self.val = val
        self.children = children


class Solution(object):
    def height(self, root, cached):
        max_height = 1

        if len(root.children) > 0:
            for child in root.children:
                if child.val not in cached:
                    max_height = max(max_height, self.height(child, cached))
                else:
                    max_height = max(max_height, cached[child.val])

            max_height += 1

        cached[root.val] = max(cached[root.val], max_height)

        return max_height


    def build_tree(self, root, matrix, row1, col1, row2, col2, node_map, root_nodes):
        p = 0 <= row2 < len(matrix)
        q = 0 <= col2 < len(matrix[0])

        if p and q and matrix[row2][col2] > matrix[row1][col1]:
            if (row2, col2) in node_map:
                new_node = node_map[(row2, col2)]

                if (row2, col2) in root_nodes:
                    root_nodes.remove((row2, col2))
            else:
                new_node = Node((row2, col2), [])
                node_map[(row2, col2)] = new_node

            root.children.append(new_node)


    def longestIncreasingPath(self, matrix):
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return 0

        node_map = collections.defaultdict(Node)
        root_nodes = set()

        for row in range(len(matrix)):
            for col in range(len(matrix[0])):

                if (row, col) in node_map:
                    node = node_map[(row, col)]
                else:
                    node = Node((row, col), [])
                    node_map[(row, col)] = node
                    root_nodes.add((row, col))

                self.build_tree(node, matrix, row, col, row - 1, col, node_map, root_nodes)
                self.build_tree(node, matrix, row, col, row, col - 1, node_map, root_nodes)
                self.build_tree(node, matrix, row, col, row + 1, col, node_map, root_nodes)
                self.build_tree(node, matrix, row, col, row, col + 1, node_map, root_nodes)

        max_len = 1

        for key in root_nodes:
            node = node_map[key]
            cached = collections.defaultdict(int)
            max_len = max(max_len, self.height(node, cached))

        return max_len

sol = Solution()
print sol.longestIncreasingPath([
  [9,9,4],
  [6,6,8],
  [2,1,1]
])