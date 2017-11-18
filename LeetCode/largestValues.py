class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def largestValues(self, root):
        queue, largest = [(root, 0)], []

        last_depth = 0
        running = []

        while len(queue) > 0:
            node, depth = queue.pop()

            if depth > last_depth:
                max_val = -float("Inf")

                for idx in range(len(running)):
                    if running[idx] > max_val:
                        max_val = running[idx]

                largest.append(max_val)
                running = [node.val]
            else:
                running.append(node.val)

            last_depth = depth

            if node.left is not None:
                queue.insert(0, (node.left, depth + 1))

            if node.right is not None:
                queue.insert(0, (node.right, depth + 1))

        max_val = -float("Inf")

        for idx in range(len(running)):
            if running[idx] > max_val:
                max_val = running[idx]

        largest.append(max_val)

        return largest