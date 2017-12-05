class Solution(object):
    def postorderTraversal(self, root):
        if root is None:
            return []

        stack = [root]
        out, visited = [], set()

        while len(stack) > 0:
            top = stack[len(stack) - 1]

            if (top.left is None and top.right is None) or top in visited:
                q = stack.pop()
                out.append(q.val)

            else:
                visited.add(top)

                if top.right is not None:
                    stack.append(top.right)
                if top.left is not None:
                    stack.append(top.left)

        return out