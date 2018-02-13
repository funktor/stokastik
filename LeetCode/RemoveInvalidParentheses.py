class Solution(object):
    def is_valid(self, mystr):
        num_0, num_1 = 0, 0

        for idx in range(len(mystr)):
            if mystr[idx] == '(':
                num_0 += 1
            elif mystr[idx] == ')':
                num_1 += 1

            if num_1 > num_0:
                return False

        return num_0 == num_1


    def valid_parenthesis(self, mystr):
        out, queue = set(), [mystr]
        visited = set([mystr])

        while len(queue) > 0:
            w = queue.pop()

            for idx in range(len(w)):
                if w[idx] == '(' or w[idx] == ')':
                    new_str = w[:idx] + w[idx + 1:]

                    if self.is_valid(new_str):
                        out.add(new_str)

                    else:
                        if new_str not in visited:
                            visited.add(new_str)
                            queue.insert(0, new_str)

            if (len(queue) == 0 or (len(queue) > 0 and len(queue[-1]) < len(w))) and len(out) > 0:
                return list(out)

        return [""]


    def removeInvalidParentheses(self, s):
        if self.is_valid(s):
            return [s]

        return self.valid_parenthesis(s)

sol = Solution()
print sol.removeInvalidParentheses("x(")

