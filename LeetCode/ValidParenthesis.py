class Solution(object):

    def longestValidParentheses(self, s):
        if len(s) == 0:
            return 0

        parenthesis = list(s)

        paired = [0]*len(s)

        for length in range(2, len(parenthesis)+1):
            for i in range(len(parenthesis)-length+1):
                start = i
                end = i+length-1

                if length == 2:
                    if parenthesis[start] == '(' and parenthesis[end] == ')':
                        paired[start] = 1
                        paired[end] = 1
                else:
                    if length % 2 == 0:
                        if parenthesis[start] == '(' and paired[start] == 0:
                            if paired[start+1] == 1 and parenthesis[end] == ')' and paired[end] == 0:
                                paired[start] = 1
                                paired[end] = 1

        max_running_count = -1
        curr_running_count = 0

        for pos in range(len(paired)):
            if paired[pos] == 1:
                curr_running_count += 1
            else:
                if curr_running_count > max_running_count:
                    max_running_count = curr_running_count
                curr_running_count = 0


        if curr_running_count > max_running_count:
            max_running_count = curr_running_count

        return max_running_count


sol = Solution()
print sol.longestValidParentheses("))())(()))()(((()())(()(((()))))((()(())()((((()))())))())))()(()(()))))())(((())(()()))((())()())((()))(()(())(())((())((((()())()))((()(())()))()(()))))))()))(()))))()())()())()()()()()()()))()(((()()((()(())((()())))(()())))))))(()()(())())(()))))))()()())((((()()()())))))((())(())()()(()((()()))()()())(()())()))()(()(()())))))())()(())(()))(())()(())()((())()((((()()))())(((((())))())())(()((())((()()((((((())))(((())))))))(()()((((((()(((())()(()))(()())((()(((()((()(())())()())(((()))()(((()))))(())))(())()())()(((()))))((())())))())()()))((((()))(())()())()(((())(())(()()((())()())()()())())))((()())(()((()()()(()())(()))(()())((((()(()(((()(((())()((()(()))())()())))))))))))()())()(()(((())()))(((()))((((()())())(()())((()())(()()((()((((()())))()(())(())()))))(())())))))(((((((())(((((()))()))(()()()()))))))(()(()(()(()()(((()()))((()))())((())())()())()))()()(((())))()(())()()(())))(((()))))))))(())((()((()((()))))()()()((())((((((((((()(())))(())((()(()())())(((((((()()()()))())(((()())()(()()))))(()()))))(((()()((()()()(((()))))(()()())()()()(()))))()(())))))))()((((((((()((())))))))(()))()((()())())(")

