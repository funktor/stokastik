import collections

class Solution(object):

    def __init__(self):
        self.val = ''
        self.idx = -1
        self.children = []

    def get_tree(self, root, str_len):
        out = []

        if len(root.children) > 0:
            for child in root.children:
                w = self.get_tree(child, str_len)
                for q in w:
                    if root.val != '':
                        out.append(root.val + " " + q)
                    else:
                        out.append(q)
        else:
            if root.idx == str_len:
                out.append(root.val)

        return out

    def check_non_break(self, s, full_left):
        stack = [0]

        while len(stack) > 0:
            q = stack.pop()

            if q in full_left:
                end_positions = full_left[q]

                flag = False
                for end in end_positions:
                    if end in full_left or end == len(s):
                        flag = flag or True
                        stack.append(end)

                if flag is False:
                    return False
        return True


    def get_breaks(self, s, wordDict):

        full_left = collections.defaultdict(dict)

        for start in range(len(s)):
            for end in range(start + 1, len(s)+1):
                if s[start:end] in wordDict:
                    if start not in full_left:
                        full_left[start] = []
                    full_left[start].append(end)

        if self.check_non_break(s, full_left) is False:
            return []

        root = Solution()
        root.val = ''
        root.idx = 0

        stack = [root]

        while len(stack) > 0:
            root_node = stack.pop()
            q = root_node.idx

            if q in full_left:
                end_positions = full_left[q]

                for end in end_positions:
                    node = Solution()
                    node.val = s[q:end]
                    node.idx = end

                    root_node.children.append(node)

                    stack.append(node)

        return self.get_tree(root, len(s))

    def wordBreak(self, s, wordDict):
        if len(wordDict) == 0:
            return []

        out = self.get_breaks(s, set(wordDict))

        return out

sol = Solution()
arr = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#print len(arr)
print(sol.wordBreak(arr, ["a","aa","aaa","aaaa","aaaaa","aaaaaa","aaaaaaa","aaaaaaaa","aaaaaaaaa","aaaaaaaaaa"]))
#print(sol.wordBreak("aggegbnngohbggalojckbdfjakgnnjadhganfdkefeddjdnabmflabckflfljafdlmmbhijojiaaifedaihnoinedhhnolcjdam", ["o","b","gbdfgiokkfnhl","glibjohcmd","bblcnhelanckn","mflabckflflja","mgda","oheafhajjo","cc","cffalholojikojm","haljiamccabh","gjkdlonmhdacd","ee","bc","mjj","fdlmmbhij","nn","jiaaifedaihn","nhligg","hooaglldlei","hajhebh","ebijeeh","me","eibm","ekkobhajgkem","ohaofonhjakc","n","kjjogm","mhn","odcamjmodie","edmagbkejiocacl","kcbfnjialef","lhifcohoe","akgnn","fbgakjhjb","belggjekmn","oinedhhnolc","ddekcnag","oneoakldakalb","bodnokemafkhkhf","dkefeddjdnab","gflcngff","fgnfmbcogmojgm","ad","jadhganf","lojckbdfj","gadkaoe","jdam","ljjndlnednnombl","aggegbnngohbgga"]))
#print(sol.wordBreak("applemangoorange", ["app", "apple", "leman", "man", "mango", "go", "orange"]))
#print(len(sol.wordBreak("aaaaaaa", ["aaaa", "aa", "a"])))
#print(sol.wordBreak("aaaaaaa", ["aaaa", "aaa"]))