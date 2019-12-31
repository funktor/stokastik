class Node():
    def __init__(self, val):
        self.val = val
        self.children = {}
        
class SuffixTree():
    def __init__(self):
        self.root = Node(None)
        
    def insert(self, suffix, start, curr_root):
        if len(suffix) > 0:
            if suffix[0] in curr_root.children:
                x, y = curr_root.children[suffix[0]]
                curr_root.children[suffix[0]] = (x, y+start)
                curr_root, curr_start = x, y
                
                curr_val = curr_root.val

                j = 0
                while j < min(len(curr_val), len(suffix)):
                    if curr_val[j] == suffix[j]:
                        j += 1
                    else:
                        break

                if j < len(curr_val):
                    curr_root.val = curr_val[:j]
                    node = Node(curr_val[j:])
                    node.children = {k:v for k, v in curr_root.children.items()}
                    curr_root.children = {curr_val[j]:(node, curr_start)}
                    
                self.insert(suffix[j:], start, curr_root)

            else: 
                node = Node(suffix)
                curr_root.children[suffix[0]] = (node, start)
        
    
    def build_tree(self, sentence):
        for i in range(len(sentence)):
            suffix = sentence[i:]
            if i == 0:
                node = Node(suffix)
                self.root.children[suffix[0]] = (node, [0])
                
            else:
                self.insert(suffix, [i], self.root)
                
    def print_tree(self, root):
        for child_key, child_node in root.children.items():
            print(child_node[0].val, child_node[1])
            self.print_tree(child_node[0])
            
                
    def search(self, pattern):
        root = self.root
        start = float("Inf")
        
        while True:
            if len(pattern) > 0 and pattern[0] in root.children:
                root, start = root.children[pattern[0]]
                curr_val = root.val
                
                i = 0
                while i < min(len(curr_val), len(pattern)):
                    if curr_val[i] == pattern[i]:
                        i += 1
                    else:
                        break
                        
                if i < min(len(curr_val), len(pattern)):
                    return None
                
                if i == len(pattern):
                    return start
                
                pattern = pattern[i:]
                
            else:
                break
                
        return None
                
                    
        