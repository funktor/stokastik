class WordFilter(object):

    def __init__(self, words):
        self.prefix = dict()
        self.suffix = dict()
        
        word_set = set()
        
        for i, word in reversed(list(enumerate(words))):
            if word not in word_set:
                p, s = '', ''

                if p not in self.prefix:
                    self.prefix[p] = []
                self.prefix[p].append(i)

                if s not in self.suffix:
                    self.suffix[s] = []
                self.suffix[s].append(i)

                for char in word:
                    p += char
                    if p not in self.prefix:
                        self.prefix[p] = []
                    self.prefix[p].append(i)

                for char in word[::-1]:
                    s = char + s
                    if s not in self.suffix:
                        self.suffix[s] = []
                    self.suffix[s].append(i)
                    
                word_set.add(word)
        

    def f(self, prefix, suffix):
        p_indices = set(self.prefix[prefix]) if prefix in self.prefix else set()
        s_indices = self.suffix[suffix] if suffix in self.suffix else []
        
        if len(p_indices) > 0 and len(s_indices) > 0:
            for i in s_indices:
                if i in p_indices:
                    return i
        return -1
        


# Your WordFilter object will be instantiated and called as such:
# obj = WordFilter(words)
# param_1 = obj.f(prefix,suffix)
