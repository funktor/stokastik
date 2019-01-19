import collections, string

class Solution(object):
    def get_list_out(self, beginWord, endWord, adjacency_list):
        if beginWord == endWord:
            return [[beginWord]]
        elif beginWord not in adjacency_list:
            return []
        else:
            new_lst = adjacency_list[beginWord]
            out = []
            for x in new_lst:
                w = self.get_list_out(x, endWord, adjacency_list)
                for y in w:
                    out.append([beginWord] + y)
            return out
        
    def findLadders(self, beginWord, endWord, wordList):
        word_set = set(wordList)
        
        if endWord not in word_set:
            return []
        
        queue, adjacency_list, visited = collections.deque([(beginWord, 0)]), collections.defaultdict(list), dict()
        visited[beginWord] = 0
        
        found = False
        
        while len(queue) > 0:
            first, depth = queue.popleft()
            curr_word = first
            
            first = list(first)
                    
            if curr_word != endWord:
                for pos in range(len(first)):
                    prefix, suffix = first[:pos], first[pos+1:]
                    for alphabet in list(string.ascii_lowercase):
                        if alphabet != first[pos]:
                            new_str = ''.join(prefix + [alphabet] + suffix)
                            if new_str in word_set:
                                if new_str not in visited:
                                    queue.append((new_str, depth+1))
                                    visited[new_str] = depth+1
                                    
                                if visited[new_str] == depth+1:
                                    adjacency_list[curr_word].append(new_str)
            else:
                found = True
        
        if found is False:
            return []
        
        return self.get_list_out(beginWord, endWord, adjacency_list)
