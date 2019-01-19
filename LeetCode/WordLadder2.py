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
        
    def get_adjacency_list(self, word, word_set, max_depth, reverse=False):
        queue, adjacency_list, visited = collections.deque([(word, 0)]), collections.defaultdict(list), dict()
        visited[word] = 0
        
        while len(queue) > 0:
            first, depth = queue.popleft()
            
            if depth < max_depth:
                for pos in range(len(first)):
                    prefix, suffix = first[:pos], first[pos+1:]
                    for alphabet in list(string.ascii_lowercase):
                        if alphabet != first[pos]:
                            new_str = prefix + alphabet + suffix
                            if new_str in word_set:
                                if new_str not in visited:
                                    queue.append((new_str, depth+1))
                                    visited[new_str] = depth+1

                                if visited[new_str] == depth+1:
                                    if reverse:
                                        adjacency_list[new_str].append(first)
                                    else:
                                        adjacency_list[first].append(new_str)
                                    
        return adjacency_list
    
    
    def min_distance(self, beginWord, endWord, word_set):
        queue1, queue2 = collections.deque([(beginWord, 0)]), collections.deque([(endWord, 0)])
        visited1, visited2 = dict(), dict()
        
        visited1[beginWord] = 0
        visited2[endWord] = 0
        
        while len(queue1) > 0 and len(queue2) > 0:
            first1, depth1 = queue1.popleft()
            first2, depth2 = queue2.popleft()
            
            if first1 == first2:
                return depth1+depth2
            
            for pos in range(len(first1)):
                prefix, suffix = first1[:pos], first1[pos+1:]
                for alphabet in list(string.ascii_lowercase):
                    if alphabet != first1[pos]:
                        new_str = prefix + alphabet + suffix
                        if new_str in word_set and new_str not in visited1:
                            if new_str in visited2:
                                return depth1+visited2[new_str]+1
                            
                            queue1.append((new_str, depth1+1))
                            visited1[new_str] = depth1+1
                            
            for pos in range(len(first2)):
                prefix, suffix = first2[:pos], first2[pos+1:]
                for alphabet in list(string.ascii_lowercase):
                    if alphabet != first2[pos]:
                        new_str = prefix + alphabet + suffix
                        if new_str in word_set and new_str not in visited2:
                            if new_str in visited1:
                                return depth2+visited1[new_str]+1
                            
                            queue2.append((new_str, depth2+1))
                            visited2[new_str] = depth2+1
                            
        return -1
        
    def findLadders(self, beginWord, endWord, wordList):
        word_set = set(wordList)
        
        if endWord not in word_set:
            return []
        
        dist = self.min_distance(beginWord, endWord, word_set)
        if dist == -1:
            return []
        
        if dist % 2 == 0:
            a, b = dist/2, dist/2
        else:
            a, b = dist/2+1, dist/2
            
        adjacency_list = self.get_adjacency_list(beginWord, word_set, a)
        adjacency_list.update(self.get_adjacency_list(endWord, word_set, b, reverse=True))
        
        return self.get_list_out(beginWord, endWord, adjacency_list)
