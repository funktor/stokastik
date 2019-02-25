import collections
class Solution(object):
    def get_strings_at_depth(self, A, B, max_depth):
        chars, n = ['a', 'b', 'c', 'd', 'e', 'f'], len(A)
        
        queue = collections.deque([(A, 0, 0, ''.join(chars))])
        visited, cache = set([(A, ''.join(chars))]), {A : 0}
        
        pos_b_dict = collections.defaultdict(list)
        
        for i in range(n):
            pos_b_dict[B[i]].append(i)
        
        while len(queue) > 0:
            front, cnt, depth, all_chars = queue.popleft()
            
            if depth > max_depth:
                return cache
            
            for char in all_chars:
                q = set(all_chars)
                q.remove(char)
                rem_chars = ''.join(sorted(q))
                new_cnt, taken  = 0, set()
                pos_a = [i for i in range(len(front)) if front[i] == char]
                
                new_front = front
                if len(pos_a) > 0:
                    for x in pos_a:
                        flag = False
                        for y in pos_b_dict[char]:
                            if new_front[y] == B[x] and new_front[x] != B[x] and new_front[y] != B[y] and new_front[x] != new_front[y] and y not in taken:
                                a, b = sorted((x, y))
                                new_front = new_front[:a] + new_front[b] + new_front[a+1:b] + new_front[a] + new_front[b+1:]
                                new_cnt += 1
                                flag = True
                                taken.add(y)
                                break

                            if flag:
                                break

                        if flag is False:
                            for y in pos_b_dict[char]:
                                if new_front[y] != B[y] and new_front[x] != B[x] and new_front[x] != new_front[y] and y not in taken:
                                    a, b = sorted((x, y))
                                    new_front = new_front[:a] + new_front[b] + new_front[a+1:b] + new_front[a] + new_front[b+1:]
                                    new_cnt += 1
                                    taken.add(y)
                                    break
                                    
                    if len(new_front) > 0 and (new_front, rem_chars) not in visited:
                        queue.append((new_front, cnt + new_cnt, depth + 1, rem_chars))
                        visited.add((new_front, rem_chars))
                        
                        if new_front not in cache:
                            cache[new_front] = cnt + new_cnt
                        else:
                            cache[new_front] = min(cache[new_front], cnt + new_cnt)
        return cache
                        
                        
                        
    def kSimilarity(self, A, B):
        cache_a = self.get_strings_at_depth(A, B, max_depth=4)
        cache_b = self.get_strings_at_depth(B, A, max_depth=3)
        
        min_cnt = float("Inf")
        for mystr, cnt in cache_a.items():
            if mystr in cache_b:
                min_cnt = min(min_cnt, cnt + cache_b[mystr])
        
        return min_cnt
