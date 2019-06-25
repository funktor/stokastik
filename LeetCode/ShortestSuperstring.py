import collections, heapq

class Solution(object):
    def merge(self, A, s1, s2, m, n):
        if s2 in s1:
            return s1
        
        else:
            pos, curr_pref, curr_suff = -1, "", ""
            for i in range(len(A[n])):
                if len(A[m])-i-1 >= 0:
                    curr_pref += A[n][i]
                    curr_suff = A[m][len(A[m])-i-1] + curr_suff
                    
                    if curr_pref == curr_suff:
                        pos = i
            
            return s1 + s2[pos+1:]
        
    def shortestSuperstring(self, A):
        if len(A) == 1:
            return A[0]
        
        heap, min_len, best_str = [], float("Inf"), ""
        cache, g = {}, set(range(len(A)))
        
        for i in range(len(A)):
            heap.append((0, set([i])))
            cache[tuple(set([i]))] = (A[i], i, i)
        
        heapq.heapify(heap)
        
        while len(heap) > 0:
            u, h = heapq.heappop(heap)
            v = cache[tuple(h)]
            
            t = g.difference(h)
            
            if tuple(t) in cache:
                y = cache[tuple(t)]
                
                p1 = self.merge(A, v[0], y[0], v[2], y[1])
                p2 = self.merge(A, y[0], v[0], y[2], v[1])
                    
                p = p1 if len(p1) < len(p2) else p2
                
                if len(p) < min_len:
                    min_len = len(p)
                    best_str = p
            
            else:
                for i in range(len(A)):
                    if i not in h:
                        h1 = h.copy()
                        h1.add(i)

                        p1 = self.merge(A, v[0], A[i], v[2], i)
                        p2 = self.merge(A, A[i], v[0], i, v[1])

                        if len(p1) < len(p2):
                            p = p1
                            if len(p) == len(v[0]):
                                r1, r2 = v[1], v[2]
                            else:
                                r1, r2 = v[1], i

                        else:
                            p = p2
                            if len(p) == len(A[i]):
                                r1, r2 = i, i
                            else:
                                r1, r2 = i, v[2]

                        q = -(len(v[0]) + len(A[i]) - len(p)) + u

                        if len(h1) == len(A):
                            if len(p) < min_len:
                                min_len = len(p)
                                best_str = p

                        else:
                            if tuple(h1) not in cache or len(p) < len(cache[tuple(h1)][0]):
                                if best_str == "" or len(p) < len(best_str):
                                    heapq.heappush(heap, (q, h1))
                                    cache[tuple(h1)] = (p, r1, r2)
        return best_str
