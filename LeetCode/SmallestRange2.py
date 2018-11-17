import collections

class Solution(object):
    def smallestRangeII(self, A, K):
        if len(A) == 1:
            return 0
        
        w = collections.defaultdict(set)
        
        for i in range(len(A)):
            x = A[i]
            w[x-K].add(i)
            w[x+K].add(i)
        
        v = [k for k, v in w.items()]
        v = sorted(v)
        
        m = len(v)/2
        lowest_possible_min_diff = min([v[i+m-1]-v[i] for i in range(len(v)-m+1)])
        
        deq = collections.deque([v[0]])
        seen_set = set(w[v[0]])
        
        min_diff = float("Inf")
        
        for i in range(1, len(v)):
            if w[deq[0]].intersection(w[v[i]]) == w[deq[0]]:
                deq.popleft()
            
            deq.append(v[i])
            seen_set.update(w[v[i]])
            
            if len(seen_set) == len(A):
                min_diff = min(min_diff, deq[-1]-deq[0])
                
                if min_diff == lowest_possible_min_diff:
                    break

                y = w[deq[0]].intersection(w[deq[-1]])
                seen_set = seen_set.difference(w[deq[0]])
                seen_set.update(y)
                
                deq.popleft()
        
        return min_diff