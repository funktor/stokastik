class Solution(object):
    def splitIntoFibonacci(self, S):
        heap = [(0, -1, [])]
        visited = set([-1])
        
        output = []
        while len(heap) > 0:
            cost, start, vals = heapq.heappop(heap)
            cost = -cost
            
            if start == len(S)-1 and len(vals) >= 3:
                return vals
            
            if start < len(S)-1:
                end = len(S) if S[start+1] != '0' else start+2
                
                if len(vals) < 2:
                    curr_str = ""
                    for i in range(start+1, end):
                        curr_str += S[i]
                        int_c = int(curr_str)
                        key = (start, i)

                        if key not in visited:
                            heapq.heappush(heap, (-(cost+1), i, vals + [int_c]))
                            visited.add(key)
                else:
                    int_c = vals[-1] + vals[-2]
                    l = len(str(int_c))
                    
                    key = (start, start+l)
                    
                    if int_c <= (1<<31)-1 and start+l+1 <= end and int(S[start+1:start+l+1]) == int_c and key not in visited:
                        heapq.heappush(heap, (-(cost+1), start+l, vals + [int_c]))
                        visited.add(key)
        
        return output
                    
        
