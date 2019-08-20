class Solution(object):
    def max_smaller(self, arr, end, k):
        left, right = 0, end
        while left <= right:
            mid = (left + right)/2
            if arr[mid][1] < k:
                left = mid + 1
            else:
                right = mid - 1
                
        return left-1
    
    def findLongestChain(self, pairs):
        pairs = sorted(pairs, key=lambda k:k[1])
        max_chain_length = [1]*len(pairs)
        
        for i in range(1, len(pairs)):
            max_small_index = self.max_smaller(pairs, i-1, pairs[i][0])
            
            if max_small_index != -1:
                max_chain_length[i] = max(max_chain_length[i], max_chain_length[i-1], max_chain_length[max_small_index] + 1)
            else:
                max_chain_length[i] = max(max_chain_length[i], max_chain_length[i-1])
                    
        return max_chain_length[-1]
