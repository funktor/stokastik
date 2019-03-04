import heapq, math

class Solution(object):
    def medianSlidingWindow(self, nums, k):
        if k == 1:
            return [float(x) for x in nums]
        
        nums = zip(nums, range(len(nums)))
        pref = sorted(nums[:k], key=lambda k:k[0])
            
        max_heap, min_heap = pref[:int(k/2)], pref[int(k/2):]
        max_heap = [(-x, y) for x, y in max_heap]
        
        heapq.heapify(max_heap)
        heapq.heapify(min_heap)
        
        out = []
        
        if k % 2 == 0:
            out.append((-max_heap[0][0] + min_heap[0][0])/2.0)
        else:
            out.append(float(min_heap[0][0]))
            
        for i in range(k, len(nums)):
            
            num, num_deleted = nums[i][0], nums[i-k][0]

            if num <= -max_heap[0][0]:
                heapq.heappush(max_heap, (-num, i))
                if num_deleted >= -max_heap[0][0]:
                    while max_heap[0][1] < i-k:
                        heapq.heappop(max_heap)

                    u = max_heap[0]
                    heapq.heappop(max_heap)
                    if u[1] > i-k:
                        heapq.heappush(min_heap, (-u[0], u[1]))

            else:
                heapq.heappush(min_heap, (num, i))
                if num_deleted <= min_heap[0][0]:
                    while min_heap[0][1] < i-k:
                        heapq.heappop(min_heap)

                    u = min_heap[0]
                    heapq.heappop(min_heap)
                    if u[1] > i-k:
                        heapq.heappush(max_heap, (-u[0], u[1]))
            
            while max_heap[0][1] <= i-k:
                heapq.heappop(max_heap)
                
            while min_heap[0][1] <= i-k:
                heapq.heappop(min_heap)

            if k % 2 == 0:
                out.append((-max_heap[0][0] + min_heap[0][0])/2.0)
            else:
                out.append(float(min_heap[0][0]))
        
        return out
