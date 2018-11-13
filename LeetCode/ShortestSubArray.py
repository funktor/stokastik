import collections

class Solution(object):
    def get_max_negative(self, arr):
        max_negatives = zip(list(range(len(arr))), arr)

        for i in reversed(range(len(arr))):
            if arr[i] >= 0:
                x, y = arr[i], i

                while y + 1 < len(arr) and max_negatives[y+1][1] + x >= 0:
                    x += max_negatives[y+1][1]
                    y = max_negatives[y+1][0]

                if y + 1 >= len(arr):
                    max_negatives[i] = (len(arr), x)
                else:
                    max_negatives[i] = (max_negatives[y+1][0], max_negatives[y+1][1] + x)

        return max_negatives
    
    def shortestSubarray(self, A, K):
        min_length = len(A) + 1
        max_negatives = self.get_max_negative(A)
        
        queue = collections.deque([0])
        index, queue_sum = 0, A[0]
        
        while True:
            if len(queue) > 0:
                index = queue[-1]
                
                if queue_sum >= K:
                    queue_sum -= A[queue[0]]
                    min_length = min(min_length, len(queue))
                    queue.popleft()
                    
                else:
                    if max_negatives[queue[0]][0] <= queue[-1]:
                        queue_sum -= A[queue[0]]
                        queue.popleft()
                    else:
                        if index == len(A) - 1:
                            break
                            
                        queue.append(index + 1)
                        queue_sum += A[index + 1]
                        
            elif index < len(A) - 1:
                queue.append(index + 1)
                queue_sum += A[index + 1]
                
            else:
                break
        
        if min_length == len(A) + 1:
            return -1
        
        return min_length