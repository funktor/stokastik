import heapq, collections

class HeapEl(object):
    def __init__(self, pos1, pos2, val1, val2):
        self.pos1 = pos1
        self.pos2 = pos2
        self.val1 = val1
        self.val2 = val2

class Solution(object):

    def insert_into_heap(self, myheap, el, max_size):
        if len(myheap) < max_size:
            heapq.heappush(myheap, (-abs(el.val1 - el.val2), el))
        else:
            if -myheap[0][0] > abs(el.val1 - el.val2):
                heapq.heappop(myheap)
                heapq.heappush(myheap, (-abs(el.val1 - el.val2), el))

        return myheap


    def smallestDistancePair(self, nums, k):
        sorted_nums = sorted(nums)

        myheap = []

        for idx in range(len(sorted_nums)-1):
            el = HeapEl(idx, idx + 1, sorted_nums[idx], sorted_nums[idx + 1])
            myheap = self.insert_into_heap(myheap, el, k)

        myheap = [(-x[0], x[1]) for x in myheap]
        heapq.heapify(myheap)

        curr_cnt, out = 0, None

        while curr_cnt < k:
            min_el = heapq.heappop(myheap)
            out = min_el[0]

            data = min_el[1]

            if data.pos2 + 1 < len(nums):
                new_pos2 = data.pos2 + 1
                new_val2 = sorted_nums[new_pos2]

                el = HeapEl(data.pos1, new_pos2, data.val1, new_val2)
                heapq.heappush(myheap, (abs(el.val1 - el.val2), el))

            curr_cnt += 1

        return out

sol = Solution()
print sol.num_pairs_less([1, 5, 13, 15, 20, 21, 24], 10)