import collections, heapq

class Solution(object):
    def merge_sorted_arrays(self, nums):
        min_heap = []
        for idx in range(len(nums)):
            min_heap += zip(nums[idx], [idx] * len(nums[idx]))

        heapq.heapify(min_heap)

        out = []
        while len(min_heap) > 0:
            out.append(heapq.heappop(min_heap))

        return out

    def smallestRange(self, nums):
        min_num, max_num = 100001, -100001
        flag = True

        for num in nums:
            if len(num) == 0:
                flag = False
            else:
                min_num = min(min_num, num[0])
                max_num = max(max_num, num[len(num) - 1])

        if flag is False:
            return [min_num, max_num]

        merged = self.merge_sorted_arrays(nums)

        b = [-100001, 100001]

        arr_set = collections.defaultdict(int)

        arr_set[merged[0][1]] = 1
        start, end = 0, 0

        while len(arr_set) > 0 and start <= end:
            if len(arr_set) == len(nums):
                curr_min = merged[start][0]
                curr_max = merged[end][0]

                a = [curr_min, curr_max]

                if a[1] - a[0] < b[1] - b[0] or (a[1] - a[0] == b[1] - b[0] and a[0] < b[0]):
                    b = a

                arr_set[merged[start][1]] -= 1

                if arr_set[merged[start][1]] == 0:
                    arr_set.pop(merged[start][1])

                start += 1
            else:
                if end < len(merged) - 1:
                    end += 1
                    arr_set[merged[end][1]] += 1
                else:
                    arr_set[merged[start][1]] -= 1

                    if arr_set[merged[start][1]] == 0:
                        arr_set.pop(merged[start][1])

                    start += 1

        return b