import collections

class Solution(object):

    def get_max(self, nums):
        cache = collections.defaultdict(dict)

        for length in range(1, len(nums) + 1):
            for start in range(len(nums) - length + 1):
                end = start + length - 1

                if length == 1:
                    cache[start][end] = (nums[start], start)
                else:
                    if nums[start] >= cache[start + 1][end][0]:
                        cache[start][end] = (nums[start], start)
                    else:
                        cache[start][end] = cache[start + 1][end]
        return cache


    def get_max_num(self, nums, max_cache, k):
        if len(nums) == k:
            return nums
        else:
            start, end = 0, len(nums) - k
            m = k

            out = []
            while len(out) < m:
                max_num = max_cache[start][end]
                out.append(max_num[0])
                k -= 1
                start = max_num[1] + 1
                end = len(nums) - k

            return out


    def compare(self, w1, w2):
        if len(w1) == 0 and len(w2) == 0:
            return []
        elif len(w1) == 0:
            return w2
        elif len(w2) == 0:
            return w1

        i = 0
        while w1[i] == w2[i]:
            i += 1
            if i >= len(w1):
                return w1

        if w1[i] > w2[i]:
            return w1
        else:
            return w2


    def merge(self, nums1, nums2):
        out = []

        i, j = 0, 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                i1, j1 = i, j
                while i1 < len(nums1) and j1 < len(nums2) and nums1[i1] == nums2[j1]:
                    i1 += 1
                    j1 += 1

                if i1 < len(nums1) and j1 < len(nums2):
                    if nums1[i1] > nums2[j1]:
                        out.append(nums1[i])
                        i += 1
                    else:
                        out.append(nums2[j])
                        j += 1
                else:
                    if i1 < len(nums1):
                        out.append(nums1[i])
                        i += 1
                    else:
                        out.append(nums2[j])
                        j += 1
            elif nums1[i] > nums2[j]:
                out.append(nums1[i])
                i += 1
            elif nums1[i] < nums2[j]:
                out.append(nums2[j])
                j += 1

        if i < len(nums1):
            out += nums1[i:]

        if j < len(nums2):
            out += nums2[j:]

        return out


    def maxNumber(self, nums1, nums2, k):
        best = []

        w = max(len(nums1), len(nums2))

        start = max(0, k - w)
        end = min(k, w)

        p, q = self.get_max(nums1), self.get_max(nums2)

        for idx in range(start, end + 1):
            x, y = idx, k - idx

            if x == 0 and y <= len(nums2):
                b = self.get_max_num(nums2, q, y)
                best = self.compare(best, b)
            elif y == 0 and x <= len(nums1):
                a = self.get_max_num(nums1, p, x)
                best = self.compare(best, a)
            elif 0 < x <= len(nums1) and 0 < y <= len(nums2):
                a = self.get_max_num(nums1, p, x)
                b = self.get_max_num(nums2, q, y)
                print a, b

                c = self.merge(a, b)
                best = self.compare(best, c)

        return best


sol = Solution()
print(sol.maxNumber([5,5,1], [4,0,1], 3))