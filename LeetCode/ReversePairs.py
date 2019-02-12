class Solution(object):
    def count_reverse(self, nums, left, right):
        if left < right:
            mid = (left+right)/2
            a, x = self.count_reverse(nums, left, mid)
            b, y = self.count_reverse(nums, mid+1, right)
            
            n, m = len(a), len(b)

            i, j, cnt, last_cnt, sum_cnt = 0, 0, 0, 0, 0
            while True:
                if i < n and j < m and a[i] > 2*b[j]:
                    cnt += 1
                    i += 1
                elif (i < n and j < m and a[i] <= 2*b[j]) or (i == n and j < m):
                    last_cnt += cnt
                    sum_cnt += last_cnt
                    cnt = 0
                    j += 1
                else:
                    break

            out, i, j = [], 0, 0
            while True:
                if (i < n and j < m and a[i] > b[j]) or (i < n and j == m):
                    out.append(a[i])
                    i += 1
                elif (i < n and j < m and a[i] <= b[j]) or (i == n and j < m):
                    out.append(b[j])
                    j += 1
                else:
                    break

            return out, sum_cnt + x + y
        
        return [nums[left]], 0
        
        
    def reversePairs(self, nums):
        if len(nums) == 0:
            return 0
        
        out, cnt = self.count_reverse(nums, 0, len(nums)-1)
        return cnt
