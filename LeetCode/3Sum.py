class Solution(object):
    def get_left_index(self, arr, num, left, right):

        while left < right:
            mid = (left + right) / 2

            if arr[mid] == num and mid + 1 <= right and arr[mid + 1] > num:
                return mid

            elif arr[mid] <= num:
                left = mid + 1
            else:
                right = mid - 1

        return left

    def search(self, arr, num, left, right):
        out = []

        while left < right:
            if arr[left] + arr[right] == num:
                out.append((arr[left], arr[right]))
                left += 1
                right -= 1
            elif arr[left] + arr[right] < num:
                left += 1
            else:
                right -= 1

        return out

    def threeSum(self, nums):
        if len(nums) == 0:
            return []

        sorted_nums = sorted(nums)

        max_num = sorted_nums[len(sorted_nums) - 1]

        res, visited = [], set()

        for idx in range(len(nums) - 1):
            q = sorted_nums[idx]

            if q not in visited:
                visited.add(q)

                left_idx = self.get_left_index(sorted_nums, -q - max_num, idx + 1, len(nums) - 2)

                out = self.search(sorted_nums, -q, left_idx, len(nums) - 1)

                if len(out) > 0:
                    g = []
                    for k in out:
                        g.append((q, k[0], k[1]))

                    res += g

        return list(set(res))

sol = Solution()
print sol.threeSum([-1,0,1,2,-1,-4])