import numpy as np

class Solution(object):
    def bestRotation(self, arr):
        counts_arr = np.zeros(len(arr))

        for idx in range(len(arr)):
            num = arr[idx]

            if num > idx:
                min_rotation = idx + 1
                max_rotation = idx + len(arr) - num

                counts_arr[min_rotation : max_rotation + 1] += 1

            else:
                min_rotation = 0
                max_rotation = idx - num

                counts_arr[min_rotation: max_rotation + 1] += 1

                if idx + 1 < len(arr):
                    min_rotation = idx + 1
                    max_rotation = len(arr) - 1

                    counts_arr[min_rotation: max_rotation + 1] += 1

        return np.argmax(counts_arr)

sol = Solution()
print sol.bestRotation([1, 3, 0, 2, 4])