class Solution(object):
    def jump(self, nums):
        if len(nums) == 0:
            return 0

        queue = [(0, 0)]

        while len(queue) > 0:
            q = queue.pop()
            index = q[0]

            jump_len = nums[index]

            if index == len(nums) - 1:
                return q[1]
            elif index + jump_len >= len(nums) - 1:
                return 1 + q[1]
            else:
                if len(queue) > 0:
                    start = queue[0][0] + 1
                else:
                    start = index + 1

                for idx in range(start, index + jump_len + 1):
                    queue.insert(0, (idx, q[1] + 1))

        return -1