class Solution(object):
    def trap(self, height):
        if len(height) < 3:
            return 0

        prefix_sums, suffix_max = [], [-1] * len(height)

        for idx in range(len(height)):
            if idx == 0:
                prefix_sums.append(height[0])
            else:
                prefix_sums.append(height[idx] + prefix_sums[len(prefix_sums) - 1])

        for idx in reversed(range(len(height) - 1)):
            if idx == len(height) - 2:
                suffix_max[idx] = idx + 1
            else:
                if height[suffix_max[idx + 1]] > height[idx + 1]:
                    suffix_max[idx] = suffix_max[idx + 1]
                else:
                    suffix_max[idx] = idx + 1

        start, total = 0, 0

        while start < len(height) - 1:
            if height[suffix_max[start]] <= height[start]:
                end = suffix_max[start]

                w = min(height[start], height[end])

                x = prefix_sums[end - 1] - prefix_sums[start]
                total += w * (end - start - 1) - x
                start = end
            else:
                flag = False

                for end in range(start + 1, suffix_max[start]):
                    if height[end] > height[start]:
                        w = min(height[start], height[end])

                        x = prefix_sums[end - 1] - prefix_sums[start]
                        total += w * (end - start - 1) - x
                        start = end
                        flag = True
                        break

                if flag is False:
                    end = suffix_max[start]

                    w = min(height[start], height[end])

                    x = prefix_sums[end - 1] - prefix_sums[start]
                    total += w * (end - start - 1) - x
                    start = end

        return total