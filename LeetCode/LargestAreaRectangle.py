class Solution(object):
    def get_lower_height_index(self, increasing_heights, curr_height):
        left, right = 0, len(increasing_heights) - 1

        while left <= right:
            mid = int((left + right) / 2)

            if increasing_heights[mid][0] <= curr_height:
                left = mid + 1
            else:
                right = mid - 1

        return left - 1


    def largestRectangleArea_w(self, heights):
        increasing_heights = []
        widths = [0] * len(heights)

        for idx in range(len(heights)):
            height = heights[idx]

            if idx == 0:
                increasing_heights.append((height, 0))

            else:
                if height >= heights[idx - 1]:
                    increasing_heights.append((height, idx))

                else:
                    lower_index = self.get_lower_height_index(increasing_heights, height)

                    for pos in range(lower_index + 1, len(increasing_heights)):
                        width = increasing_heights[-1][1] - increasing_heights[pos][1] + 1
                        widths[increasing_heights[pos][1]] += width

                    increasing_heights = increasing_heights[:lower_index + 1]
                    increasing_heights.append((height, idx))

        for pos in range(len(increasing_heights)):
            width = increasing_heights[-1][1] - increasing_heights[pos][1] + 1
            widths[increasing_heights[pos][1]] += width

        return widths

    def largestRectangleArea(self, heights):
        if len(heights) == 0:
            return 0

        a = self.largestRectangleArea_w(heights)
        b = self.largestRectangleArea_w(heights[::-1])[::-1]

        c = [a[x] + b[x] - 1 for x in range(len(a))]
        areas = [heights[idx] * c[idx] for idx in range(len(heights))]

        return max(areas)

sol = Solution()
print sol.largestRectangleArea([0,1,0,2,1,0,1,3,2,1,2,1])

