import collections

class Solution(object):
    def isRectangleCover(self, rectangles):
        min_row, min_col = float("Inf"), float("Inf")
        max_row, max_col = -float("Inf"), -float("Inf")

        area_sum = 0

        for rectangle in rectangles:
            a, b, c, d = rectangle

            area_sum += (c - a) * (d - b)

            min_row = min(min_row, a)
            min_col = min(min_col, b)
            max_row = max(max_row, c)
            max_col = max(max_col, d)

        big_rectangle_area = (max_row - min_row) * (max_col - min_col)

        if big_rectangle_area != area_sum:
            return False

        counts = collections.defaultdict(int)

        for rectangle in rectangles:
            a, b, c, d = rectangle

            counts[(a, b)] += 1
            counts[(c, d)] += 1
            counts[(a, d)] += 1
            counts[(c, b)] += 1

        for point, count in counts.items():
            x, y = point
            if (x, y) not in [(min_row, min_col), (max_row, max_col), (min_row, max_col), (max_row, min_col)] and count % 2 == 1:
                return False

            if (x, y) in [(min_row, min_col), (max_row, max_col), (min_row, max_col), (max_row, min_col)] and count % 2 == 0:
                return False

        return True