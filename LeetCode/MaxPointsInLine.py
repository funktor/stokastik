import collections

class Point(object):
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

class Solution(object):

    def maxPoints(self, points):
        nested_dict = lambda: collections.defaultdict(nested_dict)

        if len(points) == 0:
            return 0
        slopes_dict = nested_dict()
        points_cnt = dict()

        for i in range(len(points)):
            point = points[i]
            point_key = str(point.x) + '__' + str(point.y)

            if point_key not in points_cnt:
                points_cnt[point_key] = 0
            points_cnt[point_key] += 1

        for i in range(len(points)):
            point1 = points[i]
            point1_key = str(point1.x) + '__' + str(point1.y)

            for j in range(i+1, len(points)):
                if i != j:
                    point2 = points[j]
                    point2_key = str(point2.x) + '__' + str(point2.y)

                    if point2.x == point1.x:
                        slope = 'Inf'
                    else:
                        slope = float(point2.y - point1.y)*10 / (point2.x - point1.x)

                    if point1_key != point2_key:
                        if slope not in slopes_dict[point1_key]:
                            slopes_dict[point1_key][slope] = set()
                        slopes_dict[point1_key][slope].add(point2_key)

                        if slope not in slopes_dict[point2_key]:
                            slopes_dict[point2_key][slope] = set()
                        slopes_dict[point2_key][slope].add(point1_key)

        max_count = 0

        for point1_key, duplicates in points_cnt.iteritems():
            if point1_key in slopes_dict:
                slope_points_dict = slopes_dict[point1_key]

                for slope, point2_keys in slope_points_dict.iteritems():
                    num_lines_count = duplicates

                    for point2_key in point2_keys:
                        num_lines_count += points_cnt[point2_key]

                    if num_lines_count >= max_count:
                        max_count = num_lines_count

            else:
                if duplicates >= max_count:
                    max_count = duplicates

        return max_count

points = [[0,0],[94911151,94911150],[94911152,94911151]]
points_inp = []

for point in points:
    pt = Point(point[0], point[1])
    points_inp.append(pt)

sol = Solution()
print(sol.maxPoints(points_inp))
