import heapq

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

class Solution(object):
    def employeeFreeTime(self, schedule):
        intervals = []

        for emp_sch in schedule:
            for interval in emp_sch:
                intervals.append((interval.start, interval.end))

        heapq.heapify(intervals)

        start, end = -1, -1
        free_times = []

        while len(intervals) > 0:
            root = heapq.heappop(intervals)

            if start <= root[0] <= end:
                end = max(end, root[1])
            else:
                if start >= 0:
                    interval = Interval(end, root[0])
                    free_times.append(interval)

                start, end = root

        return free_times


sol = Solution()
arr = [[[1,2],[5,6]],[[1,3]],[[4,10]]]

print sol.employeeFreeTime()
