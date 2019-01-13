# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def get_start_overlap(self, intervals, newInterval):
        left, right = 0, len(intervals)-1
        while left <= right:
            mid = (left + right)/2
            mid_interval = intervals[mid]
            
            if newInterval.start == mid_interval.end:
                return mid
            elif newInterval.start > mid_interval.end:
                left = mid + 1
            else:
                right = mid - 1
                
        return left
    
    def get_end_overlap(self, intervals, newInterval):
        left, right = 0, len(intervals)-1
        while left <= right:
            mid = (left + right)/2
            mid_interval = intervals[mid]
            
            if newInterval.end == mid_interval.start:
                return mid
            elif newInterval.end > mid_interval.start:
                left = mid + 1
            else:
                right = mid - 1
                
        return right
    
    def insert(self, intervals, newInterval):
        if len(intervals) == 0:
            return [newInterval]
        
        start_overlap = self.get_start_overlap(intervals, newInterval)
        end_overlap = self.get_end_overlap(intervals, newInterval)
        
        print start_overlap, end_overlap
        
        prefix, suffix = intervals[:start_overlap], intervals[end_overlap+1:]
        
        if start_overlap <= end_overlap:
            start = min(newInterval.start, intervals[start_overlap].start)
            end = max(newInterval.end, intervals[end_overlap].end)
        else:
            start, end = newInterval.start, newInterval.end

        interval = Interval()
        interval.start, interval.end = start, end

        return prefix + [interval] + suffix
