import collections, heapq

class Solution(object):
    def scheduleCourse(self, courses):
        courses = sorted(courses, key=lambda k:k[1])
        max_heap = [(-courses[0][0], courses[0][1])]
        end_day = courses[0][0]
        max_length = 1
        
        for i in range(1, len(courses)):
            heapq.heappush(max_heap, (-courses[i][0], courses[i][1]))
            end_day += courses[i][0]
            
            if end_day > courses[i][1]:
                while len(max_heap) > 0 and end_day > courses[i][1]:
                    a, b = heapq.heappop(max_heap)
                    end_day += a
                    
            max_length = max(max_length, len(max_heap))
        
        return max_length
