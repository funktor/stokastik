import random
class Solution(object):
    def __init__(self, N, blacklist):
        self.N = N
        self.intervals, last = [], -1
        for x in sorted(blacklist):
            if x-1 >= last+1:
                self.intervals.append((last+1, x-1))
            last = x
            
        if last+1 <= N-1:
            self.intervals.append((last+1, N-1))
        random.shuffle(self.intervals)
        self.cnt = 0

    def pick(self):
        if len(self.intervals) > 0:
            a, b = self.intervals[self.cnt]
            self.cnt = (self.cnt + 1) % len(self.intervals)
            return random.randint(a, b)
        return random.randint(0, self.N-1)
