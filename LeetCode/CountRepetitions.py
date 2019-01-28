class Solution(object):
    def get_period(self, s1, s2):
        end_positions, end_pos_map = [], dict()
        i, j = 0, 0
        n, m = len(s1), len(s2)
        period = 0
        
        while True:
            if s1[i%n] == s2[j%m]:
                i += 1
                j += 1
                if j % m == 0:
                    if (i-1) % n in end_pos_map:
                        period = i-1-end_pos_map[(i-1)%n]
                        break
                    end_positions.append(i-1)
                    end_pos_map[(i-1)%n] = i-1
            else:
                i += 1
        
        return end_positions, period

    def getMaxRepetitions(self, s1, n1, s2, n2):
        if n1 == 0 or len(set(s2).difference(set(s1))) > 0:
            return 0
        
        positions, period = self.get_period(s1, s2)
        
        max_pos, sums = n1*len(s1)-1, len(positions)
        
        for x in positions:
            sums += (max_pos-x)/period
            
        return sums/n2
