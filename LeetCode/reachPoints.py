class Solution(object):
    def can_reach(self, sx, sy, tx, ty):
        if (sx, sy) == (tx, ty):
            return True

        elif sx > tx or sy > ty:
            return False

        else:
            if tx >= ty:
                factor = max(int((tx - sx) / ty), 1)
                print factor
                return self.can_reach(sx, sy, tx - factor * ty, ty)
            else:
                factor = max(int((ty - sy) / tx), 1)
                print factor
                return self.can_reach(sx, sy, tx, ty - factor * tx)


    def reachingPoints(self, sx, sy, tx, ty):
        return self.can_reach(sx, sy, tx, ty)

sol = Solution()
print sol.reachingPoints(9,10,9,19)