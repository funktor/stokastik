class Solution(object):
    def reachingPoints(self, sx, sy, tx, ty):
        if (sx, sy) == (tx, ty):
            return True

        elif sx > tx or sy > ty:
            return False

        else:
            if tx >= ty:
                factor = max(int((tx - sx) / ty), 1)
                return self.reachingPoints(sx, sy, tx - factor * ty, ty)
            else:
                factor = max(int((ty - sy) / tx), 1)
                return self.reachingPoints(sx, sy, tx, ty - factor * tx)

sol = Solution()
print sol.reachingPoints(9,10,9,19)