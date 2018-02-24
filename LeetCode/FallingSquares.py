class Solution(object):
    def fallingSquares(self, positions):
        height_cache, out = [], []
        max_height = 0

        for idx in range(len(positions)):
            position = positions[idx]

            if idx == 0:
                height_cache.append(position[1])
                max_height = position[1]

            else:
                start, end = position[0], position[0] + position[1]
                height = position[1]

                for idx2 in range(idx):
                    start2, end2 = positions[idx2][0], positions[idx2][0] + positions[idx2][1]

                    if start2 < end and end2 > start:
                        height = max(height, height_cache[idx2] + position[1])

                height_cache.append(height)
                max_height = max(max_height, height)

            out.append(max_height)

        return out


sol = Solution()
print sol.fallingSquares([[4,2]])