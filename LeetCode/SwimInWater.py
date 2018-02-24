class Solution(object):
    def get_flag(self, grid, curr_height, start, end, visited):
        a = 0 <= start[0] < len(grid) and 0 <= start[1] < len(grid)
        b = 0 <= end[0] < len(grid) and 0 <= end[1] < len(grid)

        if a and b and grid[end[0]][end[1]] <= curr_height and end not in visited:
            return self.traverse(grid, curr_height, end, visited)

        return False


    def traverse(self, grid, curr_height, start, visited):
        if start == (len(grid) - 1, len(grid) - 1):
            return True

        else:
            visited.add(start)

            flag = False

            flag = flag or self.get_flag(grid, curr_height, start, (start[0] - 1, start[1]), visited)
            flag = flag or self.get_flag(grid, curr_height, start, (start[0], start[1] - 1), visited)
            flag = flag or self.get_flag(grid, curr_height, start, (start[0] + 1, start[1]), visited)
            flag = flag or self.get_flag(grid, curr_height, start, (start[0], start[1] + 1), visited)

            return flag


    def can_swim(self, grid, curr_height):
        if grid[0][0] > curr_height or grid[len(grid) - 1][len(grid) - 1] > curr_height:
            return False

        visited = set()
        return self.traverse(grid, curr_height, (0,0), visited)


    def swimInWater(self, grid):
        n = len(grid)
        left, right = 0, n*n - 1

        while left <= right:
            curr_height = int((left + right) / 2)

            print curr_height

            if self.can_swim(grid, curr_height):
                right = curr_height - 1
            else:
                left = curr_height + 1

        return left


sol = Solution()
print sol.swimInWater([[0,1,2,3,4],[24,23,22,21,5],[12,13,14,15,16],[11,17,18,19,20],[10,9,8,7,6]])