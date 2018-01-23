class Solution(object):

    def is_compatible(self, a, b):
        if a % 2 == b % 2:
            return False
        else:
            if a % 2 == 0:
                return b == a + 1
            else:
                return a == b + 1


    def min_swaps(self, row, positions, start):
        if len(row) - start == 4:
            a, b = row[start], row[start + 1]

            if self.is_compatible(a, b) is False:
                return 1
            else:
                return 0

        else:
            a, b = row[start], row[start + 1]

            if self.is_compatible(a, b) is False:
                if a % 2 == 0:
                    a1 = a + 1
                else:
                    a1 = a - 1

                if b % 2 == 0:
                    b1 = b + 1
                else:
                    b1 = b - 1

                pos_a1, pos_b1 = positions[a1], positions[b1]

                if (pos_a1 % 2 == 0 and row[pos_a1 + 1] == b1) or (pos_a1 % 2 == 1 and row[pos_a1 - 1] == b1):
                    c = 1
                else:
                    c = 2

                    if pos_a1 % 2 == 0:
                        temp = row[pos_a1 + 1]
                        row[pos_a1 + 1] = b1
                        row[pos_b1] = temp

                        positions[b1] = pos_a1 + 1
                        positions[temp] = pos_b1
                    else:
                        temp = row[pos_a1 - 1]
                        row[pos_a1 - 1] = b1
                        row[pos_b1] = temp

                        positions[b1] = pos_a1 - 1
                        positions[temp] = pos_b1

                temp = b
                row[start + 1] = a1
                row[pos_a1] = temp

                positions[b] = pos_a1
                positions[a1] = start + 1

                return c + self.min_swaps(row, positions, start + 2)
            else:
                return self.min_swaps(row, positions, start + 2)


    def minSwapsCouples(self, row):
        positions = [0]*len(row)

        for idx in range(len(row)):
            positions[row[idx]] = idx

        return self.min_swaps(row, positions, 0)


sol = Solution()
print sol.minSwapsCouples([28,4,37,54,35,41,43,42,45,38,19,51,49,17,47,25,12,53,57,20,2,1,9,27,31,55,32,48,59,15,14,8,3,7,58,23,10,52,22,30,6,21,24,16,46,5,33,56,18,50,39,34,29,36,26,40,44,0,11,13])
