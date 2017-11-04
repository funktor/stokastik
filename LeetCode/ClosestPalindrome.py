class Solution(object):

    def nearby_palindromes(self, n):
        nums = list(n)

        if len(nums) == 1:
            a = int(nums[0])
            if a == 0:
                return ["9", "0", "1"]
            elif a == 9:
                return ["8", "9", "0"]
            else:
                return [str(a - 1), str(a), str(a + 1)]

        elif len(nums) == 2:
            a = int(nums[0])
            if a == 0:
                return ["99", "00", "11"]
            elif a == 9:
                return ["88", "99", "00"]
            else:
                return [str(a - 1) + str(a - 1), str(a) + str(a), str(a + 1) + str(a + 1)]

        else:
            a = int(nums[0])
            middle = nums[1:-1]

            mid_palindromes = self.nearby_palindromes(middle)

            out = []

            for mid_palin in mid_palindromes:
                if a != 0:
                    out.append(str(a - 1) + str(mid_palin) + str(a - 1))

                out.append(str(a) + str(mid_palin) + str(a))

                if a != 9:
                    out.append(str(a + 1) + str(mid_palin) + str(a + 1))

            return out

    def nearestPalindromic(self, n):
        nearby = []

        if n == "":
            return ""
        if n[0] == "1" and len(n) == 1:
            return "0"
        elif n[0] == "1" and n[1:] == "0"*(len(n)-1):
            return "9"*(len(n)-1)

        if n == "9"*len(n):
            nearby.append("1" + "0" * (len(n)-1) + "1")

        if n == "1" + "0"*(len(n)-2) + "1":
            nearby.append("9" * (len(n) - 1))

        nearby += self.nearby_palindromes(n)
        num = int(n)

        nearest = None
        nearest_dist = num

        for x in nearby:
            if x[0] != '0':
                x = int(x)
                if abs(x - num) < nearest_dist and x != num:
                    nearest_dist = abs(x - num)
                    nearest = x

        return str(nearest)

sol = Solution()
print sol.nearestPalindromic("9995")


