class Solution(object):
    def removeKdigits(self, num, k):
        if len(num) == 0:
            return "0"

        for m in range(k):
            flag = False

            for idx in range(1, len(num)):
                if num[idx - 1] > num[idx]:
                    num = num[:idx - 1] + num[idx:]
                    flag = True
                    break

            if flag is False:
                num = num[:len(num) - 1]

            start = 0

            while start < len(num):
                if num[start] != "0":
                    break

                start += 1

            if start == len(num):
                return "0"

            num = num[start:]

        return num

sol = Solution()
print sol.removeKdigits("32", 3)
