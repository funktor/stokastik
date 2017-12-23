class Solution(object):

    def get_str(self, w):
        if w > 0:
            return int('1' * w)
        else:
            return 0

    def get_pos(self, n, m):
        n1, m1 = str(n), str(m)
        curr_pos, w = 0, len(n1)

        for idx in range(len(m1)):
            if idx == 0:
                if int(m1[0]) <= int(n1[0]):
                    curr_pos += (int(m1[0]) - 1) * self.get_str(w) + 1
                else:
                    curr_pos += (int(n1[0]) - 1) * self.get_str(w)

                    if len(n1) > 1:
                        curr_pos += int(n1[1:]) + 1
                    else:
                        curr_pos += 1

                    curr_pos += (int(m1[0]) - int(n1[0])) * self.get_str(w - 1) + 1
            else:
                if int(m1[:idx + 1]) <= int(n1[:idx + 1]):
                    curr_pos += int(m1[idx]) * self.get_str(w) + 1
                else:
                    if int(n1[:idx]) < int(m1[:idx]):
                        curr_pos += int(m1[idx]) * self.get_str(w - 1) + 1
                    else:
                        curr_pos += int(n1[idx]) * self.get_str(w)

                        if int(n1[idx]) < int(m1[idx]):
                            if len(n1) > idx + 1:
                                curr_pos += int(n1[idx + 1:]) + 1
                            else:
                                curr_pos += 1

                        curr_pos += (int(m1[idx]) - int(n1[idx])) * self.get_str(w - 1) + 1

            w -= 1

        return curr_pos


    def findKthNumber(self, n, k):
        out = ''

        while True:
            if len(out) == 0:
                left = 1
            else:
                left = int(out + '0')

            right = int(out + '9')

            while left <= right:
                mid = (left + right) / 2
                pos = self.get_pos(n, mid)

                if pos == k:
                    return mid

                elif pos > k:
                    right = mid - 1
                else:
                    left = mid + 1

            if left - 1 > n:
                break

            out = str(left - 1)

        return int(out)

sol = Solution()
print sol.findKthNumber(1692778, 1545580)
