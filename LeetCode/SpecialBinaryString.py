class Solution(object):
    def is_greater(self, str1, str2):
        for idx in range(len(str1)):
            if str1[idx] == '1' and str2[idx] == '0':
                return 1
            elif str1[idx] == '0' and str2[idx] == '1':
                return -1

        return 1

    def sort_bin_str(self, str_arr, left, right):
        if left == right:
            return [str_arr[left]]

        mid = int((left + right) / 2)

        a = self.sort_bin_str(str_arr, left, mid)
        b = self.sort_bin_str(str_arr, mid + 1, right)

        i, j = 0, 0
        c = []

        while i < len(a) and j < len(b):
            w = self.is_greater(a[i] + b[j], b[j] + a[i])

            if w == 1:
                c.append(a[i])
                i += 1
            else:
                c.append(b[j])
                j += 1

        if i < len(a):
            for idx in range(i, len(a)):
                c.append(a[idx])

        if j < len(b):
            for idx in range(j, len(b)):
                c.append(b[idx])

        return c

    def make_largest(self, mystr, left, right):
        if right - left >= 2:
            components, start = [], left

            while start <= right:
                special = tuple()

                count_0, count_1 = 0, 0
                end = start

                while end <= right:
                    if mystr[end] == '0' and count_1 == count_0 + 1:
                        special = (start, end)
                        break
                    else:
                        if mystr[end] == '0':
                            count_0 += 1
                        else:
                            count_1 += 1

                        end += 1

                if len(special) > 0:
                    if special[0] == left and special[1] == right:
                        start += 1
                    else:
                        if len(components) > 0 and components[-1][-1][1] == special[0] - 1:
                            components[-1] += [special]
                        else:
                            components.append([special])

                        start = end + 1
                else:
                    start += 1

            for idx in range(len(components)):
                component = components[idx]
                x, y = component[0][0], component[-1][1]

                w = []

                for a, b in component:
                    self.make_largest(mystr, a, b)
                    w.append(''.join(mystr[a: b + 1]))

                w = self.sort_bin_str(w, 0, len(w) - 1)

                mystr[x: y + 1] = list(''.join(w))

    def makeLargestSpecial(self, S):
        mystr = list(S)

        self.make_largest(mystr, 0, len(mystr) - 1)

        return ''.join(mystr)