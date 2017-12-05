import collections

class Solution(object):
    def k_sums(self, nums, k):
        out = []

        curr_sums = 0
        for idx in range(k):
            curr_sums += nums[idx]

        out.append((0, k - 1, curr_sums))

        start, end = 1, k
        while end < len(nums):
            curr_sums = curr_sums - nums[start - 1] + nums[end]
            out.append((start, end, curr_sums))
            start += 1
            end += 1

        return out

    def maxSumOfThreeSubarrays(self, nums, k):
        ksums = self.k_sums(nums, k)

        suffix_max_sums = collections.defaultdict(dict)
        max_starts = collections.defaultdict(dict)

        length = 1
        while length <= 3:
            for idx in reversed(range(len(ksums) - length + 1)):
                if ksums[idx][0] + 1 not in suffix_max_sums[length]:
                    a = 0
                else:
                    a = suffix_max_sums[length][ksums[idx][0] + 1]

                if ksums[idx][0] + 1 not in max_starts[length]:
                    c = []
                else:
                    c = max_starts[length][ksums[idx][0] + 1]

                if length == 1:
                    if a > ksums[idx][2]:
                        suffix_max_sums[length][ksums[idx][0]] = a
                        max_starts[length][ksums[idx][0]] = c
                    else:
                        suffix_max_sums[length][ksums[idx][0]] = ksums[idx][2]
                        max_starts[length][ksums[idx][0]] = [ksums[idx][0]]
                else:
                    if ksums[idx][1] + 1 not in suffix_max_sums[length - 1]:
                        b = 0
                    else:
                        b = suffix_max_sums[length - 1][ksums[idx][1] + 1] + ksums[idx][2]

                    if ksums[idx][1] + 1 not in max_starts[length - 1]:
                        d = []
                    else:
                        d = [ksums[idx][0]] + max_starts[length - 1][ksums[idx][1] + 1]

                    if a > b:
                        suffix_max_sums[length][ksums[idx][0]] = a
                        max_starts[length][ksums[idx][0]] = c
                    else:
                        suffix_max_sums[length][ksums[idx][0]] = b
                        max_starts[length][ksums[idx][0]] = d

            length += 1

        return max_starts[3][0]