import collections

class Solution(object):
    def minWindow(self, s, t):
        t_counts_map = collections.defaultdict(int)

        for x in t:
            t_counts_map[x] += 1

        positions = []

        for idx in range(len(s)):
            if s[idx] in t_counts_map:
                positions.append((idx, s[idx]))

        start, end = 0, 0
        min_window_size, min_start, min_end = float("Inf"), -1, -1

        run_counts_map = collections.defaultdict(int)

        while start <= end:
            flag = True

            for k, v in t_counts_map.iteritems():
                if k not in run_counts_map or run_counts_map[k] < v:
                    flag = False
                    break

            if flag:
                a = positions[start]
                b = positions[end - 1]

                if b[0] - a[0] + 1 < min_window_size:
                    min_window_size = b[0] - a[0] + 1
                    min_start = a[0]
                    min_end = b[0]

                run_counts_map[positions[start][1]] -= 1
                start += 1

            else:
                if end < len(positions):
                    run_counts_map[positions[end][1]] += 1
                    end += 1
                else:
                    if start < len(positions):
                        run_counts_map[positions[start][1]] -= 1
                    start += 1

        if min_start == -1:
            return ""

        return s[min_start:min_end + 1]


sol = Solution()
print sol.minWindow("a", "a")