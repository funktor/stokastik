import collections

class Solution(object):
    def longestConsecutive(self, nums):
        buckets, resolve_map = collections.defaultdict(list), collections.defaultdict(int)

        last_bucket_num = 0

        for num in nums:
            if num not in buckets:
                if num - 1 in buckets or num + 1 in buckets:
                    if num - 1 in buckets:
                        b = buckets[num - 1]
                        buckets[num] += b
                    if num + 1 in buckets:
                        b = buckets[num + 1]
                        buckets[num] += b

                    if len(buckets[num]) > 1:
                        resolve_map[buckets[num][1]] = buckets[num][0]
                        buckets[num].pop()
                else:
                    last_bucket_num += 1
                    buckets[num] = [last_bucket_num]

        for k in resolve_map:
            v = resolve_map[k]

            while v in resolve_map:
                v = resolve_map[v]

            resolve_map[k] = v

        bucket_counts = collections.defaultdict(int)

        for k, v in buckets.items():
            if v[0] in resolve_map:
                bucket_counts[resolve_map[v[0]]] += 1
            else:
                bucket_counts[v[0]] += 1

        max_count = 0

        for k, v in bucket_counts.items():
            max_count = max(max_count, v)

        return max_count

sol = Solution()
print sol.longestConsecutive([1,2,0,1])