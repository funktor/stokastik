import collections

class Solution(object):
    def sequenceReconstruction(self, org, seqs):

        next_num, org_num = collections.defaultdict(), collections.defaultdict()
        index_map = collections.defaultdict()

        for idx in range(len(org) - 1):
            index_map[org[idx]] = idx
            org_num[org[idx]] = org[idx + 1]

        index_map[org[len(org) - 1]] = len(org) - 1
        org_num[org[len(org) - 1]] = len(org) + 1

        for seq in seqs:
            for idx in range(len(seq) - 1):
                if seq[idx] not in next_num or \
                        (seq[idx + 1] in index_map and next_num[seq[idx]] in index_map and
                                 index_map[next_num[seq[idx]]] > index_map[seq[idx + 1]]):
                    next_num[seq[idx]] = seq[idx + 1]

        for seq in seqs:
            if len(seq) > 0 and seq[len(seq) - 1] not in next_num:
                next_num[seq[len(seq) - 1]] = len(org) + 1

        if len(org_num) != len(next_num) or len(next_num) == 0:
            return False

        for key, val in org_num.iteritems():
            if key not in next_num or next_num[key] != val:
                return False

        return True

sol = Solution()
print sol.sequenceReconstruction([1,2,3], [[1,2], [2,1], [1,3], [2,3]])
