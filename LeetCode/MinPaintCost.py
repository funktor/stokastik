import collections

class Solution(object):
    def minCostII(self, costs):
        if len(costs) == 0 or len(costs[0]) == 0:
            return 0

        costs_map = collections.defaultdict(dict)

        for n in range(len(costs)):
            for k in range(len(costs[0])):
                if n == 0:
                    costs_map[n][k] = costs[n][k]
                else:
                    costs_map[n][k] = float("Inf")

                    for col, cost in costs_map[n - 1].items():
                        if k != col:
                            costs_map[n][k] = min(costs_map[n][k], cost + costs[n][k])

        min_cost = float("Inf")

        for col, cost in costs_map[len(costs) - 1].items():
            min_cost = min(min_cost, cost)

        return min_cost