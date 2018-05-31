import collections


class Solution(object):
    def take_bus(self, route_set, rr_map, curr_route, dest):
        queue = [(curr_route, 1)]
        visited = set([curr_route])

        while len(queue) > 0:
            route, depth = queue.pop()

            if dest in route_set[route]:
                return depth
            else:
                possible_routes = rr_map[route]

                for rt in possible_routes:
                    if rt not in visited:
                        visited.add(rt)
                        queue.insert(0, (rt, depth + 1))

        return len(route_set) + 1

    def numBusesToDestination(self, routes, S, T):
        if S == T:
            return 0

        rr_map = collections.defaultdict(set)

        route_set = []

        for route in routes:
            route_set.append(set(route))

        for i in range(len(route_set)):
            for j in range(len(route_set)):
                if i != j:
                    x, y = route_set[i], route_set[j]
                    if len(x.intersection(y)) > 0:
                        rr_map[i].add(j)
                        rr_map[j].add(i)

        source_routes = []
        for idx in range(len(route_set)):
            x = route_set[idx]

            if S in x:
                source_routes.append(idx)

        min_num_buses = len(route_set) + 1

        for s_route in source_routes:
            min_num_buses = min(min_num_buses, self.take_bus(route_set, rr_map, s_route, T))

        if min_num_buses == len(route_set) + 1:
            return -1

        return min_num_buses