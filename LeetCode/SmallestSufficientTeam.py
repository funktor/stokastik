class Solution(object):
    def smallestSufficientTeam(self, req_skills, people):
        heap = [(0, -1, set(), set())]
        cache = {}
        
        min_len, best_people = float("Inf"), set()
        
        while len(heap) > 0:
            cost, curr_idx, people_set, skill_set = heapq.heappop(heap)
            
            if len(skill_set) == len(req_skills):
                return list(people_set)
            
            elif len(people_set) == 0:
                for i in range(len(people)):
                    heapq.heappush(heap, (1, i, set([i]), set(people[i])))
                    cache[tuple(set(people[i]))] = 1
            
            else:
                for i in range(curr_idx+1, len(people)):
                    if len(people[i]) > 0:
                        new_people_set = people_set.copy()
                        new_people_set.add(i)
                        
                        new_skill_set = skill_set.copy()
                        new_skill_set.update(people[i])
                        
                        a = len(new_people_set) - len(new_skill_set)
                        
                        if a <= 0 and (tuple(new_skill_set) not in cache or len(new_people_set) < cache[tuple(new_skill_set)]):
                            heapq.heappush(heap, (len(new_people_set), i, new_people_set, new_skill_set))
                            cache[tuple(new_skill_set)] = len(new_people_set)
        
        return list(best_people)
            
        
