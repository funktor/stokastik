import collections

class Solution(object):
    def removeBoxes(self, boxes):
        config = []
        curr_count, curr_val = 1, boxes[0]
        for i in range(1, len(boxes)):
            if boxes[i] == boxes[i-1]:
                curr_count += 1
            else:
                config.append((curr_val, curr_count))
                curr_val = boxes[i]
                curr_count = 1
        
        config.append((curr_val, curr_count))
        cached = collections.defaultdict(dict)
        
        for length in range(1, len(config)+1):
            for i in range(len(config)-length+1):
                j = i + length - 1
                if length == 1:
                    cached[i][j] = (config[i][1]**2, [config[i]])
                    
                else:
                    best_cost, best_config = -1, []
                    
                    for k in range(i+1, j+1):
                        a, b = cached[i][k-1], cached[k][j]
                        
                        if len(a[1]) == 1 or len(b[1]) == 1:
                            if len(a[1]) == 1:
                                new_cost, new_count, idx, conf = b[0], a[1][0][1], -1, []
                                
                                for l in range(len(b[1])):
                                    x, y = b[1][l]
                                    if x == a[1][0][0]:
                                        new_count = y + a[1][0][1]
                                        new_cost = a[0] + b[0] - y**2 - a[1][0][1]**2 + new_count**2
                                        conf = [(a[1][0][0], new_count)] + b[1][l+1:]
                                        break
                            else:
                                new_cost, new_count, idx, conf = a[0], b[1][0][1], -1, []
                                
                                for l in reversed(range(len(b[1]))):
                                    x, y = a[1][l]
                                    if x == b[1][0][0]:
                                        new_count = y + b[1][0][1]
                                        new_cost = a[0] + b[0] - y**2 - b[1][0][1]**2 + new_count**2
                                        conf = a[1][:l] + [(b[1][0][0], new_count)]
                                        break
                    
                            p = a[0] + b[0]
                            h = max(new_cost, p)
                            
                            if h > best_cost:
                                if new_cost == h:
                                    best_cost = new_cost
                                    best_config = conf

                                else:
                                    best_cost = p
                                    best_config = a[1] + b[1]
                                
                    cached[i][j] = (best_cost, best_config)
                    
        return cached[0][len(config)-1][0]
