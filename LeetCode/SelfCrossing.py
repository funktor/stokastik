class Solution(object):
    def isSelfCrossing(self, x):
        curr_x, curr_y = 0, 0
        h = []
        
        for i in range(len(x)):
            a = (curr_x, curr_y)
            
            if i % 2 == 0:
                if i % 4 == 0:
                    curr_y += x[i]
                    j = 0
                    while 0 <= j < len(h):
                        p, q = min(h[j][0][0], h[j][1][0]), max(h[j][0][0], h[j][1][0])
                        
                        if a[1] < h[j][0][1] <= curr_y and p <= curr_x <= q:
                            return True
                        
                        if h[j][0][1] > curr_y and h[j][1][1] > curr_y:
                            h.pop(j)
                            j -= 1
                        j += 1
                        
                else:
                    curr_y -= x[i]
                    j = 0
                    while 0 <= j < len(h):
                        p, q = min(h[j][0][0], h[j][1][0]), max(h[j][0][0], h[j][1][0])
                        
                        if a[1] > h[j][0][1] >= curr_y and p <= curr_x <= q:
                            return True
                        
                        if h[j][0][1] < curr_y and h[j][1][1] < curr_y:
                            h.pop(j)
                            j -= 1
                        j += 1
                    
            else:
                if (i+1) % 4 == 0:
                    curr_x += x[i]
                    j = 0
                    while 0 <= j < len(h):
                        p, q = min(h[j][0][1], h[j][1][1]), max(h[j][0][1], h[j][1][1])
                        
                        if a[0] < h[j][0][0] <= curr_x and p <= curr_y <= q:
                            return True
                        
                        if h[j][0][0] > curr_x and h[j][1][0] > curr_x:
                            h.pop(j)
                            j -= 1
                        j += 1
                        
                else:
                    curr_x -= x[i]
                    j = 0
                    while 0 <= j < len(h):
                        p, q = min(h[j][0][1], h[j][1][1]), max(h[j][0][1], h[j][1][1])
                        
                        if a[0] > h[j][0][0] >= curr_x and p <= curr_y <= q:
                            return True
                        
                        if h[j][0][0] < curr_x and h[j][1][0] < curr_x:
                            h.pop(j)
                            j -= 1
                        j += 1
                    
            if len(h) >= 4:
                h.pop(0)
                
            b = (curr_x, curr_y)
            h.append((a, b))
            
        return False
