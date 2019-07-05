class Solution(object):
    def minHeightShelves(self, books, shelf_width):
        cache = {}
        
        for i in range(len(books)):
            width, height = 0, books[i][1]
            cache[i] = float("Inf")
            
            for j in reversed(range(i+1)):
                height = max(height, books[j][1])
                width += books[j][0]
                
                if width <= shelf_width:
                    if j == 0:
                        cache[i] = min(cache[i], height)
                    else:
                        cache[i] = min(cache[i], cache[j-1] + height)
                else:
                    break
        
        return cache[len(books)-1]
        
