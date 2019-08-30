class Solution(object):
    def minStickers(self, stickers, target):
        word_char_dict, char_counts = {}, [0]*26
        
        for i in range(len(stickers)):
            sticker = stickers[i]
            for c in sticker:
                if c not in word_char_dict:
                    word_char_dict[c] = []
                word_char_dict[c].append(i)
        
        heap = [(0, -1, 0, char_counts)]
        cache = {(-1, str(char_counts)):0}
        
        min_stickers = float("Inf")
        
        while len(heap) > 0:
            cost, target_pos, num_stickers, c_cnts = heapq.heappop(heap)
            
            if target_pos == len(target)-1:
                min_stickers = min(min_stickers, num_stickers)
            
            else:
                char = target[target_pos+1]
                p = ord(char)-ord('a')
                
                if char not in word_char_dict:
                    return -1
                
                if c_cnts[p] == 0:
                    word_indices = word_char_dict[char]
                    for wid in word_indices:
                        c_cnts_new = c_cnts[:]
                        n_stickers = num_stickers+1
                        
                        for x in stickers[wid]:
                            c_cnts_new[ord(x)-ord('a')] += 1
                        
                        c_cnts_new[p] -= 1
                        key = (target_pos+1, str(c_cnts_new))

                        if (key not in cache or cache[key] > n_stickers) and n_stickers < min_stickers:
                            heapq.heappush(heap, (n_stickers-target_pos, target_pos+1, n_stickers, c_cnts_new))
                            cache[key] = n_stickers
                else:
                    n_stickers = num_stickers
                    c_cnts_new = c_cnts[:]
                    c_cnts_new[p] -= 1
                    key = (target_pos+1, str(c_cnts_new))

                    if (key not in cache or cache[key] > n_stickers) and n_stickers < min_stickers:
                        heapq.heappush(heap, (n_stickers-target_pos, target_pos+1, n_stickers, c_cnts_new))
                        cache[key] = n_stickers
                    
        return min_stickers
                        
        
