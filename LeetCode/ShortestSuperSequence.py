import collections

class Solution(object):
    def shortestCommonSupersequence(self, str1, str2):
        cache = {}
        arr = [-1]*26
        
        for i in range(-1, len(str1)):
            if i not in cache:
                cache[i] = {}
                
            for j in range(-1, len(str2)):
                if i == -1 and j == -1:
                    cache[i][j] = (-1, -1, arr[:], 0)
                    
                elif i == -1:
                    if j == 0:
                        a = arr[:]
                        a[ord(str2[0])-ord('a')] = 0
                        cache[i][j] = (-1, 0, a, 1)
                    else:
                        a = arr[:]
                        a[ord(str2[j])-ord('a')] = j
                        cache[i][j] = (-1, j, a, j+1)
                        
                elif j == -1:
                    if i == 0:
                        a = arr[:]
                        a[ord(str1[0])-ord('a')] = 0
                        cache[i][j] = (0, -1, a, 1)
                    else:
                        a = arr[:]
                        a[ord(str1[i])-ord('a')] = i
                        cache[i][j] = (i, -1, a, i+1)
                        
                else:
                    min_len, best_a, best_b, best_c = float("Inf"), -1, -1, []
                    
                    if str1[i] == str2[j]:
                        a, b, c, d = cache[i-1][j-1]
                        
                        min_len = d+1
                        best_a, best_b, best_c = d, d, c[:]

                        best_c[ord(str1[i])-ord('a')] = d
                        best_c[ord(str2[j])-ord('a')] = d
                                
                    else:
                        a1, b1, c1, d1 = cache[i][j-1]
                        a2, b2, c2, d2 = cache[i-1][j]
                        
                        
                        if d1 < d2:
                            x = c1[ord(str2[j])-ord('a')]

                            if x > b1:
                                if d1 < min_len:
                                    min_len = d1
                                    best_a, best_b, best_c = a1, x, c1[:]
                            else:
                                if d1+1 < min_len:
                                    min_len = d1+1
                                    best_a, best_b, best_c = a1, d1, c1[:]
                                    best_c[ord(str2[j])-ord('a')] = d1
                        else:
                            x = c2[ord(str1[i])-ord('a')]

                            if x > a2:
                                if d2 < min_len:
                                    min_len = d2
                                    best_a, best_b, best_c = x, b2, c2[:]
                            else:
                                if d2+1 < min_len:
                                    min_len = d2+1
                                    best_a, best_b, best_c = d2, b2, c2[:]
                                    best_c[ord(str1[i])-ord('a')] = d2
                    
                    cache[i][j] = (best_a, best_b, best_c, min_len)
        
        w = ""
        i, j = len(str1)-1, len(str2)-1
        while True:
            if i == -1 and j == -1:
                break
            elif i == -1:
                w = str2[:j+1] + w
                break
            elif j == -1:
                w = str1[:i+1] + w
                break
            else:
                q, d = cache[i][j][2], cache[i][j][3]
                for t in range(len(q)):
                    v = q[t]
                    k = chr(97+t)
                    
                    if v == d-1:
                        w = k + w
                        if str1[i] == str2[j]:
                            i -= 1
                            j -= 1
                        else:
                            if str1[i] == k:
                                i -= 1
                            else:
                                j -= 1
                        break
        
        return w
