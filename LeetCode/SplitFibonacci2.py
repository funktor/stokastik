class Solution(object):
    def splitIntoFibonacci(self, S):
        str1 = ""
        end1 = len(S)-2 if S[0] != '0' else 1
        
        for i in range(end1):
            str1 += S[i]
            str2 = ""
            end2 = len(S)-1 if S[i+1] != '0' else i+2
            
            for j in range(i+1, end2):
                str2 += S[j]
                int_c = int(str1) + int(str2)
                
                k, vals = j+1, []
                while k < len(S):
                    l = len(str(int_c))
                    if int_c <= (1<<31)-1 and k+l <= len(S) and int(S[k:k+l]) == int_c:
                        vals.append(int_c)
                    else:
                        break
                    
                    k += l
                    
                    if len(vals) < 2:
                        int_c = vals[-1] + int(str2)
                    else:
                        int_c = vals[-1] + vals[-2]
                
                if k == len(S):
                    return [int(str1)] + [int(str2)] + vals
        
        return []
