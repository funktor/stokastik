from string import ascii_lowercase

class Solution(object):
    def removeDuplicateLetters(self, s):
        if len(s) == 0:
            return ''
        
        char_last_pos = {s[i]:i for i in range(len(s))}
        
        output_str, curr_out, visited, carry = [], [], set(), []
        
        for i in range(len(s)):
            if char_last_pos[s[i]] != i:
                if s[i] not in visited:
                    curr_out.append(s[i])
                    
            elif s[i] not in visited:
                curr_out.append(s[i])
                curr_out = carry + curr_out
                
                h, m, carry = [], -1, []
                for char in sorted(set(curr_out)): 
                    for j in range(m+1, len(curr_out)):
                        if curr_out[j] == char:
                            if curr_out[j] <= s[i]:
                                output_str.append(curr_out[j])
                                visited.add(curr_out[j])
                            else:
                                carry.append(curr_out[j])
                            m = j
                            break
                            
                curr_out = []
        
        return ''.join(output_str)
