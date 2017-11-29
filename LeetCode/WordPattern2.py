class Solution(object):
    def is_match(self, pattern, str, bijection_map):
        if len(str) == 0 and len(pattern) > 0:
            return False

        if len(pattern) == 0:
            if len(str) > 0:
                return False
            return True

        if pattern[0] in bijection_map:
            str_map = bijection_map[pattern[0]]
            out = len(str_map) <= len(str) and str[:len(str_map)] == str_map

            if out:
                return self.is_match(pattern[1:], str[len(str_map):], bijection_map)

            return False

        else:
            bijection_map[pattern[0]] = ''
            i, out = 0, False

            while out is False and i + 1 <= len(str):
                bijection_map[pattern[0]] += str[i]
                out = self.is_match(pattern[1:], str[i + 1:], bijection_map)
                i += 1

            q = [x for x in bijection_map.values()]

            if out is False or len(q) != len(set(q)):
                bijection_map.pop(pattern[0])
                return False

            return self.is_match(pattern[1:], str[i:], bijection_map)

    def wordPatternMatch(self, pattern, str):
        return self.is_match(pattern, str, {})