class Solution(object):

    def distribute_spaces(self, spaces, lines, maxWidth, curr_running_len, is_last_line=0):

        remaining = maxWidth - curr_running_len + 1

        if is_last_line == 1:
            spaces[-1] += remaining - 1
            return spaces

        if len(spaces) == 1:
            spaces[0] += remaining - 1
            return spaces
        else:
            spaces.pop()

        num_in_between = len(lines) - 1

        avg_space = int(remaining / num_in_between)

        for idx in reversed(range(len(spaces))):
            spaces[idx] += avg_space
            remaining -= avg_space
            num_in_between -= 1

            if num_in_between == 0:
                break

            avg_space = int(remaining / num_in_between)

        return spaces


    def fullJustify(self, words, maxWidth):
        if len(words) == 0:
            return []
        lines = []
        spaces = []

        curr_running_len = 0

        for word in words:
            if curr_running_len + len(word) > maxWidth:
                spaces[len(spaces) - 1] = self.distribute_spaces(spaces[len(spaces) - 1], lines[len(lines) - 1],
                                                                 maxWidth, curr_running_len)

                lines += [[word]]
                spaces += [[1]]

                curr_running_len = len(word)+1
            else:
                if len(lines) == 0:
                    lines += [[word]]
                    spaces += [[1]]
                else:
                    lines[len(lines)-1].append(word)
                    spaces[len(spaces) - 1].append(1)

                curr_running_len += len(word)+1

        spaces[len(spaces) - 1] = self.distribute_spaces(spaces[len(spaces) - 1], lines[len(lines) - 1],
                                                         maxWidth, curr_running_len, is_last_line=1)

        output = []

        for idx in range(len(lines)):
            line = lines[idx]
            space = spaces[idx]

            out = ""
            for i in range(len(line)):
                if i < len(space):
                    out += line[i] + " " * space[i]
                else:
                    out += line[i]

            output.append(out)

        return output

sol = Solution()
print sol.fullJustify("".split(), 12)