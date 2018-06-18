import collections

class Solution(object):
    def uniqueLetterString(self, S):
        if len(S) == 0:
            return 0

        chr_pos_before, last_char_pos = collections.defaultdict(int), collections.defaultdict(int)

        for idx in range(len(S)):
            char = S[idx]

            if char in last_char_pos:
                chr_pos_before[idx] = last_char_pos[char]
                last_char_pos[char] = idx
            else:
                last_char_pos[char] = idx

        unique = collections.defaultdict(int)
        res = 0

        for end in range(len(S)):
            if end == 0:
                unique[end] = 1
            else:
                count = unique[end - 1]

                if end in chr_pos_before:
                    last_pos = chr_pos_before[end]
                else:
                    last_pos = -1

                if last_pos == -1:
                    count += end

                else:
                    if last_pos in chr_pos_before:
                        last_last_pos = chr_pos_before[last_pos]
                        count += end - 2 * last_pos + last_last_pos - 1
                    else:
                        count += end - 2 * last_pos - 2

                count += 1
                unique[end] = count

            res += unique[end]

        return res


sol = Solution()
print sol.uniqueLetterString("OLOFGLDWPFRRMWUVYPEIFWJTODVNAZTTUJRHKECFNOXMEDPBKAZPLARKDUZEBUXAUCMVKJQERDLDELPXSOJXQFGNWBVBUTKQGSOOZDIXTGDQQIOSLPBPQGUAGMFNVZYFKINLRHIQTENOCPSYENUSIFKPLMNQVLIESPUNOLNFPBFTVTLEIFZBPQZCKBJZHRWOKRJIQIPNZYNYRSNTYEKLNEQKRTIZOMINZXVAFOKWSKNNIDELMGUOLVRTOIBXGPCNQZOVAHXAKQYKDFZLLZDQCISKYIAQMKOWXZWBWGNIZQIXAQYPJBBPDXZTGPYDDCRTEMWCBKVXQYMWPUKDGZFWRMBQTSJMXZZJUARNWTUCZQBFUFOCYYEWLAAJKPTWDOXPUDBVIIHQFTZWIPLMETDNAWZLLWPMJAFNHHGZQMTWBCBQBEKBWKPDAWHDHNCEDZVESCMQNCTVSOFMONTVBAJYWMAWYNYZUCWIUFQZIIURZPMYDWPTRDLIOLRCBJNSOUSTVSWWHBZYOGOIALBVAYEFDFWVHHGWWLCKUMRZRJCXUXZQXCIFYIRYNYBRWGZYQISARLNFLFZYUXMEPWNEYUNYJREMKZGBLSFSRTMOMDYDESHLDQXMWIKJTFNUPBCWHFCIPZVUCIEHPELKMTNUTTECTQZEAESWRITRFRRCHKUQSWCGSUOSHKXVTHZJJCXFGTCEZGXBLHKDGUBHEMPNAOOSSYYPEWIHCCBZBKDJJXBQVNZQYCDLWMRJJFYKUITKZFHCHUAMBDAGICTMJATWNTTPCENRAOWHZMLGFVXCYAMFONUPLDRRNVNEBTZQXDJONGAPKTGMIYTQIQSEIZONPITNKNFZWUENDMVXHBSOBIDNQWHPLOLAHPIJAFZJISTXTNFDCXXKRRUXBPHMJQZANEBMIOYQMYQWAYUNOKWBVMCKGPMCXOEQTADAFKBXNJVKNHMJTLZKIQXIROBJCPSIKCYHVMOEHSOMPFTKXXFKMNEQ")