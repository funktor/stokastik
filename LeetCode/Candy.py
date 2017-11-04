class Solution(object):

    def assign_candies(self, ratings):
        mydata = zip(ratings, range(len(ratings)))
        mydata = sorted(mydata, key=lambda k: k[0])

        num_candies = [0]*len(ratings)
        num_candies[mydata[0][1]] = 1

        for idx in range(1, len(mydata)):
            curr_pos = mydata[idx][1]

            if curr_pos == len(mydata)-1:
                prev_pos = curr_pos - 1
                if ratings[curr_pos] > ratings[prev_pos]:
                    num_candies[curr_pos] = num_candies[prev_pos] + 1
                else:
                    num_candies[curr_pos] = 1
            elif curr_pos == 0:
                next_pos = curr_pos + 1
                if ratings[curr_pos] > ratings[next_pos]:
                    num_candies[curr_pos] = num_candies[next_pos] + 1
                else:
                    num_candies[curr_pos] = 1
            else:
                prev_pos = curr_pos - 1
                next_pos = curr_pos + 1

                if ratings[curr_pos] > ratings[prev_pos] and ratings[curr_pos] > ratings[next_pos]:
                    num_candies[curr_pos] = max(num_candies[prev_pos], num_candies[next_pos]) + 1
                elif ratings[prev_pos] < ratings[curr_pos] <= ratings[next_pos]:
                    num_candies[curr_pos] = num_candies[prev_pos] + 1
                elif ratings[next_pos] < ratings[curr_pos] <= ratings[prev_pos]:
                    num_candies[curr_pos] = num_candies[next_pos] + 1
                else:
                    num_candies[curr_pos] = 1

        return num_candies


    def candy(self, ratings):
        if len(ratings) == 0:
            return 0
        if len(ratings) == 1:
            return 1

        out = self.assign_candies(ratings)

        return sum(out)


arr = [1,2,4,4,3]
sol = Solution()
print sol.candy(arr)