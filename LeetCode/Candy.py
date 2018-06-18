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



#arr = [58,21,72,77,48,9,38,71,68,77,82,47,25,94,89,54,26,54,54,99,64,71,76,63,81,82,60,64,29,51,87,87,72,12,16,20,21,54,43,41,83,77,41,61,72,82,15,50,36,69,49,53,92,77,16,73,12,28,37,41,79,25,80,3,37,48,23,10,55,19,51,38,96,92,99,68,75,14,18,63,35,19,68,28,49,36,53,61,64,91,2,43,68,34,46,57,82,22,67,89]
arr = [1,2,4,4,3]
sol = Solution()
print sol.candy(arr)